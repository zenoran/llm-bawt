"""MCP tool for agent image generation via Grok Imagine (xAI).

One tool — kept deliberately small:

    generate_image  — text->image (and image+text->image for iteration)
                      via ``GrokMediaClient``, stored in the content-addressed
                      media store, returned so the CALLING AGENT sees the image
                      inline (to critique/iterate) and the chat renders a
                      clickable thumbnail.

Design (see TASK-599)
---------------------

This mirrors the Playwright-screenshot path, NOT the Studio
``/v1/media/generations`` route:

- The tool returns a FastMCP ``Image`` content block. That block is what the
  model actually *sees* — enabling "make it warmer"-style iteration — and is
  also what the claude-code bridge persists to the media store + stamps on the
  ``TOOL_END`` event so the chat shows a clickable thumbnail (TASK-483).
- The tool ALSO stores the *identical* raw bytes via :func:`get_media_store`.
  Because :class:`MediaStore` dedups on the post-normalization sha256, the
  bridge's re-upload of the same bytes resolves to the SAME asset — one asset,
  no duplication — while giving the agent a stable ``asset_id`` to reference.
- Iteration: pass ``reference_asset_id`` (an ``asset_id`` from a previous
  ``generate_image`` result). The tool loads that asset as the ``source_image``
  so Grok refines the prior image instead of starting from scratch.

Imported by ``server.py`` so registration happens on startup.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import Image

from llm_bawt.media import get_media_store
from llm_bawt.media.clients import GrokMediaClient

from .server import mcp

logger = logging.getLogger(__name__)

# xAI Grok Imagine image model (auto-selected — the caller never picks it).
_IMAGE_MODEL = "grok-imagine-image"

# Source tag recorded on the stored asset (one of media.store.ALLOWED_SOURCES).
_SOURCE = "tool_generated"


def _sniff_image_format(raw: bytes) -> str:
    """Best-effort image subtype from magic bytes ('png' | 'jpeg' | 'webp' | 'gif').

    Only used for the declared MIME hint and the FastMCP ``Image`` format; the
    media store re-derives the real format from the bytes via Pillow regardless,
    so a wrong guess is cosmetic, never corrupting.
    """
    if raw[:8].startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"


@mcp.tool(name="generate_image", structured_output=False)
async def generate_image(
    prompt: str,
    reference_asset_id: str | None = None,
    aspect_ratio: str = "1:1",
    resolution: str = "1k",
    user_id: str = "nick",
) -> list:
    """Generate an image from a text prompt with Grok Imagine (xAI).

    The generated image is returned inline so you SEE it in this turn — read it,
    critique it, and iterate. It also appears as a clickable thumbnail in the
    chat automatically.

    To REFINE a previous image (e.g. the user says "make it warmer" / "add a
    dog"), call this again with ``reference_asset_id`` set to the ``asset_id``
    from the earlier result and a prompt describing the change — Grok will edit
    the prior image (image+text -> image) rather than start over.

    Args:
        prompt: What to draw, or how to change the referenced image.
        reference_asset_id: Optional ``asset_id`` of a prior generated image to
            use as the reference/base for this generation (enables iteration).
        aspect_ratio: ``"1:1"`` (default), ``"16:9"``, ``"9:16"``, ``"4:3"``, etc.
        resolution: ``"1k"`` (default, faster) or ``"2k"`` (higher detail).
        user_id: Owner scope for the stored asset (defaults to the primary user).

    Returns:
        A list of ``[<image>, <info>]`` where ``<info>`` is a dict containing
        ``asset_id`` (pass it back as ``reference_asset_id`` to iterate),
        ``revised_prompt`` (if xAI rewrote the prompt), and the image ``urls``.
        On failure, returns ``[{"error": "..."}]``.
    """
    logger.info(
        "MCP tool invoked: generate_image prompt=%r reference_asset_id=%s ar=%s res=%s",
        prompt[:120], reference_asset_id, aspect_ratio, resolution,
    )

    store = get_media_store()

    # 1) Resolve the reference image (iteration) into a data URL, if asked.
    source_image: str | None = None
    if reference_asset_id:
        try:
            source_image = store.read_preview_as_data_url(reference_asset_id)
        except Exception as e:  # unknown id, missing blob, backend down
            logger.warning(
                "generate_image: reference_asset_id=%s unreadable: %s",
                reference_asset_id, e,
            )
            return [{
                "error": (
                    f"reference_asset_id {reference_asset_id!r} could not be loaded "
                    f"({e}). Omit it to generate from scratch, or use a valid "
                    f"asset_id from a previous generate_image result."
                ),
            }]

    # 2) Generate via Grok (image gen is synchronous — URL returned inline).
    client = GrokMediaClient()
    try:
        result = await client.generate(
            prompt=prompt,
            media_type="image",
            model=_IMAGE_MODEL,
            source_image=source_image,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        if result.status != "completed" or not result.media_url:
            err = result.error or "generation did not complete"
            logger.warning("generate_image: xAI failed: %s", err)
            return [{"error": f"Image generation failed: {err}"}]

        # 3) Download the finished image bytes from xAI's (temporary) URL.
        raw = await client.download(result.media_url)
    except Exception as e:
        logger.exception("generate_image: generation/download error")
        return [{"error": f"Image generation error: {e}"}]
    finally:
        await client.close()

    if not raw:
        return [{"error": "Image generation returned no bytes"}]

    fmt = _sniff_image_format(raw)

    # 4) Store the raw bytes (content-addressed; deduped). The bridge re-uploads
    #    these SAME bytes for the chat thumbnail and dedups to this same asset.
    try:
        asset = store.upload(
            raw_bytes=raw,
            original_mime=f"image/{fmt}",
            source=_SOURCE,
            owner_user_id=user_id,
        )
    except Exception as e:
        logger.exception("generate_image: media store upload failed")
        return [{"error": f"Generated image but failed to store it: {e}"}]

    info = {
        "asset_id": asset.id,
        "prompt": prompt,
        "revised_prompt": result.revised_prompt,
        "reference_asset_id": reference_asset_id,
        "urls": {
            "thumb": f"/v1/uploads/{asset.id}/thumb",
            "preview": f"/v1/uploads/{asset.id}/preview",
            "original": f"/v1/uploads/{asset.id}",
        },
        "hint": (
            "The image is shown above. To refine it, call generate_image again "
            f"with reference_asset_id='{asset.id}' and a prompt describing the change."
        ),
    }

    # 5) Return [image, info]: the Image block is what the model sees (and what
    #    the bridge persists for the chat thumbnail); the dict carries the
    #    asset_id the agent needs to iterate.
    return [Image(data=raw, format=fmt), info]
