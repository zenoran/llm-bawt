"""Unified media generation module for llm-bawt.

Provides a provider-agnostic API for generating video and image media,
with filesystem-based blob storage and PostgreSQL metadata tracking.
"""

from .asset_kinds import UnsupportedVariant, kind_for_mime
from .assets import MediaAsset, MediaAssetStore, new_asset_id
from .serializers import (
    asset_to_attachment_dict,
    asset_to_upload_response_dict,
    public_urls,
)
from .store import (
    MediaAssetNotFound,
    MediaStore,
    get_media_store,
    reset_media_store,
)

__all__ = [
    "MediaAsset",
    "MediaAssetStore",
    "MediaAssetNotFound",
    "MediaStore",
    "UnsupportedVariant",
    "asset_to_attachment_dict",
    "asset_to_upload_response_dict",
    "get_media_store",
    "kind_for_mime",
    "new_asset_id",
    "public_urls",
    "reset_media_store",
]
