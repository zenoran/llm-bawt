"""Unified media generation module for llm-bawt.

Provides a provider-agnostic API for generating video and image media,
with filesystem-based blob storage and PostgreSQL metadata tracking.
"""

from .assets import MediaAsset, MediaAssetStore, new_asset_id
from .serializers import (
    asset_to_attachment_dict,
    asset_to_upload_response_dict,
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
    "asset_to_attachment_dict",
    "asset_to_upload_response_dict",
    "get_media_store",
    "new_asset_id",
    "reset_media_store",
]
