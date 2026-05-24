"""Unified media generation module for llm-bawt.

Provides a provider-agnostic API for generating video and image media,
with filesystem-based blob storage and PostgreSQL metadata tracking.
"""

from .assets import MediaAsset, MediaAssetStore, new_asset_id

__all__ = ["MediaAsset", "MediaAssetStore", "new_asset_id"]
