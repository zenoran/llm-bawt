"""Nextcloud Talk bot integration."""

from .config import NextcloudBot, NextcloudBotConfig
from .manager import NextcloudBotManager, get_nextcloud_manager

__all__ = [
    'NextcloudBot',
    'NextcloudBotConfig',
    'NextcloudBotManager',
    'get_nextcloud_manager',
]
