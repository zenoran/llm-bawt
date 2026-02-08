"""Nextcloud bot manager singleton."""

from typing import Optional
from .config import NextcloudBot, NextcloudBotConfig


class NextcloudBotManager:
    """Singleton manager for Nextcloud bots."""

    _instance: Optional['NextcloudBotManager'] = None

    def __init__(self):
        self.config = NextcloudBotConfig()

    @classmethod
    def get_instance(cls, bots_yaml_path=None) -> 'NextcloudBotManager':
        """Get singleton instance.
        
        Args:
            bots_yaml_path: Ignored - kept for backwards compatibility.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def list_bots(self) -> list[NextcloudBot]:
        """List all configured Nextcloud bots."""
        self.config._check_reload()
        return list(self.config.bots.values())

    def reload(self) -> None:
        """Force reload config from disk."""
        self.config.load()

    def add_bot(
        self,
        llm_bawt_bot: str,
        nextcloud_bot_id: int,
        secret: str,
        conversation_token: str,
    ) -> NextcloudBot:
        """Add Nextcloud config to a bot."""
        return self.config.add_bot(
            llm_bawt_bot=llm_bawt_bot,
            nextcloud_bot_id=nextcloud_bot_id,
            secret=secret,
            conversation_token=conversation_token,
        )

    def remove_bot(self, llm_bawt_bot: str) -> bool:
        """Remove Nextcloud config from a bot."""
        return self.config.remove_bot(llm_bawt_bot)

    def get_bot_by_conversation(self, token: str) -> Optional[NextcloudBot]:
        """Get bot for a conversation token."""
        return self.config.get_bot_by_conversation(token)

    def get_bot(self, llm_bawt_bot: str) -> Optional[NextcloudBot]:
        """Get bot config by llm-bawt bot ID."""
        return self.config.get_bot(llm_bawt_bot)


# Global accessor
def get_nextcloud_manager() -> NextcloudBotManager:
    """Get the global Nextcloud bot manager."""
    return NextcloudBotManager.get_instance()
