"""Provider-connect subsystem.

A provider-agnostic "connect a provider" flow used by the first-run wizard and
settings UI. Each provider (GitHub, OpenAI, Anthropic, Grok, ...) is a
``ProviderAdapter`` exposing one or more auth methods (``device_oauth`` /
``api_key``). Credentials are persisted **encrypted at rest** in the
``runtime_settings`` table under key ``provider_connection:<id>``.

Imports of heavy/optional deps (cryptography, jwt) are lazy inside submodules so
this package stays out of the hot path until a provider is actually connected.
"""

from __future__ import annotations
