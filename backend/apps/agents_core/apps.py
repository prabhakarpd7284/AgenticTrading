"""AppConfig for agents_core.

Loads strategy plugins at boot. Two paths, applied in order:

  1. Entry-point auto-load (`alphadesk.strategies` group in pyproject.toml).
     This is the production path — requires `uv pip install -e .` after
     adding a new plugin so the entry-points database is rebuilt.

  2. Explicit `STRATEGY_REGISTRY_EXTRA` list in settings. Registers a
     plugin by its import path (`module.path:ClassName`) regardless of
     whether it's been pip-installed yet. Lets new plugins go live
     immediately during dev without a venv reinstall — the exact pain
     point we just hit with strategy_vertical_spread.
"""
from __future__ import annotations

import importlib
import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgentsCoreConfig(AppConfig):
    name = "apps.agents_core"

    def ready(self) -> None:
        from django.conf import settings
        from apps.agents_core.registry import strategy_registry

        # 1) Entry-point auto-load (only if enabled)
        if settings.ALPHADESK.get("STRATEGY_REGISTRY_AUTOLOAD", True):
            strategy_registry.load_entry_points()

        # 2) Explicit extras — registers any plugin missing from step 1.
        # We DON'T overwrite an already-registered name (entry-points win)
        # because that's the deliberate prod path. Extras are a dev fallback.
        for spec in settings.ALPHADESK.get("STRATEGY_REGISTRY_EXTRA", []) or []:
            try:
                mod_path, cls_name = spec.split(":", 1)
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)
            except Exception as e:  # noqa: BLE001
                logger.exception("strategy_extra.import_failed spec=%s err=%s", spec, e)
                continue
            try:
                instance = cls()
            except Exception as e:  # noqa: BLE001
                logger.exception("strategy_extra.instantiate_failed spec=%s err=%s",
                                  spec, e)
                continue
            name = getattr(instance, "name", None)
            if not name:
                logger.warning("strategy_extra.missing_name_attr spec=%s", spec)
                continue
            if name in strategy_registry.names():
                logger.info("strategy_extra.skip_already_registered name=%s", name)
                continue
            strategy_registry.register(name, instance)
            logger.info("strategy_extra.registered name=%s from=%s", name, spec)
