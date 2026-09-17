"""Developer-ops command runner — discovery + introspection helpers.

Backs the /api/v1/ops/ REST surface and the /ws/ops/ WebSocket consumer.
Browser-side dev console that lets an operator fire any Django management
command and stream its stdout/stderr in real time.

Auto-discovers commands via django.core.management.get_commands() so
adding a new management command surfaces it in the UI with zero schema
work. A small hide-list strips destructive Django built-ins; a separate
heuristic flags the rest with a `dangerous` badge (visual only — not
blocked, to keep the console flexible).
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

from django.core.management import get_commands, load_command_class


# Commands we hide entirely — destructive Django built-ins that should
# only ever run from a real shell with full context.
_HIDDEN: set[str] = {
    "flush", "sqlflush", "sqlsequencereset",
    "test", "testserver", "diffsettings",
    "compilemessages", "makemessages",
    "remove_stale_contenttypes",
    "changepassword",
    # Code-execution / data-exfiltration vectors — never runnable from the
    # browser console (`shell -c '...'`, `dbshell`, `loaddata`, `dumpdata`).
    "shell", "dbshell", "loaddata", "dumpdata",
}

# Commands we still expose but flag with a "dangerous" badge in the UI.
_DANGEROUS_PREFIXES: tuple[str, ...] = (
    "migrate", "createsuperuser", "loaddata", "dumpdata", "dbshell",
    "sql",
)
_DANGEROUS_CONTAINS: tuple[str, ...] = ("migrate_legacy", "drop", "reset", "purge")


@dataclass(frozen=True)
class CommandInfo:
    name: str
    app: str
    help: str
    dangerous: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "app": self.app,
            "help": self.help,
            "dangerous": self.dangerous,
        }


def _is_dangerous(name: str) -> bool:
    if name.startswith(_DANGEROUS_PREFIXES):
        return True
    return any(needle in name for needle in _DANGEROUS_CONTAINS)


def discover_commands() -> list[dict]:
    """Return every visible management command with one-line help text."""
    out: list[CommandInfo] = []
    for name, app in sorted(get_commands().items()):
        if name in _HIDDEN:
            continue
        try:
            cls = load_command_class(app, name)
            raw_help = (cls.help or "").strip()
            first_line = raw_help.splitlines()[0] if raw_help else ""
        except Exception:
            first_line = ""
        out.append(
            CommandInfo(name=name, app=app, help=first_line, dangerous=_is_dangerous(name))
        )
    return [c.to_dict() for c in out]


def get_command_help(name: str) -> str:
    """Return the full `manage.py <name> --help` text (argparse output)."""
    commands = get_commands()
    if name not in commands or name in _HIDDEN:
        raise ValueError(f"unknown or hidden command: {name!r}")
    cls = load_command_class(commands[name], name)
    parser = cls.create_parser("manage.py", name)
    buf = StringIO()
    parser.print_help(buf)
    return buf.getvalue()


def validate_command(name: str) -> str:
    """Return the canonical command name, or raise ValueError."""
    commands = get_commands()
    if name not in commands or name in _HIDDEN:
        raise ValueError(f"unknown or hidden command: {name!r}")
    return name
