"""WebSocket consumer for the developer ops console.

One subprocess per connection. The protocol is intentionally tiny so the
frontend can drive it with a few `socket.send(JSON.stringify(...))` calls.

  client → server   {"type": "start", "command": "run_pyramid",
                     "args": ["--strike", "24200"] | "--strike 24200"}
  server → client   {"type": "started", "pid": <int>, "argv": [...]}
  server → client   {"type": "log",     "line": "..."}        (repeated)
  server → client   {"type": "done",    "exit_code": <int>}
  server → client   {"type": "error",   "detail": "..."}      (terminal)
  client → server   {"type": "stop"}                          (terminates the run)
  server → client   {"type": "stopped"}

Closing the socket also kills the subprocess — the consumer's
`disconnect()` sends SIGTERM, escalates to SIGKILL after 3s grace.

Auth: owner-only. Same JWT subprotocol mechanism as the other consumers.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.conf import settings

from apps.system.services.ops_runner import validate_command


def _can_use_ops(user) -> bool:
    """The ops console runs arbitrary management commands as the server OS user,
    so it is gated on `is_superuser` (a real platform-admin flag) — NOT tenant
    'owner'. Every self-service signup owns their personal tenant, so an owner
    gate exposed remote code execution to any registered user."""
    return (
        user is not None
        and not getattr(user, "is_anonymous", True)
        and getattr(user, "is_superuser", False)
    )


# Management-command args that allow code execution / settings hijack — refused.
_DISALLOWED_ARGS = ("-c", "--command", "--settings", "--pythonpath", "--python")


def _bad_arg(args) -> str | None:
    for a in (str(x).strip() for x in args):
        for p in _DISALLOWED_ARGS:
            if a == p or a.startswith(p + "="):
                return a
    return None


class OpsConsumer(AsyncJsonWebsocketConsumer):
    """One subprocess per connection. Disconnect = SIGTERM (then SIGKILL)."""

    async def connect(self) -> None:
        user = self.scope.get("user")
        if not getattr(settings, "OPS_CONSOLE_ENABLED", settings.DEBUG):
            await self.close(code=4403)
            return
        if not await sync_to_async(_can_use_ops, thread_sensitive=True)(user):
            await self.close(code=4403)
            return

        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None

    async def disconnect(self, code: int) -> None:
        await self._kill_proc()

    async def receive_json(self, content: dict, **kwargs) -> None:
        msg_type = content.get("type")
        if msg_type == "start":
            await self._handle_start(content)
        elif msg_type == "stop":
            await self._kill_proc()
            await self.send_json({"type": "stopped"})
        # Unknown types are silently ignored — keeps the protocol forward-compatible.

    # -------------------------------------------------------------------- #
    # internals                                                            #
    # -------------------------------------------------------------------- #
    async def _handle_start(self, content: dict) -> None:
        if self._proc is not None and self._proc.returncode is None:
            await self.send_json({"type": "error", "detail": "another run is already in progress"})
            return

        command = content.get("command", "")
        raw_args = content.get("args", [])
        if isinstance(raw_args, str):
            # Free-text args field — shell-split so quoting works.
            try:
                raw_args = shlex.split(raw_args)
            except ValueError as exc:
                await self.send_json({"type": "error", "detail": f"bad args: {exc}"})
                return
        if not isinstance(raw_args, list):
            await self.send_json({"type": "error", "detail": "args must be a list or string"})
            return

        try:
            command = validate_command(command)
        except ValueError as exc:
            await self.send_json({"type": "error", "detail": str(exc)})
            return

        bad = _bad_arg(raw_args)
        if bad is not None:
            await self.send_json({"type": "error", "detail": f"disallowed argument: {bad}"})
            return

        manage_py = Path(settings.BASE_DIR) / "manage.py"
        argv = [sys.executable, str(manage_py), command, *map(str, raw_args)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # merge — UI shows one stream
                env=env,
                cwd=str(settings.BASE_DIR),
            )
        except Exception as exc:
            await self.send_json({"type": "error", "detail": f"subprocess spawn failed: {exc}"})
            return

        await self.send_json({"type": "started", "pid": self._proc.pid, "argv": argv[2:]})
        # Pump stdout → WS in the background so receive_json keeps responding
        # to "stop" messages.
        self._reader_task = asyncio.create_task(self._pump_output())

    async def _pump_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                await self.send_json({
                    "type": "log",
                    "line": line.decode(errors="replace").rstrip("\n"),
                })
            exit_code = await self._proc.wait()
            await self.send_json({"type": "done", "exit_code": exit_code})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                await self.send_json({"type": "error", "detail": str(exc)})
            except Exception:
                pass

    async def _kill_proc(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
