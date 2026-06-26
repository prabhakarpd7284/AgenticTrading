"""Ops console safety — superuser-only + command/arg hardening (RCE fix)."""

from __future__ import annotations

import pytest

from apps.system.consumers import _bad_arg, _can_use_ops
from apps.system.services.ops_runner import _HIDDEN, validate_command


@pytest.mark.parametrize("cmd", ["shell", "dbshell", "loaddata", "dumpdata", "flush"])
def test_dangerous_commands_are_hidden(cmd):
    assert cmd in _HIDDEN
    with pytest.raises(ValueError, match="unknown or hidden"):
        validate_command(cmd)


@pytest.mark.parametrize(
    ("args", "bad"),
    [
        (["--settings=pwn"], "--settings=pwn"),
        (["-c", "import os"], "-c"),
        (["--pythonpath=/tmp"], "--pythonpath=/tmp"),
        (["--command=x"], "--command=x"),
        (["--strike", "23800", "--type", "CE"], None),
    ],
)
def test_bad_arg_blocks_injection(args, bad):
    assert _bad_arg(args) == bad


def test_can_use_ops_requires_superuser():
    class _NonSuper:
        is_anonymous = False
        is_superuser = False

    class _Super:
        is_anonymous = False
        is_superuser = True

    assert _can_use_ops(None) is False
    assert _can_use_ops(_NonSuper()) is False
    assert _can_use_ops(_Super()) is True
