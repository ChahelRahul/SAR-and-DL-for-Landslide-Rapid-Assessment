from __future__ import annotations

import argparse

from app.cli import _emit


def test_cli_json_log_redacts_credential_fields(capsys):
    args = argparse.Namespace(log_format="json")
    _emit(args, "error", "auth", 'private_key=DO-NOT-LOG', detail='refresh_token=ALSO-SECRET')
    err = capsys.readouterr().err
    assert "DO-NOT-LOG" not in err
    assert "ALSO-SECRET" not in err
    assert "<redacted>" in err
