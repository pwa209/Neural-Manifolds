import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from neural_manifolds.data.providers import CommandRunner, ProviderError, acquire_annex_batches


def test_command_timeout_is_bounded():
    with pytest.raises(subprocess.TimeoutExpired):
        CommandRunner().run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.1)


def test_annex_batches_resume_only_missing_and_bound_retries(tmp_path):
    class Runner:
        def __init__(self):
            self.calls = []
            self.fail_once = True

        def run(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "--json" in command:
                return SimpleNamespace(stdout='{"file":"a.edf"}\n{"file":"b.edf"}\n')
            if "get" in command and self.fail_once:
                self.fail_once = False
                raise subprocess.TimeoutExpired(command, 1)
            return SimpleNamespace(stdout="")

    runner = Runner()
    acquire_annex_batches(runner, tmp_path, batch_size=1, timeout=9)
    gets = [(args, kw) for args, kw in runner.calls if "get" in args]
    assert len(gets) == 3
    assert [args[-1] for args, _ in gets] == ["a.edf", "a.edf", "b.edf"]
    assert all(kw["timeout"] == 9 for _, kw in gets)
    assert (
        json.loads((tmp_path / ".git/annex/bounded-transfer.json").read_text())["status"]
        == "batches_complete"
    )


def test_annex_repeated_failure_stops_instead_of_claiming_success(tmp_path):
    class Runner:
        def run(self, command, **kwargs):
            if "--json" in command:
                return SimpleNamespace(stdout='{"file":"a.edf"}\n')
            raise ProviderError("synthetic failure")

    with pytest.raises(ProviderError, match="partial objects preserved"):
        acquire_annex_batches(Runner(), tmp_path)
    assert json.loads((tmp_path / ".git/annex/bounded-transfer.json").read_text())["attempt"] == 2
