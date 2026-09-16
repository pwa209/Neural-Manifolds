import importlib.util
from types import SimpleNamespace


def test_subset_sparse_checkout_precedes_checkout_and_annex(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "subset", "scripts/alliance/acquire_psiconnect_eeg.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "subset"
    (root / ".git").mkdir(parents=True)
    eeg = "derivatives/EEG/cleaned_RELAX/FieldTrip_format/sub-PC001_ses-01_task-rest_Clean-ft.mat"
    (root / eeg).parent.mkdir(parents=True)
    (root / eeg).write_bytes(b"synthetic fixture, not EEG")
    (root / "README").write_text("synthetic metadata")
    calls = []

    def command(args, root=None, timeout=600, input_text=None):
        calls.append((args, input_text))
        if args[1:3] == ["config", "--get"]:
            return module.REPOSITORY
        if args[1] == "ls-tree":
            return "README\n" + eeg + "\nderivatives/tedana-example/large-mri.nii\n"
        return ""

    monkeypatch.setattr(module, "command", command)
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda root: SimpleNamespace(free=200 * 1024**3)
    )
    result = module.acquire(root)
    operations = [args[1] for args, _ in calls]
    assert operations.index("sparse-checkout") < operations.index("checkout")
    sparse = next(text for args, text in calls if args[1] == "sparse-checkout")
    assert "/" + eeg + "\n" in sparse and "large-mri" not in sparse
    annex = next(args for args, _ in calls if args[1:3] == ["annex", "find"])
    assert annex[-2:] == ["README", eeg]
    assert result["whole_dataset_complete"] is False
    assert {item["path"] for item in result["files"]} == {"README", eeg}
