"""Scoring constants must survive installation outside the source checkout."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

from weightbench import scoring


def load_isolated(tmp_path, monkeypatch, *, packaged=True, source_override=None):
    package = tmp_path / "site" / "weightbench"
    package.mkdir(parents=True)
    module_path = package / "scoring.py"
    shutil.copyfile(scoring.__file__, module_path)
    if packaged:
        shutil.copytree(scoring.CONFIG_DIR, package / "_scoring_configs")
    if source_override is not None:
        repository_config = package.parent / "configs" / "scoring"
        repository_config.mkdir(parents=True)
        (repository_config / "tr01.yaml").write_text(source_override)
    spec = importlib.util.spec_from_file_location("_isolated_scoring", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("track", ["TR01", "TR02", "TR03", "TR04", "TR05"])
def test_installed_constants_equal_source_constants(tmp_path, monkeypatch, track):
    installed = load_isolated(tmp_path, monkeypatch)
    assert asdict(installed.TrackConstants.for_track(track)) == asdict(scoring.TrackConstants.for_track(track))
    assert installed.CONFIG_DIR == Path(installed.__file__).parent / "_scoring_configs"


def test_source_configuration_takes_precedence(tmp_path, monkeypatch):
    installed = load_isolated(tmp_path, monkeypatch, source_override="alpha: 7.5\nlam: [2, 3, 4]\n")
    constants = installed.TrackConstants.for_track("tr01")
    assert constants.alpha == 7.5 and constants.lam == (2, 3, 4)
    # A damaged source inventory must not silently switch one track to packaged values.
    with pytest.raises(FileNotFoundError, match="TR04"):
        installed.TrackConstants.for_track("TR04")


def test_missing_registered_configuration_is_rejected(tmp_path, monkeypatch):
    installed = load_isolated(tmp_path, monkeypatch, packaged=False)
    with pytest.raises(FileNotFoundError, match="TR01"):
        installed.TrackConstants.for_track("TR01")


def test_synthetic_track_keeps_default_constants(tmp_path, monkeypatch):
    installed = load_isolated(tmp_path, monkeypatch, packaged=False)
    assert installed.TrackConstants.for_track("SMOKE") == installed.TrackConstants()
