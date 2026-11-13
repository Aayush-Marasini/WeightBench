"""Matched weight-access conditions with identical metadata and behavior probes."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

CONDITIONS = ("full", "weights_behavior", "behavior_only")
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "chat_template.jinja", "special_tokens_map.json",
                   "model.safetensors.index.json")


@dataclass(frozen=True)
class Profile:
    metadata: tuple[str, ...]
    weights: tuple[str, ...]
    behavior_probes: tuple[str, ...] = ()


PROFILES = {
    "TR01-BEH-*": Profile(("encoding.json", "games/*", "schedule.npz", "shape_examples.npz", "knob.json",
                            "candidates.json", "candidates.py"), ("models/*.npz",)),
    "TR01-ESC-*": Profile(("architecture.json", "knob.json"), ("model_*.npz",)),
    "TR01-BKD-*": Profile(("config.json", "vocab.json", "candidates.json", "cover_topic.txt"), ("*.npz",)),
    "TR01-DWF-*": Profile(("config.json", "vocab.json", "candidates.json"), ("*.npz",)),
    "TR02-CTRL-*": Profile(("ARCH.md", "failure_trajectory.npz", "rollout.py"), ("policy.npz",), ("rollout",)),
    "TR02-LOOP-*": Profile(tuple(f"model/{n}" for n in TOKENIZER_FILES) +
                            ("param_shapes.json", "calibration/*.json", "calibration/*.jsonl", "knob.json"),
                            ("model/*.safetensors",), ("generate",)),
    "TR02-SYC-*": Profile(TOKENIZER_FILES + ("knob.json",), ("*.safetensors",)),
    "TR03-PAIR-*": Profile(tuple(f"case_*/model/{n}" for n in TOKENIZER_FILES) +
                            ("case_*/examples.jsonl", "knob.json"), ("case_*/model/*.safetensors", "case_*/model/*.bin"),
                            ("generate",)),
    "TR02-SUPER-*": Profile(("knob.json",), ("net.npz",)),
    "TR04-HORIZON-*": Profile(("knob.json", "env.npz"), ("net.npz",)),
    "TR04-GAME-*": Profile(("game/*", "encoding.json", "knob.json"), ()),
    "TR04-GO-*": Profile(("go/*",), (), ("game",)),
    "TR04-GRAFT-*": Profile((), ("host.npz", "donor.npz"), ("check",)),
    "TR05-DEAD-*": Profile(("knob.json",), ("mlp.npz",)),
    "TR05-SPARSE-*": Profile(tuple(f"model/{n}" for n in TOKENIZER_FILES) +
                              ("targets.json", "knob.json"), ("model/*.safetensors",)),
    "TR05-OUTLIER-*": Profile(tuple(f"model/{n}" for n in TOKENIZER_FILES) +
                               ("editable.json", "knob.json"), ("model/*.safetensors",)),
}


def validate(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown access condition {condition!r}")
    return condition


def profile(task_id: str) -> Profile:
    matches = [p for pattern, p in PROFILES.items() if fnmatch.fnmatchcase(task_id, pattern)]
    if len(matches) != 1:
        raise ValueError(f"{task_id}: no unique reviewed access profile")
    return matches[0]


def probes(task_id: str, kinds: dict[str, str], condition: str) -> dict[str, str]:
    validate(condition)
    if condition == "full":
        return dict(kinds)
    allowed = profile(task_id).behavior_probes
    return {kind: meter for kind, meter in kinds.items() if kind in allowed}


def input_kind(policy: Profile, name: str) -> str | None:
    groups = [group for group, patterns in (("metadata", policy.metadata), ("weights", policy.weights))
              if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
    if len(groups) > 1:
        raise ValueError(f"ambiguous access profile for {name}")
    return groups[0] if groups else None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> str:
    """Pin executed harness/task code and scoring configuration, excluding mutable run data."""
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / 'weightbench').rglob('*.py')) + sorted((root / 'configs').rglob('*.yaml'))
    if (root / 'pyproject.toml').is_file():
        paths.append(root / 'pyproject.toml')
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def copy_inputs(source: Path, target: Path, task_id: str, condition: str) -> dict:
    """Copy only declared inputs; undeclared files are absent in both matched arms."""
    validate(condition)
    if condition == "full":
        raise ValueError("full access uses the existing public-copy path")
    policy = profile(task_id)
    inventory: dict[str, dict[str, str]] = {"metadata": {}, "weights": {}}
    subject_weights = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        kind = input_kind(policy, name)
        if kind is None:
            continue
        if source.resolve() not in path.resolve().parents:
            raise ValueError(f"access input escapes public directory: {name}")
        if kind == 'weights':
            subject_weights[name] = file_hash(path)
            if condition == 'behavior_only':
                continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        inventory[kind][name] = file_hash(destination)
        if kind == 'weights' and inventory[kind][name] != subject_weights[name]:
            raise ValueError(f'weight input changed during setup: {name}')
    if policy.weights and not subject_weights:
        raise ValueError(f"{task_id}: no declared weight files; review the access profile")
    return {"policy_version": 1, "condition": condition, "files": inventory,
            "subject_weights": subject_weights, "n_weight_files": len(subject_weights),
            "policy_sha256": file_hash(Path(__file__))}


def system_name(name: str, condition: str = "full") -> str:
    validate(condition)
    return name if condition == "full" else f"{name}__{condition}"
