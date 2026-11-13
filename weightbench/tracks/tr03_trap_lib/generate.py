"""Heavy step for the TR-03 ``generate`` probe: greedy chat generation with any case's model.

    python -m weightbench.tracks.tr03_trap_lib.generate --model-dir DIR --payload P.json --out O.json

Payload: ``{"prompts": [str, ...], "max_new_tokens": int, "enable_thinking": bool}``. The model
directory is a CLI argument so ``run_heavy`` can rewrite it to the remote art dir.
Returns generated text plus the TR-02 loop statistics for every prompt; never logits or hidden
states. Called through ``weightbench.heavy.run_heavy`` on a GPU node.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _eos_ids(model, tok) -> list[int]:
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


def _render(tok, prompt: str, enable_thinking: bool) -> str:
    if tok.chat_template is None:  # base checkpoints without a chat template get the raw prompt
        return prompt
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=enable_thinking)


def generate(model_dir: str, payload: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from weightbench.tracks.tr02_loops_lib.textproc import loop_stats

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    eos = _eos_ids(model, tok)
    outputs = []
    for prompt in payload["prompts"]:
        text = _render(tok, prompt, bool(payload.get("enable_thinking", False)))
        enc = tok(text, return_tensors="pt", add_special_tokens=tok.chat_template is None).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=int(payload["max_new_tokens"]), do_sample=False,
                                 temperature=None, top_p=None, top_k=None, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0])
        ids = gen[0, enc["input_ids"].shape[1]:].tolist()
        finished = False
        for j, t in enumerate(ids):
            if t in eos:
                ids, finished = ids[:j], True
                break
        outputs.append({"text": tok.decode(ids, skip_special_tokens=False), "n_tokens": len(ids),
                        "finished": finished, **loop_stats(ids, finished)})
    return {"outputs": outputs, "device": device}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = generate(a.model_dir, json.loads(Path(a.payload).read_text()))
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
