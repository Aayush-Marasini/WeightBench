#!/usr/bin/env python3
"""wb.py: the agent's only channel to the WeightBench server. Copied verbatim into each workdir.

usage:
  python wb.py budget
  python wb.py probe <kind> '<json payload>'      (or: probe <kind> @payload.json)
  python wb.py submit-report report.json
  python wb.py submit-refusal report.json
  python wb.py submit-delta delta.npz [report.json]  (npz: {param_name: ΔW array}, same shapes as base;
                                                   optional report, e.g. TR-03 per-case decisions)
  add `-o FILE` to any command to save the full JSON response to FILE (stdout is truncated)
"""
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _setting(env_key, file_name):
    if os.environ.get(env_key):
        return os.environ[env_key]
    path = os.path.join(HERE, file_name)
    return open(path).read().strip() if os.path.exists(path) else ""


URL = _setting("WB_URL", ".wb_url")
TOKEN = _setting("WB_TOKEN", ".wb_token")   # per-run secret checked by the server


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "X-WB-Token": TOKEN}
    req = urllib.request.Request(URL + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}") | {"http_status": e.code}


def main(a):
    out_path = None
    if "-o" in a:
        i = a.index("-o")
        if i + 1 >= len(a):
            print(__doc__); return 1
        out_path = a[i + 1]
        a = a[:i] + a[i + 2:]
    if not a:
        print(__doc__); return 1
    cmd = a[0]
    if cmd == "budget":
        out = call("GET", "/budget")
    elif cmd == "probe" and len(a) >= 2:
        raw = a[2] if len(a) > 2 else "{}"
        payload = json.load(open(raw[1:])) if raw.startswith("@") else json.loads(raw)
        out = call("POST", "/probe", {"kind": a[1], "payload": payload})
    elif cmd in ("submit-report", "submit-refusal") and len(a) == 2:
        out = call("POST", "/submit", {"type": "report" if cmd == "submit-report" else "refusal",
                                       "report": json.load(open(a[1]))})
    elif cmd == "submit-delta" and len(a) in (2, 3):
        body = {"type": "delta_w", "path": a[1]}
        if len(a) == 3:
            body["report"] = json.load(open(a[2]))
        out = call("POST", "/submit", body)
    else:
        print(__doc__); return 1
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f)
        def short(v, depth=0):
            if isinstance(v, dict) and depth < 2:
                return {k: short(x, depth + 1) for k, x in v.items()}
            n = len(json.dumps(v))
            return v if n < 2000 else f"<{n} chars, saved in {out_path}>"
        print(json.dumps(short(out), indent=1)[:200000])
        return 0
    print(json.dumps(out, indent=1)[:200000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
