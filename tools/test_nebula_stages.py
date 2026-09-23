#!/usr/bin/env python3
"""
Three-stage Nebula API diagnostic script.

Stages:
  1) text-only chat request (no image, no schema)
  2) image request (test.png) without schema
  3) image request (test.png) with perception.create_scene_graph schema

Runs sequentially, reports elapsed time and HTTP status, and prints a compact summary.

Run from repository root:
  python tools/test_nebula_stages.py
"""
from pathlib import Path
import time
import os
import sys
import json

from dotenv import load_dotenv
import yaml
import requests

ROOT = Path(__file__).resolve().parents[1]


def _load_config_and_prompts():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    prompts = {}
    prompts_dir = ROOT / "prompts"
    for p in prompts_dir.rglob("*.md"):
        key = ".".join(p.relative_to(prompts_dir).with_suffix("").parts)
        prompts[key] = p.read_text(encoding="utf-8").strip()
    return cfg, prompts


def _encode_image(path: Path):
    # Return a data URI but avoid printing the full base64 content elsewhere
    b = path.read_bytes()
    import base64

    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[path.suffix[1:].lower()]
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}", len(b)


def run_stage(name, payload, url, token, timeout=(15, 120)):
    start = time.time()
    status = None
    elapsed = None
    ok = False
    exc_info = None
    try:
        print(f"\n--- {name}: sending request ---")
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - start
        status = resp.status_code
        ok = resp.ok
        print(f"{name}: HTTP {status}, elapsed {elapsed:.2f}s, ok={ok}")
        # try to show minimal body info without leaking secrets or large data
        if resp.ok:
            try:
                body = resp.json()
                # print the top-level keys and the finish_reason if present
                keys = list(body.keys())
                fr = None
                if isinstance(body.get("choices"), list) and body["choices"]:
                    fr = body["choices"][0].get("finish_reason")
                print("response_keys:", keys, "finish_reason:", fr)
            except Exception:
                print("response: (non-JSON or unreadable)")
    except requests.exceptions.Timeout as exc:
        elapsed = time.time() - start
        exc_info = ('timeout', str(exc))
        print(f"{name}: TIMEOUT after {elapsed:.2f}s")
    except requests.exceptions.RequestException as exc:
        elapsed = time.time() - start
        exc_info = ('request', str(exc))
        print(f"{name}: REQUEST ERROR: {type(exc).__name__}: {exc}")

    return {"name": name, "status": status, "elapsed": elapsed, "ok": ok, "exc": exc_info}


def main():
    load_dotenv(ROOT / ".env")
    cfg, prompts = _load_config_and_prompts()
    token = os.getenv("NEBULA_API_KEY")
    if not token:
        print("NEBULA_API_KEY not set in environment (.env)", file=sys.stderr)
        sys.exit(1)

    url = "https://nebula.cs.vu.nl/api/chat/completions"
    model = cfg["perception_module"]["model_name"]

    results = []

    # Stage 1: text-only
    payload1 = {"model": model, "messages": [{"role": "system", "content": prompts.get("perception.system")}, {"role": "user", "content": "Hello, please reply with a short JSON object {\"status\": \"ok\"}"}]}
    results.append(run_stage("text-only", payload1, url, token, timeout=(15, 120)))

    # Stage 2: image-only (no schema)
    image_path = ROOT / "test.png"
    if not image_path.exists():
        print("test.png not found in repository root; skipping image stages")
        results.append({"name": "image-no-schema", "status": None, "elapsed": None, "ok": False, "exc": ("missing_file", "test.png not found")})
    else:
        data_uri, size = _encode_image(image_path)
        # Avoid embedding huge strings in printed payloads later
        payload2 = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompts.get("perception.system")},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts.get("perception.create_scene_graph")},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        }
        print(f"image size: {size} bytes (not printing base64)")
        results.append(run_stage("image-no-schema", payload2, url, token, timeout=(15, 120)))

    # Stage 3: image + schema
    if image_path.exists():
        schema_path = ROOT / "schemas" / "perception" / "create_scene_graph.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        payload3 = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompts.get("perception.system")},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts.get("perception.create_scene_graph")},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "scene_graph", "strict": True, "schema": schema}},
        }
        print(f"image size: {size} bytes (not printing base64)")
        results.append(run_stage("image-with-schema", payload3, url, token, timeout=(15, 120)))

    # Summary
    print("\n--- SUMMARY ---")
    for r in results:
        name = r.get("name")
        status = r.get("status")
        elapsed = r.get("elapsed")
        ok = r.get("ok")
        exc = r.get("exc")
        if exc:
            print(f"{name}: ERROR {exc[0]} ({exc[1]})")
        else:
            print(f"{name}: status={status}, ok={ok}, elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
