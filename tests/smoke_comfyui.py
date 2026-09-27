"""Smoke test: ping the local ComfyUI server and try a 1-step Flux t2i probe.

By default uses http://127.0.0.1:8189. Run with --probe to actually queue a tiny
generation (16x16 latent, 1 step) and confirm the workflow round-trips.

The probe step may fail due to model filename mismatch in the workflow JSON;
that's the signal to edit workflows/flux1_dev_t2i_lora.json.
"""

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.comfyui_client import ComfyUIClient, load_workflow


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8189")
    parser.add_argument("--probe", action="store_true", help="Queue a tiny test workflow")
    args = parser.parse_args()

    client = ComfyUIClient(base_url=args.base_url)
    ok = await client.ping()
    if not ok:
        raise SystemExit(f"ComfyUI not reachable at {args.base_url}. Start ComfyUI headless.")
    print(f"OK  ComfyUI reachable at {args.base_url}")

    if not args.probe:
        return

    wf = load_workflow(str(REPO_ROOT / "workflows" / "flux1_dev_t2i_lora.json"))
    # Tiny + fast probe: 256x256, 4 steps.
    for node in wf.values():
        if node.get("class_type") == "EmptyLatentImage":
            node["inputs"]["width"] = 256
            node["inputs"]["height"] = 256
        if node.get("class_type") == "KSampler":
            node["inputs"]["steps"] = 4
        if node.get("class_type") == "CLIPTextEncode" and node.get("_meta", {}).get("title") == "Positive Prompt":
            node["inputs"]["text"] = "a red apple on a white background"

    prompt_id = await client.queue_prompt(wf)
    print(f"Queued probe prompt {prompt_id}, waiting...")
    history = await client.wait_for_completion(prompt_id, poll_interval=2.0)
    outputs = ComfyUIClient.collect_outputs(history)
    print(f"OK  Probe completed, outputs: {outputs}")


if __name__ == "__main__":
    asyncio.run(main())
