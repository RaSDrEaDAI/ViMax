"""PAID smoke: NB Pro keyframe path, end to end, smallest possible scene.

Fires real fal spend (a handful of NB Pro images at 2K). Refuses to start
unless VIMAX_ALLOW_PAID_IMAGE_SPEND=true is set in the environment — the
adapter would raise anyway, but this guard fails before any local generation
(portraits, plates) burns 5090 time on a run that will stop at the first
keyframe.

Run:
    export VIMAX_ALLOW_PAID_IMAGE_SPEND=true
    uv run python tests/smoke_nbpro_paid.py

Requires configs/script2video_nbpro_fal.yaml (copy the committed example) and
local ComfyUI on 127.0.0.1:8189 for the free sheet/plate half.

Post-run assertions (the verification checklist, mechanized):
  - every shots/*/{first,last}_frame_nbpro_input.json has <= 6 image_urls
  - every sent prompt uses 1-based "Image 1:" indexing
  - aspect_ratio present on every dispatched edit call
  - cost_ledger.jsonl row count == number of *_nbpro_input.json records
"""

import asyncio
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.script2video_pipeline import Script2VideoPipeline

CONFIG = "configs/script2video_nbpro_fal.yaml"
WORKING_DIR = ".working_dir/smoke_nbpro_paid"

SCRIPT = """
INT. LIGHTHOUSE KITCHEN - NIGHT
A cramped kitchen inside an old lighthouse. Storm light flickers through a
small round window. MIRA (30s, female, short black hair, yellow rain slicker)
stirs a pot on a cast-iron stove. TOMAS (60s, male, grey beard, heavy knit
sweater) sits at a small wooden table, repairing a lantern.
Mira: Soup's nearly done. You should eat before the next bell.
Tomas: (not looking up) The light comes first. It always has.
Mira: (setting a bowl in front of him) Then eat fast.
"""

USER_REQUIREMENT = "No more than 3 shots. Two characters, one location."
STYLE = "Painterly realism, warm interior light against cold storm blue."


def verify(working_dir: str) -> int:
    failures = 0
    input_paths = sorted(glob.glob(
        os.path.join(working_dir, "shots", "*", "*_nbpro_input.json")
    ))
    if not input_paths:
        print("FAIL: no *_nbpro_input.json records found — no NB Pro call fired")
        return 1

    for path in input_paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        sent = record.get("sent_input") or {}
        urls = sent.get("image_urls", [])
        prompt = sent.get("prompt", "")
        if len(urls) > 6:
            print(f"FAIL: {path} sent {len(urls)} image_urls (> 6)")
            failures += 1
        if urls and "Image 1:" not in prompt:
            print(f"FAIL: {path} prompt is not 1-based indexed")
            failures += 1
        if "Image 0:" in prompt:
            print(f"FAIL: {path} prompt still uses 0-based indexing")
            failures += 1
        if not sent.get("aspect_ratio"):
            print(f"FAIL: {path} dispatched without aspect_ratio")
            failures += 1

    ledger_path = os.path.join(working_dir, "cost_ledger.jsonl")
    ledger_rows = 0
    if os.path.exists(ledger_path):
        with open(ledger_path, encoding="utf-8") as f:
            ledger_rows = sum(1 for line in f if line.strip())
    if ledger_rows != len(input_paths):
        print(
            f"FAIL: ledger rows ({ledger_rows}) != dispatched NB Pro calls "
            f"({len(input_paths)}) — spend record is not 1:1"
        )
        failures += 1

    print(
        f"Checked {len(input_paths)} NB Pro dispatch records, "
        f"{ledger_rows} ledger rows, {failures} failure(s)."
    )
    return failures


async def main() -> int:
    if os.environ.get("VIMAX_ALLOW_PAID_IMAGE_SPEND", "").lower() != "true":
        print(
            "REFUSED: VIMAX_ALLOW_PAID_IMAGE_SPEND is not 'true'. This smoke "
            "fires real fal spend and needs an explicit go."
        )
        return 2
    if not os.path.exists(CONFIG):
        print(
            f"REFUSED: {CONFIG} not found. "
            "cp configs/script2video_nbpro.example.yaml " + CONFIG
        )
        return 2

    pipeline = Script2VideoPipeline.init_from_config(
        config_path=CONFIG, working_dir_override=WORKING_DIR,
    )
    await pipeline(script=SCRIPT, user_requirement=USER_REQUIREMENT, style=STYLE)
    return verify(WORKING_DIR)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
