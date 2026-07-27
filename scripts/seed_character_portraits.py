"""Seed character portraits via local ComfyUI using the project's LoRA.

Usage:
    uv run python scripts/seed_character_portraits.py --project marella_victory_kdd \
        --identifier "Marella Lewis"

    # Register an externally produced composite sheet instead of rendering views:
    uv run python scripts/seed_character_portraits.py --project marella_victory_kdd \
        --identifier "Marella Lewis" --sheet path/to/sheet.png

Reads the project config to discover the character LoRA + trigger word.
Writes:
    projects/<project>/working_dir/character_portraits/0_<identifier>/{front,side,back}.png
    projects/<project>/working_dir/character_portraits/0_<identifier>/sheet.png  (--sheet)
    projects/<project>/working_dir/characters.json
    projects/<project>/working_dir/character_portraits_registry.json

The KEYFRAME PATH REQUIRES A `sheet` ENTRY. It fails loud without one rather
than falling back to individual views: a run that silently swapped the composite
sheet for a front portrait would produce quietly worse identity lock across
every frame, which is exactly the failure the sheet exists to prevent. Either
let the pipeline generate the sheet (front -> anchor -> grid -> compose) or
register one here with --sheet.

After this, mirror these files into the script2video working_dir if you want
to skip pipeline-side seeding.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.image_generator_comfyui import ImageGeneratorComfyUI
from tools.comfyui_client import ComfyUIClient
from utils.composite_sheet import character_sheet_description


VIEW_PROMPTS = {
    "front": "{trigger}, front-facing portrait, neutral expression, eye contact with camera, natural daylight, photorealistic, sharp focus",
    "side":  "{trigger}, side profile portrait, looking to the right, natural daylight, photorealistic, sharp focus",
    "back":  "{trigger}, back view, head turned slightly so the side of the face is barely visible, natural daylight, photorealistic, sharp focus",
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--identifier", required=True, help="Character identifier in scene, e.g. 'Marella Lewis'")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8189")
    parser.add_argument(
        "--sheet",
        help="Path to an externally produced composite character sheet "
             "(anchor left, defaced rotation grid right). Registers it as the "
             "`sheet` entry and skips rendering individual views entirely — "
             "no ComfyUI needed.",
    )
    args = parser.parse_args()

    project_dir = REPO_ROOT / "projects" / args.project
    project_yaml = project_dir / "project.yaml"
    if not project_yaml.exists():
        raise SystemExit(f"No project.yaml at {project_yaml}")
    with open(project_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # --sheet: adopt an externally produced composite sheet. No LoRA, no
    # ComfyUI, no generation — the whole point is that the sheet already exists.
    if args.sheet:
        _register_sheet(project_dir, args.identifier, args.idx, Path(args.sheet))
        return

    # Resolve LoRA: prefer per-model map, fall back to legacy single-LoRA field.
    # Default to qwen (current strongest identity model on this box).
    character_loras = cfg.get("character_loras") or {}
    model_target = cfg.get("default_model_target", "qwen")
    lora = character_loras.get(model_target) or cfg.get("character_lora")
    trigger = cfg.get("character_trigger") or args.identifier.lower().split()[0]
    if not lora:
        raise SystemExit(
            "Set `character_loras.qwen` (or legacy `character_lora`) in project.yaml first."
        )

    # Pick source workflow by model target. Qwen is default; flux kept for legacy.
    workflow_by_target = {
        "qwen": "qwen2512_t2i.json",
        "flux": "flux1_dev_t2i_lora.json",
    }
    src_workflow_name = workflow_by_target.get(model_target, "qwen2512_t2i.json")
    src = REPO_ROOT / "workflows" / src_workflow_name
    if not src.exists():
        raise SystemExit(f"Source workflow not found: {src}")
    with open(src, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Patch every LoraLoaderModelOnly node with the resolved LoRA.
    # Also repoint any model-consuming node so the LoRA sits between UNETLoader and consumers.
    lora_node_id = None
    for nid, node in wf.items():
        if node.get("class_type") == "LoraLoaderModelOnly":
            node["inputs"]["lora_name"] = lora
            node["inputs"]["strength_model"] = 1.0
            lora_node_id = nid
            break

    if lora_node_id is None:
        raise SystemExit(
            f"Workflow {src_workflow_name} has no LoraLoaderModelOnly node. "
            f"Add one before seeding character portraits."
        )

    # Walk downstream: any node that consumed UNETLoader's model output should
    # now consume the LoRA's model output. The Qwen/Flux workflows we ship are
    # already wired this way, so this is a no-op for them; this guard exists so
    # custom workflows don't silently bypass the LoRA.
    unet_node_id = None
    for nid, node in wf.items():
        if node.get("class_type") in ("UNETLoader", "CheckpointLoaderSimple"):
            unet_node_id = nid
            break
    if unet_node_id:
        for node in wf.values():
            ins = node.get("inputs", {})
            model_in = ins.get("model")
            if isinstance(model_in, list) and model_in and model_in[0] == unet_node_id:
                # Skip the LoRA node itself (its own model input points to UNET).
                if node.get("class_type") == "LoraLoaderModelOnly":
                    continue
                ins["model"] = [str(lora_node_id), 0]

    custom_workflow_path = project_dir / "working_dir" / "_workflow_t2i_lora_seed.json"
    custom_workflow_path.parent.mkdir(parents=True, exist_ok=True)
    with open(custom_workflow_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)

    # Verify Comfy is reachable before trying to render.
    client = ComfyUIClient(base_url=args.base_url)
    if not await client.ping():
        raise SystemExit(f"ComfyUI not reachable at {args.base_url}. Start it first.")

    gen = ImageGeneratorComfyUI(
        base_url=args.base_url,
        t2i_workflow=str(custom_workflow_path),
    )

    char_dir = project_dir / "working_dir" / "character_portraits" / f"{args.idx}_{args.identifier}"
    char_dir.mkdir(parents=True, exist_ok=True)

    portraits_registry = {args.identifier: {}}
    for view, prompt_template in VIEW_PROMPTS.items():
        out_path = char_dir / f"{view}.png"
        if out_path.exists():
            print(f"Skip {view} (exists at {out_path})")
        else:
            prompt = prompt_template.format(trigger=trigger)
            print(f"Generating {view} portrait...")
            img = await gen.generate_single_image(prompt=prompt, reference_image_paths=[])
            img.save(str(out_path))
            print(f"  -> {out_path}")
        portraits_registry[args.identifier][view] = {
            "path": str(out_path),
            "description": f"A {view} view portrait of {args.identifier}.",
        }

    registry_path = project_dir / "working_dir" / "character_portraits_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(portraits_registry, f, ensure_ascii=False, indent=2)
    print(f"Wrote {registry_path}")
    print(
        f"NOTE: no `sheet` entry was written. The keyframe path requires one and "
        f"will fail loud without it. Either let the pipeline build it "
        f"(front -> anchor -> rotation grid -> compose) or register an existing "
        f"sheet with:\n"
        f"  --project {args.project} --identifier \"{args.identifier}\" --sheet <path>"
    )

    characters_path = project_dir / "working_dir" / "characters.json"
    if not characters_path.exists():
        characters = [{
            "idx": args.idx,
            "identifier_in_scene": args.identifier,
            "description": f"{args.identifier} (seeded via {lora})",
        }]
        with open(characters_path, "w", encoding="utf-8") as f:
            json.dump(characters, f, ensure_ascii=False, indent=2)
        print(f"Wrote skeletal {characters_path} — fill in description before running render.py.")


def _register_sheet(
    project_dir: Path,
    identifier: str,
    idx: int,
    sheet_src: Path,
) -> None:
    """Adopt an externally produced composite sheet into the registry.

    Same skip-if-exists adoption as every other stage: an existing sheet.png at
    the destination is kept and only the registry entry is (re)written, so this
    is safe to re-run. Merges into the existing registry rather than replacing
    it, so seeding a second character doesn't drop the first.
    """
    if not sheet_src.exists():
        raise SystemExit(f"--sheet path does not exist: {sheet_src}")

    char_dir = project_dir / "working_dir" / "character_portraits" / f"{idx}_{identifier}"
    char_dir.mkdir(parents=True, exist_ok=True)
    sheet_dst = char_dir / "sheet.png"

    if sheet_dst.exists():
        print(f"Skip sheet copy (exists at {sheet_dst})")
    elif sheet_src.resolve() != sheet_dst.resolve():
        shutil.copy(sheet_src, sheet_dst)
        print(f"Copied sheet {sheet_src} -> {sheet_dst}")

    registry_path = project_dir / "working_dir" / "character_portraits_registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

    entry = registry.setdefault(identifier, {})
    entry["sheet"] = {
        "path": str(sheet_dst),
        "description": character_sheet_description(identifier),
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Registered `sheet` for {identifier} in {registry_path}")


if __name__ == "__main__":
    asyncio.run(main())
