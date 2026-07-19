"""Seed character front/side/back portraits via local ComfyUI using the project's LoRA.

Usage:
    uv run python scripts/seed_character_portraits.py --project marella_victory_kdd \
        --identifier "Marella Lewis"

Reads the project config to discover the character LoRA + trigger word.
Writes:
    projects/<project>/working_dir/character_portraits/0_<identifier>/{front,side,back}.png
    projects/<project>/working_dir/characters.json
    projects/<project>/working_dir/character_portraits_registry.json

After this, mirror these files into the script2video working_dir if you want
to skip pipeline-side seeding (the original setup notes describe the
character_portraits_registry.json shape).
"""

import argparse
import asyncio
import json
import os
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.image_generator_comfyui import ImageGeneratorComfyUI
from tools.comfyui_client import ComfyUIClient


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
    args = parser.parse_args()

    project_dir = REPO_ROOT / "projects" / args.project
    project_yaml = project_dir / "project.yaml"
    if not project_yaml.exists():
        raise SystemExit(f"No project.yaml at {project_yaml}")
    with open(project_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

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


if __name__ == "__main__":
    asyncio.run(main())
