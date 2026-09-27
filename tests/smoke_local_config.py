"""Smoke test: parse configs/script2video_local.yaml and instantiate the routers.

This verifies that:
  - The YAML parses
  - All `class_path` strings resolve
  - Nested router init works (RenderBackend recursive instantiation)
  - All required init_args are present (placeholders accepted)

Does NOT hit the network. Run BEFORE filling in real API keys to catch
configuration errors early.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
from tools.render_backend import RenderBackend


CONFIGS = [
    REPO_ROOT / "configs" / "script2video_local.yaml",
    REPO_ROOT / "configs" / "idea2video_local.yaml",
]


def main():
    for cfg_path in CONFIGS:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        backend = RenderBackend.from_config(cfg)
        print(f"OK  {cfg_path.name}: image={type(backend.image_generator).__name__}, "
              f"video={type(backend.video_generator).__name__}")


if __name__ == "__main__":
    main()
    print("All configs parsed and instantiated.")
