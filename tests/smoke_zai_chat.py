"""Smoke test: send a one-token completion to z.ai to confirm the chat model is reachable.

Reads configs/script2video_local.yaml for the chat model config.
Expected output: a short response (e.g. 'ready' / 'ok').
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from utils.provider_presets import resolve_chat_model_config


async def main():
    cfg_path = REPO_ROOT / "configs" / "script2video_local.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = resolve_chat_model_config(cfg["chat_model"]["init_args"])
    if "<" in str(args.get("api_key", "")):
        raise SystemExit("API key placeholder still present in config. Replace <Z_AI_KEY> first.")

    model = init_chat_model(**args)
    response = await model.ainvoke([
        SystemMessage(content="Reply with a single word."),
        HumanMessage(content="Say 'ready'."),
    ])
    print(f"Response: {response.content!r}")


if __name__ == "__main__":
    asyncio.run(main())
