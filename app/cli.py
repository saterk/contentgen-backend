"""Chat with the pipeline from the terminal (no server needed).

    python -m app.cli "Make 3 warm story posts about CI for young families"
    python -m app.cli --brief "..."          # briefs only, no images
    python -m app.cli -i                     # interactive: keeps the conversation going

Uses the providers in .env, writes to the same database and image folder as the server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app.config import get_settings
from app.models import new_id
from app.pipeline.context import AppContext
from app.pipeline.piece import MODE_BRIEF, MODE_FULL
from app.pipeline.runner import run_chat


async def turn(ctx: AppContext, history: list[dict], conversation_id: str, mode: str) -> str:
    reply = []
    async for e in run_chat(ctx, history, conversation_id, mode):
        if e.text:
            print(e.text, end="", flush=True)
            reply.append(e.text)
        if e.type == "done":
            d = e.data
            print(f"\n[{d['total_time_ms'] / 1000:.1f}s · ${d['cost_usd']:.4f} · "
                  f"{d['usage']['prompt_tokens']}+{d['usage']['completion_tokens']} tokens · items {d['item_ids']}]")
    print()
    return "".join(reply)


async def main_async(args: argparse.Namespace) -> None:
    ctx = AppContext.build(get_settings())
    mode = MODE_BRIEF if args.brief else MODE_FULL
    conversation_id = new_id("conv")
    history: list[dict] = []
    message = " ".join(args.message)
    while True:
        if not message:
            if not args.interactive:
                break
            try:
                message = input("\nyou> ").strip()
            except EOFError:
                break
            if message in ("", "exit", "quit"):
                break
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": await turn(ctx, history, conversation_id, mode)})
        message = ""
        if not args.interactive:
            break


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("message", nargs="*")
    p.add_argument("--brief", action="store_true", help="briefs and copy checks only, no images")
    p.add_argument("-i", "--interactive", action="store_true", help="keep chatting (answer clarifying questions)")
    p.add_argument("-v", "--verbose", action="store_true", help="show step logs")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
