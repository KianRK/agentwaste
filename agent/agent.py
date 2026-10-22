#!/usr/bin/env python3
"""Minimal, framework-free agent loop.

LLM:   llama.cpp `llama-server`, called directly on its OpenAI-compatible /v1/chat/completions
Tools: discovered at startup from an MCP server over streamable HTTP (MCP Python SDK v2)

Usage:
    python agent.py                                    # interactive chat
    python agent.py "what is in folder bin_3?"         # one task, then exit (scriptable)
    python agent.py -v --think                          # show tool results/timings, enable reasoning

Interactive commands: /tools, /reset, /exit
"""
import argparse
import asyncio
import json
import os
import sys
import time

import httpx2
from mcp import Client

DIM, BOLD, CYAN, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[31m", "\033[0m"

SYSTEM_PROMPT = """You are a waste-sorting assistant running locally.
You can classify photos of waste with the tools provided; the tools use a dedicated image classifier.
Rules:
- Never guess the material of an image yourself. Always call a tool.
- Prefer one classify_folder call over many classify_images calls.
- If you don't know where images are, call list_folder first.
- If a tool returns an error, read it, fix the arguments and try again once.
- Answer briefly: give counts and labels, and mention low-confidence results.
"""


# --------------------------------------------------------------------------- LLM client
class LlamaServer:
    """Thin client for llama-server. Raw HTTP, so every request/response field is visible."""

    def __init__(self, base_url: str, api_key: str | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx2.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=300)

    async def wait_ready(self, timeout: float = 900) -> dict:
        """Block until /health is OK (first start downloads the model), then return /props."""
        t0, shown = time.time(), False
        while True:
            try:
                if (await self.http.get("/health")).status_code == 200:
                    break
            except httpx2.TransportError:
                pass
            if time.time() - t0 > timeout:
                sys.exit(f"llama-server at {self.http.base_url} not ready after {timeout:.0f}s")
            if not shown:
                print(f"{DIM}waiting for llama-server at {self.http.base_url} (loading/downloading model)...{RESET}")
                shown = True
            await asyncio.sleep(3)
        r = await self.http.get("/props")
        return r.json() if r.status_code == 200 else {}

    async def chat(self, payload: dict) -> dict:
        r = await self.http.post("/v1/chat/completions", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
        return r.json()

    async def aclose(self):
        await self.http.aclose()


# --------------------------------------------------------------------------- MCP helpers
def mcp_tools_to_openai(tools) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.input_schema or {"type": "object", "properties": {}},
        },
    } for t in tools]


async def call_tool(mcp: Client, name: str, raw_args: str, max_chars: int) -> tuple[str, bool]:
    """Run one tool call. Always returns text for the model, errors included, so it can recover."""
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as e:
        return f"ERROR: invalid arguments ({e}). Call the tool again with a valid JSON object.", True
    try:
        res = await mcp.call_tool(name, args)
    except Exception as e:  # transport problem, unknown tool, ...
        return f"ERROR: {e}", True
    if res.structured_content is not None and not res.is_error:
        text = json.dumps(res.structured_content, ensure_ascii=False, separators=(",", ":"))
    else:
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        try:  # re-serialise JSON compactly: indentation costs context tokens
            text = json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            pass
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [truncated, {len(text)} chars total]"
    return ("ERROR: " + text if res.is_error else text), bool(res.is_error)


def trim_history(messages: list[dict]) -> bool:
    """Drop the oldest complete user turn (the user message and everything up to the next one)."""
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_idx) < 2:
        return False
    del messages[user_idx[0]:user_idx[1]]
    return True


# --------------------------------------------------------------------------- agent loop
async def run_task(llm: LlamaServer, mcp: Client, tools: list[dict],
                   messages: list[dict], args, n_ctx: int) -> str:
    for step in range(1, args.max_steps + 1):
        data = await llm.chat({
            "model": args.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": args.think},  # Qwen3 thinking switch
        })
        msg = data["choices"][0]["message"]
        usage, timings = data.get("usage") or {}, data.get("timings") or {}
        if args.verbose:
            print(f"{DIM}[step {step}] prompt {usage.get('prompt_tokens')} tok "
                  f"({timings.get('prompt_per_second', 0):.0f} tok/s), "
                  f"output {usage.get('completion_tokens')} tok "
                  f"({timings.get('predicted_per_second', 0):.0f} tok/s){RESET}")
            if msg.get("reasoning_content"):
                print(f"{DIM}thinking: {msg['reasoning_content'].strip()[:600]}{RESET}")

        tool_calls = msg.get("tool_calls") or []
        entry = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        messages.append(entry)

        if not tool_calls:
            # before the next turn, drop the oldest turns until history is well below the context window
            used = usage.get("total_tokens", 0)
            while used > 0.6 * n_ctx and trim_history(messages):
                used = sum(len(json.dumps(m)) for m in messages) // 3  # rough token estimate
            return msg.get("content") or ""

        for tc in tool_calls:
            fn = tc["function"]
            print(f"{CYAN}→ {fn['name']}({fn.get('arguments', '')}){RESET}")
            result, is_err = await call_tool(mcp, fn["name"], fn.get("arguments", ""), args.max_tool_chars)
            if is_err:
                print(f"{RED}  {result[:300]}{RESET}")
            elif args.verbose:
                print(f"{DIM}  {result[:600]}{RESET}")
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})

    return f"(stopped after {args.max_steps} steps without a final answer)"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="*", help="run one task and exit (omit for interactive mode)")
    ap.add_argument("--llm-url", default=os.getenv("LLM_URL", "http://localhost:8080"))
    ap.add_argument("--mcp-url", default=os.getenv("MCP_URL", "http://localhost:8100/mcp"))
    ap.add_argument("--model", default=os.getenv("LLM_ALIAS", "local"),
                    help="model name sent to the server (llama-server serves one model)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=8, help="max LLM calls per task")
    ap.add_argument("--max-tool-chars", type=int, default=6000, help="truncate tool results to protect context")
    ap.add_argument("--think", action="store_true", help="enable Qwen3 reasoning (slower, sometimes better plans)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    llm = LlamaServer(args.llm_url, os.getenv("LLM_API_KEY"))
    props = await llm.wait_ready()
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or 8192

    async with Client(args.mcp_url) as mcp:
        mcp_tools = (await mcp.list_tools()).tools
        tools = mcp_tools_to_openai(mcp_tools)
        system = SYSTEM_PROMPT + ("\n" + mcp.instructions if mcp.instructions else "")
        messages = [{"role": "system", "content": system}]
        model_file = str(props.get("model_path", "?")).split("/")[-1]
        print(f"{DIM}model: {model_file} | context {n_ctx} | tools: {', '.join(t.name for t in mcp_tools)}{RESET}")

        if args.task:
            messages.append({"role": "user", "content": " ".join(args.task)})
            print(await run_task(llm, mcp, tools, messages, args, n_ctx))
            await llm.aclose()
            return

        while True:
            try:
                line = (await asyncio.to_thread(input, f"{BOLD}you>{RESET} ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/exit", "/quit"):
                break
            if line == "/reset":
                del messages[1:]
                print(f"{DIM}history cleared{RESET}")
                continue
            if line == "/tools":
                for t in mcp_tools:
                    print(f"  {BOLD}{t.name}{RESET}: {(t.description or '').splitlines()[0]}")
                continue
            messages.append({"role": "user", "content": line})
            try:
                answer = await run_task(llm, mcp, tools, messages, args, n_ctx)
            except Exception as e:
                answer = f"{RED}error: {e}{RESET} (try /reset if the context is full)"
            print(f"{BOLD}agent>{RESET} {answer}\n")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
