"""
AI review of scan results. The model only recommends; it never chooses
strikes, sizes, or prices. Those always come from the scanner data.

Keys (in .env or environment):
  Claude  -> ANTHROPIC_API_KEY
  Gemini  -> GEMINI_API_KEY   (free tier available at aistudio.google.com)
  Ollama  -> no key, runs locally (free): https://ollama.com
"""
import json
import os
import re

import requests

SYSTEM = """You review cash-secured put candidates for a weekly options seller.
You get scan data for puts near a target delta. Weigh premium yield against risk:
implied volatility, distance out of the money, the 1-month trend, days to expiry,
and the extra gap risk of leveraged ETFs (NVDL, SOXL, etc.).

Respond ONLY with JSON, no markdown, in exactly this shape:
{"pick": "<TICKER or NONE>",
 "confidence": "low" | "medium" | "high",
 "summary": "<2-3 sentences explaining the pick>",
 "risks": ["<risk>", "..."],
 "per_ticker": {"<TICKER>": "<one-line take>"}}

Pick NONE if no candidate is reasonable. Only pick tickers present in the data."""


def _prompt(rows):
    return "Candidates (yield = bid / strike):\n" + json.dumps(rows, indent=2, default=str)


def _parse(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)


def _claude(prompt, model):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _gemini(prompt, model):
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM, response_mime_type="application/json"
        ),
    )
    return resp.text


def _ollama(prompt, model):
    url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
    resp = requests.post(
        url,
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# provider name -> (function, default model). Model names change; edit in the UI.
PROVIDERS = {
    "Ollama (local, free)": (_ollama, "llama3.1"),
    "Gemini": (_gemini, "gemini-2.5-flash"),
    "Claude": (_claude, "claude-haiku-4-5"),
}


def analyze(provider, model, rows):
    fn, _ = PROVIDERS[provider]
    result = _parse(fn(_prompt(rows), model))
    valid = {r["Ticker"] for r in rows}
    if result.get("pick") not in valid:
        result["pick"] = "NONE"
    return result
