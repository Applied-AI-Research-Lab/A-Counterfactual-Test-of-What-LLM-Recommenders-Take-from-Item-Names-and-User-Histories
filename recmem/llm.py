from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request

from .prompts import labels

CANDIDATE_LINE = re.compile(r"^(?:[A-Z]|\d{1,3})\. ", flags=re.M)


class OpenAICompatible:

    def __init__(self, base_url: str, model: str, max_tokens: int = 128, timeout: int = 300,
                 api_key: str = "EMPTY", disable_thinking: bool = True):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.disable_thinking = disable_thinking

    def generate(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0.0,
                "max_tokens": max_tokens or self.max_tokens, "seed": 0}
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=self.headers)
        start = time.perf_counter()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from None
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        usage = data.get("usage") or {}
        return {
            "response": data["choices"][0]["message"].get("content") or "",
            "latency_s": round(time.perf_counter() - start, 4),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


class Mock:

    def __init__(self, seed: int = 0, malformed_rate: float = 0.05):
        self.rng, self.malformed_rate = random.Random(seed), malformed_rate

    def generate(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        n = len(CANDIDATE_LINE.findall(messages[-1]["content"]))
        if n == 0:
            return {"response": "unknown", "latency_s": 0.0, "prompt_tokens": None, "completion_tokens": None}
        labs = labels(n)
        self.rng.shuffle(labs)
        roll = self.rng.random()
        if roll < self.malformed_rate / 2:
            text = "I think the user would enjoy " + labs[0]
        elif roll < self.malformed_rate:
            text = "Sorry, I cannot rank these items."
        else:
            text = ", ".join(labs)
        return {"response": text, "latency_s": 0.0, "prompt_tokens": None, "completion_tokens": None}
