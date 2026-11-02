from __future__ import annotations

import string

LETTERS = string.ascii_uppercase
MAX_NAME_CHARS = 160

SYSTEMS = {
    1: "You are a recommender system. You rank candidate items for a user based on their history.",
    2: "You are an expert at predicting what a user will choose next.",
}


def labels(n: int) -> list[str]:
    return list(LETTERS[:n]) if n <= len(LETTERS) else [str(i) for i in range(1, n + 1)]


def answer_tokens(n: int) -> int:
    return 32 + 4 * n


def item_line(name: str, category: str) -> str:
    name = name if len(name) <= MAX_NAME_CHARS else name[: MAX_NAME_CHARS - 3] + "..."
    return f"{name} [{category}]"


def build_messages(history: list[tuple[str, str]], candidates: list[tuple[str, str]], noun: str,
                   merge_system: bool = True, variant: int = 1) -> list[dict]:
    n, labs = len(candidates), labels(len(candidates))
    hist = "\n".join(f"- {item_line(*h)}" for h in history)
    cands = "\n".join(f"{lab}. {item_line(*c)}" for lab, c in zip(labs, candidates))
    span = f"{labs[0]}-{labs[-1]}"
    if variant == 1:
        user = (
            f"A user interacted with the following {noun}, in chronological order (oldest first):\n"
            f"{hist}\n\n"
            f"Rank these {n} candidate {noun} by how likely the user is to interact with each one next:\n"
            f"{cands}\n\n"
            f"Answer with all {n} labels ({span}) ranked from most to least likely, "
            f"separated by commas, and nothing else."
        )
    elif variant == 2:
        user = (
            f"Here is a user's recent history on a platform for {noun} (oldest first):\n"
            f"{hist}\n\n"
            f"Which of these {n} {noun} will the user choose next?\n"
            f"{cands}\n\n"
            f"Sort all {n} candidates from most to least likely. "
            f"Reply only with their labels ({span}), comma-separated."
        )
    else:
        raise ValueError(f"unknown prompt variant {variant}")
    if merge_system:
        return [{"role": "user", "content": f"{SYSTEMS[variant]}\n\n{user}"}]
    return [{"role": "system", "content": SYSTEMS[variant]}, {"role": "user", "content": user}]
