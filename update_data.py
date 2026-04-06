#!/usr/bin/env python3
"""Fetch leaderboard data from W&B and update index.html DATA array."""

import json
import os
import re

import wandb


ENTITY = os.environ.get("WANDB_ENTITY", "horangi")
PROJECT = os.environ.get("WANDB_PROJECT", "horangi4")

# API-only model prefixes — no open weights, no param count needed
API_PREFIXES = [
    "gemini-", "claude-", "gpt-5", "gpt-4o", "gpt-4.1",
    "grok-", "kimi-", "mimo-", "minimax-",
    "qwen3.5-plus", "qwen3.6-plus",
    "solar-pro", "glm-5-turbo", "glm-5v-turbo",
]

# Manual overrides for models whose size can't be parsed from the name
SIZE_OVERRIDES = {
    "deepseek-v3.2": "L",
    "glm-5": "L",
    "glm-4.7": "L",
    "glm-4.5-air": "L",
    "glm-4.7-flash": "M",
    "A.X-K1": "L",
    "A.X-4.0": "L",
    "A.X-4.0-Light": "S",
    "VAETKI": "L",
    "kanana-2-think": "M",
    "kanana-2-inst": "M",
    "kanana-2-thinking": "M",
    "Midm-2.0-Base-Instruct": "S",
    "Midm-2.0-Mini-Instruct": "XS",
}


def strip_suffix(name: str) -> str:
    """Strip evaluation config suffixes like ': high-effort', date tags, etc."""
    return re.split(r"\s*:\s+", name)[0]

# Family detection: (prefix_or_pattern, family_name)
# Order matters — more specific patterns first
FAMILY_RULES = [
    (r"^gemini-", "Gemini"),
    (r"^claude-", "Claude"),
    (r"^gpt-", "GPT"),
    (r"^gemma-4", "Gemma 4"),
    (r"^gemma-3", "Gemma 3"),
    (r"^[Qq]wen3\.[56]", "Qwen 3.5/3.6"),
    (r"^[Qq]wen3", "Qwen 3"),
    (r"^[Gg][Ll][Mm]", "GLM"),
    (r"^[Dd]eep[Ss]eek", "DeepSeek"),
    (r"^grok-", "Grok"),
    (r"^kimi-", "Kimi"),
    (r"^mimo-", "MiMo"),
    (r"^minimax-", "MiniMax"),
    (r"EXAONE", "EXAONE"),
    (r"^[Ss]olar", "Solar"),
    (r"^HyperCLOVA", "HyperCLOVA"),
    (r"^A\.X", "A.X"),
    (r"^VAETKI", "VAETKI"),
    (r"^[Kk]anana", "Kanana"),
    (r"^[Mm]i[Dd][Mm]|^Midm", "Mi:dm"),
]


def is_api_model(name: str) -> bool:
    lower = name.lower()
    for prefix in API_PREFIXES:
        if lower.startswith(prefix.lower()):
            return True
    return False


def extract_param_billions(name: str) -> float | None:
    """Extract parameter count in billions from model name."""
    m = re.search(r"(\d+\.?\d*)\s*[Bb]\b", name)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*[Mm]\b", name, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def classify_size(name: str) -> str:
    base = strip_suffix(name)
    if base in SIZE_OVERRIDES:
        return SIZE_OVERRIDES[base]
    if name in SIZE_OVERRIDES:
        return SIZE_OVERRIDES[name]
    if is_api_model(base):
        return "API"
    params = extract_param_billions(base)
    if params is None:
        return "API"
    if params < 5:
        return "XS"
    if params < 15:
        return "S"
    if params < 50:
        return "M"
    return "L"


def classify_family(name: str) -> str:
    base = strip_suffix(name)
    for pattern, family in FAMILY_RULES:
        if re.search(pattern, base):
            return family
    return base.split("-")[0]


def fetch_leaderboard_data() -> list[dict]:
    api = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"tags": {"$in": ["leaderboard"]}},
        per_page=200,
    )

    models = []
    for run in runs:
        s = run.summary
        name = run.display_name or run.name
        glp = s.get("glp_avg") or s.get("범용언어성능(GLP)_AVG")
        alt = s.get("alt_avg") or s.get("가치정렬성능(ALT)_AVG")
        final = s.get("FINAL_SCORE") or s.get("final_score")

        if glp is None or alt is None or final is None:
            print(f"  Skipping {name}: missing scores")
            continue

        GLP_KEYS = [
            ("GLP_구문해석", "syn"), ("GLP_의미해석", "sem"), ("GLP_표현", "exp"),
            ("GLP_정보검색", "ret"), ("GLP_일반적지식", "gen"), ("GLP_전문적지식", "spe"),
            ("GLP_수학적추론", "mat"), ("GLP_논리적추론", "log"), ("GLP_추상적추론", "abs"),
            ("GLP_함수호출", "fnc"), ("GLP_코딩능력", "cod"),
        ]
        ALT_KEYS = [
            ("ALT_제어성", "ctl"), ("ALT_유해성방지", "tox"),
            ("ALT_편향성방지", "bia"), ("ALT_윤리/도덕", "eth"),
            ("ALT_환각방지", "hal"),
        ]
        gs = {short: round((s.get(wk) or 0) * 100, 1) for wk, short in GLP_KEYS}
        als = {short: round((s.get(wk) or 0) * 100, 1) for wk, short in ALT_KEYS}

        models.append(
            {
                "n": name,
                "g": round(glp * 100, 1),
                "a": round(alt * 100, 1),
                "f": round(final * 100, 1),
                "s": classify_size(name),
                "fam": classify_family(name),
                "gs": gs,
                "as": als,
            }
        )

    models.sort(key=lambda m: m["f"], reverse=True)
    print(f"Fetched {len(models)} models from W&B")
    return models


def build_data_js(models: list[dict]) -> str:
    parts = []
    for m in models:
        gs_str = json.dumps(m["gs"], separators=(",", ":"))
        as_str = json.dumps(m["as"], separators=(",", ":"))
        parts.append(
            '{n:"%s",g:%s,a:%s,f:%s,s:"%s",fam:"%s",gs:%s,as:%s}'
            % (m["n"], m["g"], m["a"], m["f"], m["s"], m["fam"], gs_str, as_str)
        )
    return "const DATA=[" + ",".join(parts) + "];"


def update_html(models: list[dict]) -> None:
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        html = f.read()

    new_data = build_data_js(models)
    count = len(models)

    html = re.sub(
        r"const DATA=\[.*?\];",
        new_data,
        html,
        flags=re.DOTALL,
    )

    html = re.sub(
        r'(\d+) models',
        f'{count} models',
        html,
    )

    with open(html_path, "w") as f:
        f.write(html)
    print(f"Updated index.html with {count} models")


if __name__ == "__main__":
    models = fetch_leaderboard_data()
    if models:
        update_html(models)
    else:
        print("No models fetched — index.html not updated")
