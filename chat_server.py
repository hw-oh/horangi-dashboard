#!/usr/bin/env python3
"""Horangi Leaderboard chat server with W&B Weave tracing."""

import json
import os

import weave
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "wandb-korea")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "horangi4")

weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@weave.op()
def build_analysis_context(leaderboard_csv: str, selected_models: list[str]) -> str:
    sel_text = "현재 사용자가 선택한 모델: " + ", ".join(selected_models) if selected_models else "선택된 모델 없음"
    return (
        "당신은 Horangi 한국어 LLM 리더보드 분석 전문가입니다. 아래 데이터를 기반으로 한국어로 분석해주세요.\n\n"
        "## 평가 체계\n"
        "- FINAL = (GLP+ALT)/2, GLP=범용언어성능(11개 하위항목 평균), ALT=가치정렬성능(5개 하위항목 평균)\n"
        "- GLP 하위: 구문해석, 의미해석, 표현, 정보검색, 일반적지식, 전문적지식, 수학적추론, 논리적추론, 추상적추론, 함수호출, 코딩능력\n"
        "- ALT 하위: 제어성, 유해성방지, 편향성방지, 윤리/도덕, 환각방지\n"
        "- Tier: Frontier(≥75), Strong(65-75), Competitive(55-65), Emerging(<55)\n"
        "- Size: XS(<5B), S(5-15B), M(15-50B), L(50B+), API(API전용)\n\n"
        f"## {sel_text}\n\n"
        f"## 전체 모델 데이터\n```\n{leaderboard_csv}```\n\n"
        "답변 시 구체적 수치를 인용하고, 비교분석에는 표를 활용하세요. 간결하게 답변하세요."
    )


@weave.op()
def horangi_leaderboard_chat(
    user_message: str,
    history: list[dict],
    system_prompt: str,
    openai_api_key: str,
    model: str = "gpt-5.4-mini",
) -> str:
    client = OpenAI(api_key=openai_api_key)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    openai_key = body.get("openai_api_key", "")
    user_msg = body.get("message", "")
    history = body.get("history", [])
    leaderboard_csv = body.get("leaderboard_csv", "")
    selected = body.get("selected_models", [])
    stream = body.get("stream", True)

    system_prompt = build_analysis_context(leaderboard_csv, selected)

    if stream:
        client = OpenAI(api_key=openai_key)
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_msg}]

        with weave.attributes({"source": "dashboard", "user_message": user_msg}):
            response = client.chat.completions.create(model="gpt-5.4-mini", messages=messages, stream=True)

        def generate():
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    result = horangi_leaderboard_chat(user_msg, history, system_prompt, openai_key)
    return {"content": result}


if __name__ == "__main__":
    import uvicorn
    print(f"Weave tracing → https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/weave")
    uvicorn.run(app, host="0.0.0.0", port=8000)
