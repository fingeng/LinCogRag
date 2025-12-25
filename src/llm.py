"""
OpenAI-compatible LLM wrapper.

项目历史上存在两套 LLM wrapper：
- `src/llm.py::LLM`（主流程使用）
- `src/utils.py::LLM_Model`（旧实现，支持 base_url/httpx）

为避免维护分叉，这里把能力统一到 `LLM`：
- 支持 `OPENAI_API_KEY`
- 支持 `OPENAI_BASE_URL`（可选）
- 使用 httpx client 控制超时，并关闭 trust_env 以避免代理/环境变量干扰
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


class LLM:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        http_client = httpx.Client(timeout=timeout_s, trust_env=False)
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

        print(f"[LLM] Initialized with model: {self.model_name}" + (f" (base_url={base_url})" if base_url else ""))

    def infer(self, messages: List[Dict[str, Any]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return "Error: Could not get LLM response"
