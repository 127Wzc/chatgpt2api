"""OpenAI Sentinel token helpers.

- 旧接口：保留密码登录等仍在使用的 legacy sentinel token 逻辑。
- 新接口：注册 create_account 阶段改为先拿 live sentinel requirements，
  再调用官方 SDK 生成最终的 Sentinel token / SO token。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curl_cffi.requests import Session


class SentinelTokenGenerator:
    """Sentinel Token 生成器（PoW - Proof of Work）。"""
    MAX_ATTEMPTS = 500_000
    ERROR_PREFIX = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D"

    def __init__(self, device_id: str, ua: str):
        self.device_id = device_id
        self.user_agent = ua
        self.sid = str(uuid.uuid4())

    @staticmethod
    def _fnv1a_32(text: str) -> str:
        h = 2166136261
        for ch in text:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 2246822507) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 3266489909) & 0xFFFFFFFF
        h ^= h >> 16
        return format(h & 0xFFFFFFFF, "08x")

    def _get_config(self) -> list:
        perf_now = random.uniform(1000, 50000)
        return [
            "1920x1080",
            time.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)", time.gmtime()),
            4294705152,
            random.random(),
            self.user_agent,
            "https://sentinel.openai.com/sentinel/20260124ceb8/sdk.js",
            None,
            None,
            "en-US",
            random.random(),
            random.choice(["vendorSub-undefined", "plugins-undefined", "mimeTypes-undefined", "hardwareConcurrency-undefined"]),
            random.choice(["location", "implementation", "URL", "documentURI", "compatMode"]),
            random.choice(["Object", "Function", "Array", "Number", "parseFloat", "undefined"]),
            perf_now,
            self.sid,
            "",
            random.choice([4, 8, 12, 16]),
            time.time() * 1000 - perf_now,
        ]

    @staticmethod
    def _b64(data) -> str:
        return base64.b64encode(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).decode("ascii")

    def generate_requirements_token(self) -> str:
        data = self._get_config()
        data[3] = 1
        data[9] = round(random.uniform(5, 50))
        return "gAAAAAC" + self._b64(data)

    def generate_token(self, seed: str, difficulty: str) -> str:
        start = time.time()
        data = self._get_config()
        difficulty = str(difficulty or "0")
        for i in range(self.MAX_ATTEMPTS):
            data[3] = i
            data[9] = round((time.time() - start) * 1000)
            payload = self._b64(data)
            if self._fnv1a_32(seed + payload)[: len(difficulty)] <= difficulty:
                return "gAAAAAB" + payload + "~S"
        return "gAAAAAB" + self.ERROR_PREFIX + self._b64(str(None))


# ── 默认 User-Agent 和 sec-ch-ua ──────────────────────────────
DEFAULT_SENTINEL_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)
DEFAULT_SENTINEL_SEC_CH_UA = '"Chromium";v="145", "Google Chrome";v="145", "Not/A)Brand";v="99"'
DEFAULT_SENTINEL_BOOTSTRAP_URL = "https://sentinel.openai.com/backend-api/sentinel/sdk.js"
DEFAULT_SENTINEL_REQ_URL = "https://sentinel.openai.com/backend-api/sentinel/req"
DEFAULT_SENTINEL_FRAME_REFERER = "https://sentinel.openai.com/backend-api/sentinel/frame.html"
DEFAULT_SO_OBSERVER_WAIT_MS = 5000


@dataclass
class OfficialSentinelResult:
    token: str
    so_token: str
    sdk_url: str = ""
    sdk_version: str = ""
    bootstrap_url: str = DEFAULT_SENTINEL_BOOTSTRAP_URL
    token_length: int = 0
    so_token_length: int = 0
    so_generated: bool = False
    requirements: dict | None = None


def fetch_sentinel_requirements(
    session: "Session",
    device_id: str,
    flow: str,
    *,
    user_agent: str = "",
    sec_ch_ua: str = "",
) -> dict:
    """Fetch live sentinel requirements for a flow.

    注册 create_account 阶段的 requirements 由 live req 接口提供，其中包含：
    - token
    - proofofwork
    - turnstile
    - so
    """
    ua = user_agent or DEFAULT_SENTINEL_USER_AGENT
    ch_ua = sec_ch_ua or DEFAULT_SENTINEL_SEC_CH_UA
    response = session.post(
        DEFAULT_SENTINEL_REQ_URL,
        data=json.dumps({"id": device_id, "flow": flow}, separators=(",", ":")),
        headers={
            "Content-Type": "text/plain;charset=UTF-8",
            "Referer": DEFAULT_SENTINEL_FRAME_REFERER,
            "Origin": "https://sentinel.openai.com",
            "User-Agent": ua,
            "sec-ch-ua": ch_ua,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        },
        timeout=20,
        verify=False,
    )
    try:
        data = response.json() if response.text else {}
    except Exception as exc:
        raise RuntimeError(f"sentinel_req_invalid_json_{response.status_code}") from exc
    if response.status_code != 200 or not isinstance(data, dict) or not str(data.get("token") or "").strip():
        raise RuntimeError(f"sentinel_req_failed_{response.status_code}")
    return data


def _sentinel_runner_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "sentinel_sdk_runner.mjs"


def _run_official_sentinel_sdk(
    *,
    device_id: str,
    flow: str,
    requirements: dict,
    user_agent: str = "",
    observer_wait_ms: int = DEFAULT_SO_OBSERVER_WAIT_MS,
) -> OfficialSentinelResult:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("sentinel_sdk_node_missing")
    script = _sentinel_runner_script()
    if not script.exists():
        raise RuntimeError(f"sentinel_sdk_runner_missing:{script}")
    payload = {
        "deviceId": device_id,
        "flow": flow,
        "requirements": requirements,
        "observerWaitMs": max(0, int(observer_wait_ms or DEFAULT_SO_OBSERVER_WAIT_MS)),
        "bootstrapUrl": DEFAULT_SENTINEL_BOOTSTRAP_URL,
        "userAgent": user_agent or DEFAULT_SENTINEL_USER_AGENT,
    }
    try:
        completed = subprocess.run(
            [node, str(script)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=max(20, 15 + int(observer_wait_ms / 1000) + 15),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("sentinel_sdk_timeout") from exc
    stderr = str(completed.stderr or "").strip()
    stdout = str(completed.stdout or "").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"exit_{completed.returncode}"
        raise RuntimeError(f"sentinel_sdk_failed:{detail[:500]}")
    try:
        data = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError(f"sentinel_sdk_invalid_output:{stdout[:300]}") from exc
    token = str(data.get("token") or "").strip()
    if not token:
        raise RuntimeError("sentinel_sdk_missing_token")
    so_token = str(data.get("so_token") or "").strip()
    return OfficialSentinelResult(
        token=token,
        so_token=so_token,
        sdk_url=str(data.get("sdk_url") or ""),
        sdk_version=str(data.get("sdk_version") or ""),
        bootstrap_url=str(data.get("bootstrap_url") or DEFAULT_SENTINEL_BOOTSTRAP_URL),
        token_length=int(data.get("token_length") or len(token)),
        so_token_length=int(data.get("so_token_length") or len(so_token)),
        so_generated=bool(data.get("so_generated")),
        requirements=requirements,
    )


def build_official_sentinel_result(
    session: "Session",
    device_id: str,
    flow: str,
    *,
    user_agent: str = "",
    sec_ch_ua: str = "",
    observer_wait_ms: int = DEFAULT_SO_OBSERVER_WAIT_MS,
) -> OfficialSentinelResult:
    requirements = fetch_sentinel_requirements(
        session,
        device_id,
        flow,
        user_agent=user_agent,
        sec_ch_ua=sec_ch_ua,
    )
    result = _run_official_sentinel_sdk(
        device_id=device_id,
        flow=flow,
        requirements=requirements,
        user_agent=user_agent,
        observer_wait_ms=observer_wait_ms,
    )
    result.requirements = requirements
    return result


def build_sentinel_token(
    session: "Session",
    device_id: str,
    flow: str,
    *,
    user_agent: str = "",
    sec_ch_ua: str = "",
) -> tuple[str, str]:
    """请求 sentinel token 并返回 (sentinel_header_value, oai_sc_cookie_value)。

    Args:
        session: curl_cffi Session 实例
        device_id: 设备 ID
        flow: 流程标识（如 "password_verify", "username_password_create" 等）
        user_agent: 可选的 User-Agent 覆盖
        sec_ch_ua: 可选的 sec-ch-ua 覆盖

    Returns:
        (openai-sentinel-token header value, oai-sc cookie value) 元组

    Raises:
        RuntimeError: sentinel 请求失败
    """
    ua = user_agent or DEFAULT_SENTINEL_USER_AGENT
    ch_ua = sec_ch_ua or DEFAULT_SENTINEL_SEC_CH_UA
    generator = SentinelTokenGenerator(device_id, ua)
    resp = session.post(
        "https://sentinel.openai.com/backend-api/sentinel/req",
        data=json.dumps({"p": generator.generate_requirements_token(), "id": device_id, "flow": flow}),
        headers={
            "Content-Type": "text/plain;charset=UTF-8",
            "Referer": "https://sentinel.openai.com/backend-api/sentinel/frame.html",
            "Origin": "https://sentinel.openai.com",
            "User-Agent": ua,
            "sec-ch-ua": ch_ua,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        },
        timeout=20,
        verify=False,
    )

    try:
        data = resp.json() if resp.text else {}
    except Exception:
        fallback = json.dumps(
            {"p": generator.generate_requirements_token(), "t": "", "c": "", "id": device_id, "flow": flow},
            separators=(",", ":"),
        )
        return fallback, ""

    token = str(data.get("token") or "").strip()
    if resp.status_code != 200 or not token:
        raise RuntimeError(f"sentinel_req_failed_{resp.status_code}")
    pow_data = data.get("proofofwork") or {}
    p_value = (
        generator.generate_token(str(pow_data.get("seed") or ""), str(pow_data.get("difficulty") or "0"))
        if pow_data.get("required") and pow_data.get("seed")
        else generator.generate_requirements_token()
    )
    sentinel_value = json.dumps({"p": p_value, "t": "", "c": token, "id": device_id, "flow": flow}, separators=(",", ":"))
    # oai-sc cookie = "0" + sentinel token "c" value (the challenge token from the server)
    oai_sc_value = "0" + token
    return sentinel_value, oai_sc_value
