from __future__ import annotations

import asyncio
import json
import threading
import unittest

from fastapi.responses import StreamingResponse

from services.log_service import LoggedCall
from utils.helper import sse_json_stream


class LoggedCallStreamingTests(unittest.TestCase):
    def test_streaming_run_does_not_prefetch_first_item(self):
        consumed = False

        def handler():
            def events():
                nonlocal consumed
                consumed = True
                yield {"ok": True}

            return events()

        call = LoggedCall(
            {"id": "test-key", "name": "test", "role": "test"},
            "/v1/chat/completions",
            "auto",
            "测试流式调用",
        )

        response = asyncio.run(call.run(handler))

        self.assertIsInstance(response, StreamingResponse)
        self.assertFalse(consumed)
        self.assertEqual(response.headers.get("cache-control"), "no-cache")
        self.assertEqual(response.headers.get("x-accel-buffering"), "no")

    def test_sse_json_stream_sends_open_comment_before_items(self):
        consumed = False

        def events():
            nonlocal consumed
            consumed = True
            yield {"ok": True}

        stream = sse_json_stream(events())

        self.assertEqual(next(stream), ": stream-open\n\n")
        self.assertFalse(consumed)

    def test_sse_json_stream_sends_heartbeat_while_waiting(self):
        release = threading.Event()

        def events():
            release.wait(timeout=1)
            yield {"ok": True}

        stream = sse_json_stream(events(), heartbeat_interval=0.01)

        self.assertEqual(next(stream), ": stream-open\n\n")
        heartbeat = next(stream)
        self.assertTrue(heartbeat.startswith(": ping "))
        self.assertIn('"event":"heartbeat"', heartbeat)
        release.set()

        payload = next(stream)
        self.assertTrue(payload.startswith("data: "))
        self.assertEqual(json.loads(payload[6:].strip()), {"ok": True})

    def test_sse_json_stream_heartbeat_includes_last_progress(self):
        release = threading.Event()

        def events():
            yield {
                "object": "image.generation.chunk",
                "index": 1,
                "total": 2,
                "progress_text": "正在生成草图",
                "upstream_event_type": "conversation.delta",
            }
            release.wait(timeout=1)
            yield {"object": "image.generation.result", "data": []}

        stream = sse_json_stream(events(), heartbeat_interval=0.01)

        self.assertEqual(next(stream), ": stream-open\n\n")
        first_payload = next(stream)
        self.assertTrue(first_payload.startswith("data: "))
        heartbeat = next(stream)
        self.assertTrue(heartbeat.startswith(": ping "))
        self.assertIn('"progress_text":"正在生成草图"', heartbeat)
        self.assertIn('"index":1', heartbeat)
        self.assertIn('"total":2', heartbeat)
        release.set()


if __name__ == "__main__":
    unittest.main()
