from __future__ import annotations

import asyncio
import json
import threading
import unittest

from fastapi.responses import StreamingResponse

from services.protocol.openai_v1_chat_complete import stream_image_chat_completion
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
        self.assertIn('"heartbeat_index":1', heartbeat)
        release.set()

        payload = next(stream)
        self.assertTrue(payload.startswith("data: "))
        self.assertEqual(json.loads(payload[6:].strip()), {"ok": True})

    def test_sse_json_stream_sends_data_heartbeat_after_data_item(self):
        release = threading.Event()

        def events():
            yield {"object": "chat.completion.chunk", "id": "chatcmpl_test", "model": "gpt-image-2", "choices": []}
            release.wait(timeout=1)
            yield {"ok": True}

        stream = sse_json_stream(events(), heartbeat_interval=0.01)

        self.assertEqual(next(stream), ": stream-open\n\n")
        first_payload = next(stream)
        self.assertTrue(first_payload.startswith("data: "))
        heartbeat_data = next(stream)
        self.assertTrue(heartbeat_data.startswith("data: "))
        heartbeat = json.loads(heartbeat_data[6:].strip())
        self.assertEqual(heartbeat.get("object"), "chat.completion.chunk")
        self.assertEqual((heartbeat.get("x_heartbeat") or {}).get("heartbeat_index"), 1)
        heartbeat_comment = next(stream)
        self.assertTrue(heartbeat_comment.startswith(": ping "))
        release.set()

    def test_image_chat_stream_sends_initial_data_before_upstream(self):
        consumed = False

        def outputs():
            nonlocal consumed
            consumed = True
            yield from ()

        stream = stream_image_chat_completion(outputs(), "gpt-image-2")
        first = next(stream)

        self.assertEqual(first.get("object"), "chat.completion.chunk")
        delta = (((first.get("choices") or [{}])[0]).get("delta") or {})
        self.assertEqual(delta.get("role"), "assistant")
        self.assertEqual(delta.get("content"), "")
        self.assertFalse(consumed)


if __name__ == "__main__":
    unittest.main()
