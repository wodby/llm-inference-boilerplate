import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest

from inference import InferenceError, endpoint, events, infer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.server.requests.append((self.path, self.headers, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        status, content_type, body = self.server.reply
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if status == 307:
            self.send_header('Location', '/v1/redirect-target')
        self.end_headers()
        self.wfile.write(body)


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.requests = []
        self.server.reply = (200, 'application/json', b'{"choices":[{"message":{"content":"Hello"}}],"usage":{"total_tokens":7}}')
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.output = io.StringIO()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def call(self, **kwargs):
        return infer(self.base, 'test-secret', 'model', 'Hello?', output=self.output, **kwargs)

    def test_completion(self):
        self.assertEqual(self.call(no_thinking=True), {'total_tokens': 7})
        self.assertEqual(self.output.getvalue(), 'Hello\n')
        path, headers, payload = self.server.requests[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(headers['Authorization'], 'Bearer test-secret')
        self.assertEqual(payload, {'model': 'model', 'messages': [{'role': 'user', 'content': 'Hello?'}],
                                   'stream': False, 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}})

    def test_stream(self):
        self.server.reply = (200, 'text/event-stream', (
            ': keepalive\r\n\r\ndata: {"choices":[{"index":0,"delta":{"content":"Hé"}}]}\r\n\r\n'
            'data: {"choices":[{"index":0,"delta":{"content":"llo"}}]}\n\n'
            'data: {"choices":[],\ndata: "usage":{"total_tokens":9}}\n\n'
            'data: [DONE]\n\n').encode())
        self.assertEqual(self.call(stream=True), {'total_tokens': 9})
        self.assertEqual(self.output.getvalue(), 'Héllo\n')
        self.assertEqual(self.server.requests[0][2]['stream_options'], {'include_usage': True})

    def test_partial_stream_is_not_retried(self):
        self.server.reply = (200, 'text/event-stream', b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        with self.assertRaisesRegex(InferenceError, 'before.*DONE'):
            self.call(stream=True)
        self.assertEqual(self.output.getvalue(), 'partial')
        self.assertEqual(len(self.server.requests), 1)

    def test_no_secret_or_response_body_in_errors(self):
        for status in (401, 429, 503, 307):
            with self.subTest(status=status):
                self.server.reply = (status, 'text/plain', b'test-secret Hello? internal-provider-details')
                with self.assertRaises(InferenceError) as raised:
                    self.call()
                self.assertIn(str(status), str(raised.exception))
                self.assertNotIn('test-secret', str(raised.exception))
                self.assertNotIn('Hello?', str(raised.exception))
                self.assertNotIn('internal-provider', str(raised.exception))
        self.assertEqual(len(self.server.requests), 4, 'no retries or redirect follow-up')

    def test_invalid_response(self):
        for body in (b'not-json', b'{}', b'[]', b'{"error":"test-secret"}'):
            self.server.reply = (200, 'application/json', body)
            with self.assertRaises(InferenceError) as raised:
                self.call()
            self.assertNotIn('test-secret', str(raised.exception))

    def test_invalid_inputs_do_not_send_requests(self):
        for kwargs in ({'max_tokens': 0}, {'timeout': -1}, {'timeout': float('nan')}, {'timeout': float('inf')}):
            with self.assertRaises(InferenceError):
                self.call(**kwargs)
        with self.assertRaises(InferenceError):
            infer(self.base, '', 'model', 'prompt')
        self.assertEqual(self.server.requests, [])


class ProtocolTest(unittest.TestCase):
    def test_url_policy(self):
        for url in ('http://example.com/v1', 'https://a:secret@example.com/v1', 'https://example.com/v1?x=1',
                    'https://example.com/v1#f', 'https://example.com', 'file:///v1', 'https://example.com:bad/v1', ''):
            with self.subTest(url=url), self.assertRaises(InferenceError):
                endpoint(url)
        self.assertEqual(endpoint('https://example.com/prefix/v1/'), 'https://example.com/prefix/v1/chat/completions')
        self.assertEqual(endpoint('http://[::1]:8000/v1'), 'http://[::1]:8000/v1/chat/completions')

    def test_bad_sse(self):
        for body in (b'data: not-json\n\n', b'data: \xff\n\n', b'data: {}', b'data: ' + b'x' * 1048576):
            with self.subTest(length=len(body)), self.assertRaises(InferenceError):
                list(events(io.BytesIO(body)))


if __name__ == '__main__':
    unittest.main()
