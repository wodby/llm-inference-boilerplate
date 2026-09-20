"""Small synchronous and streaming client for a self-hosted chat-completions API."""
import argparse
from http.client import HTTPException
import ipaddress
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024


class InferenceError(Exception):
    """Safe-to-display failure without request content, response bodies or credentials."""


class NoRedirects(urllib.request.HTTPRedirectHandler):
    """Do not forward credentials or retry an inference request through redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint(base_url):
    """Require HTTPS except for loopback development; reject embedded credentials."""
    try:
        parsed = urllib.parse.urlsplit(base_url)
        port = parsed.port  # Validate malformed port numbers too.
        del port
        loopback = parsed.hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            pass
        if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or any(c.isspace() for c in base_url)
                or parsed.scheme not in ("https", "http")
                or (parsed.scheme == "http" and not loopback)):
            raise ValueError
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError
    except ValueError:
        raise InferenceError("LLM_BASE_URL must be an HTTPS API base ending in /v1 (HTTP is allowed only on loopback).") from None
    return base_url.rstrip("/") + "/chat/completions"


def events(response):
    """Read bounded SSE events, preserving multi-line data and ignoring comments."""
    data = []
    size = 0
    while True:
        raw = response.readline(MAX_EVENT_BYTES + 1)
        if not raw:
            # An unfinished event is not a successfully terminated completion.
            raise InferenceError("The stream ended before [DONE]; output may be partial. It was not retried.")
        size += len(raw)
        if size > MAX_EVENT_BYTES:
            raise InferenceError("The server sent an oversized streaming event.")
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            raise InferenceError("The server sent invalid UTF-8.") from None
        if not line:
            if data:
                payload = "\n".join(data)
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except (ValueError, RecursionError):
                    raise InferenceError("The server sent an invalid streaming event.") from None
            data, size = [], 0
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


def infer(base_url, api_key, model, prompt, *, stream=False, max_tokens=256,
          timeout=120, no_thinking=False, output=None):
    """Send one request without automatic retries; return provider-reported usage."""
    url = endpoint(base_url)
    if not api_key or any(c.isspace() for c in api_key):
        raise InferenceError("Set LLM_API_KEY to the inference service's API key.")
    if not model or not prompt or max_tokens < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise InferenceError("Model, prompt, positive max tokens and a finite positive timeout are required.")
    output = output if output is not None else sys.stdout
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": stream}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + api_key,
                                              "Content-Type": "application/json",
                                              "Accept": "text/event-stream" if stream else "application/json"})
    usage = None
    try:
        with urllib.request.build_opener(NoRedirects()).open(request, timeout=timeout) as response:
            if stream:
                if response.headers.get_content_type() != "text/event-stream":
                    raise InferenceError("Expected an SSE stream from the inference endpoint.")
                for event in events(response):
                    if not isinstance(event, dict) or "error" in event:
                        raise InferenceError("The server reported a streaming error; output may be partial.")
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) == 0:
                            text = choice.get("delta", {}).get("content")
                            if text:
                                output.write(text)
                                output.flush()
                    usage = event.get("usage") or usage
            else:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise InferenceError("The inference response exceeded the client size limit.")
                result = json.loads(raw)
                if "error" in result:
                    raise InferenceError("The inference server reported an error.")
                content = result["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise InferenceError("The inference response did not contain text.")
                output.write(content)
                usage = result.get("usage")
    except urllib.error.HTTPError as error:
        error.close()
        raise InferenceError(f"Inference returned HTTP {error.code}; check access, model and capacity. The request was not retried.") from None
    except (urllib.error.URLError, TimeoutError, OSError, HTTPException):
        raise InferenceError("Inference connection failed or timed out; output may be partial. The request was not retried.") from None
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        raise InferenceError("The inference server returned an invalid response.") from None
    output.write("\n")
    return usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt text; omit to read from stdin")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120, help="Socket timeout in seconds, not a total stream deadline")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking for compatible Qwen chat templates")
    args = parser.parse_args(argv)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    try:
        usage = infer(os.getenv("LLM_BASE_URL", ""), os.getenv("LLM_API_KEY", ""),
                      os.getenv("LLM_MODEL", "model"), prompt, stream=args.stream,
                      max_tokens=args.max_tokens, timeout=args.timeout, no_thinking=args.no_thinking)
    except InferenceError as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Inference interrupted; the server may have generated billable tokens.", file=sys.stderr)
        return 130
    if isinstance(usage, dict):
        # Only print recognized counters, never arbitrary server metadata.
        counters = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0}
        if counters:
            print(json.dumps({"usage": counters}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
