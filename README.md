# LLM inference boilerplate

A dependency-free Python 3.10+ client for the chat-completions API exposed by
self-hosted vLLM. It supports ordinary responses, streaming and token usage.
It calls an existing endpoint; it does not download a model, deploy a server or
purchase GPU capacity.

## Quick start

Deploy the [vLLM stack](https://github.com/wodby/stack-vllm) after its prerequisites
are met. Run the client on a machine that can reach the app's private network.
Use the app's HTTPS address with `/v1` appended and its generated API key.

```sh
export LLM_BASE_URL=https://your-private-inference-host.example/v1
export LLM_MODEL=model
# Set LLM_API_KEY through your secret manager or shell environment.
python3 inference.py --no-thinking 'Explain what a GPU does in one sentence.'
python3 inference.py --stream --no-thinking 'Write a short welcome message.'
```

`--no-thinking` is useful for the Qwen3 smoke-test preset. Omit it for models
whose chat templates do not support this option. The API model name is the
server's served-model name, not necessarily its Hugging Face repository name.
No OpenAI account or OpenAI API key is used.

Omit the prompt argument to read it from standard input, avoiding prompt text
in shell history/process arguments. The API key is accepted only through the
environment, never as a command-line argument. `.env.example` documents the
variables; the script does not automatically load `.env` files.

## Use from Python

```python
import os
from inference import infer

usage = infer(
    os.environ['LLM_BASE_URL'],
    os.environ['LLM_API_KEY'],
    os.getenv('LLM_MODEL', 'model'),
    'Explain Kubernetes in two sentences.',
    stream=True,
    max_tokens=256,
    no_thinking=True,  # Qwen3 preset only
)
```

Text is written to stdout (or a supplied file-like `output`). The CLI reports
available token counters on stderr. These are server-reported counters, not
an invoice or a cost estimate. Missing usage is not treated as zero.

## Failure and security behavior

- HTTPS is required except for loopback development URLs. Certificate checks
  are not disabled. The base URL must end in `/v1`, without credentials,
  query parameters or fragments.
- Redirects are rejected so credentials are not forwarded to another endpoint.
- Failed requests are not retried automatically: a retry can duplicate output
  and GPU cost. A missing streaming `[DONE]` is an error even if some text arrived.
- `--timeout` controls socket operations, not a total generation deadline.
  Long-running active streams can exceed it. Interrupting the client does not
  guarantee the server immediately stops generation or billing.
- Error messages omit server response bodies, prompts and credentials. Model
  output itself is untrusted and should not be executed as code.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Tests use a loopback fake API, with no model download, external request or GPU
allocation. They cover authentication, payloads, streaming, token usage,
redirect rejection, partial responses and safe error messages. Actual GPU
inference against a deployed service remains a separate smoke test.

This is a standalone client boilerplate, not a Wodby application-server build
boilerplate. It is intentionally not attached to the vLLM runtime's image build.
