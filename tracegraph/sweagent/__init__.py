"""MiniSWEAgent-style runtime bundled for TraceGraph SWE interventions.

The paper experiments used the plain chat-completion ACTION format implemented
here. Harmony/native tool prompting is not part of the reported experiments.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a software engineer tasked with solving a GitHub issue.
You have access to a bash shell to explore the repository, understand the codebase,
locate the relevant code, and make the necessary changes to fix the issue.

## Response Format

Every response must contain exactly TWO sections:

1. **THOUGHT**: Your reasoning about what to do next. Analyze the problem,
   plan your approach, and explain your next step.

2. **ACTION**: A single bash command to execute. This must be wrapped in
   a bash code block.

Example response:

THOUGHT:
I need to find the file that contains the buggy function.

ACTION:
```bash
find . -type f -name "*.py" | xargs grep -l "def process_data"
```

## Important Rules

- Each response must contain exactly ONE bash command in the ACTION section.
- Do NOT use tool calls, JSON commands, or `container.exec`; the harness only
  executes the bash command inside the ACTION code block.
- To submit your solution when done, use the special command:
  ```bash
  submit
  ```
- Do NOT write a final summary unless the ACTION command is `submit`.
- Do NOT claim tests pass unless you have observed the test command output.
- Do NOT run interactive commands (vim, nano, python REPL, etc.).
- Do NOT use `git push` or modify remote state.
- Keep your changes minimal and focused on the issue.
- Always verify your fix with relevant tests before submitting.
- If you need to edit a file, use `sed`, `awk`, or heredoc redirects.
- If a command produces too much output, pipe through `head` or `tail`.
"""


USER_PROMPT_TEMPLATE = """\
Here is the GitHub issue to solve:

<issue>
{problem_statement}
</issue>

You are in the repository root. The repository has already been checked out to the \
correct commit. Explore the repo, understand the issue, make the fix, and run tests \
to verify. When you are confident your fix is correct, use the `submit` command.
"""


OBSERVATION_TEMPLATE = """\
OBSERVATION:
{observation}
"""


def make_system_message() -> dict[str, str]:
    """Return the system message for the text ACTION agent."""
    return {"role": "system", "content": SYSTEM_PROMPT}


def make_user_message(problem_statement: str) -> dict[str, str]:
    """Return the initial user message with the issue description."""
    return {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(problem_statement=problem_statement),
    }


def make_observation_message(observation: str) -> dict[str, str]:
    """Return a user message wrapping a command observation."""
    return {
        "role": "user",
        "content": OBSERVATION_TEMPLATE.format(observation=observation),
    }


def strip_thinking(content: str) -> tuple[str, str]:
    """Separate an optional <think>...</think> block from visible content."""
    match = re.search(r"<think>(.*?)</think>", content or "", re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        visible = content[match.end():].strip()
        return thinking, visible
    return "", content or ""


def _normalize_cmd_argument(cmd: object) -> str:
    if isinstance(cmd, str):
        return cmd.strip()
    if isinstance(cmd, list) and all(isinstance(part, str) for part in cmd):
        if (
            len(cmd) >= 3
            and cmd[0] in {"bash", "sh"}
            and cmd[1] in {"-lc", "-c", "lc", "c"}
        ):
            return cmd[2].strip()
        return shlex.join(cmd).strip()
    return ""


def _command_from_payload(payload: object) -> str:
    if isinstance(payload, dict):
        for key in ("cmd", "command", "command_string", "code", "input", "script"):
            cmd = _normalize_cmd_argument(payload.get(key))
            if cmd:
                return cmd
    return ""


def _command_from_malformed_json(raw: str) -> str:
    match = re.search(r'"cmd"\s*:\s*"', raw)
    if not match:
        return ""
    start = match.end() - 1
    escaped = False
    for idx in range(start + 1, len(raw)):
        char = raw[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char != '"':
            continue
        suffix = raw[idx + 1:].strip()
        if suffix and not suffix.startswith(("}", "]}", ",", "],")):
            continue
        try:
            return _normalize_cmd_argument(json.loads(raw[start:idx + 1]))
        except json.JSONDecodeError:
            return ""
    return ""


def _extract_json_command(text: str) -> tuple[str, str]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        raw = text[match.start():]
        try:
            payload, _ = decoder.raw_decode(raw)
        except json.JSONDecodeError:
            cmd = _command_from_malformed_json(raw)
            if cmd:
                return text[:match.start()].strip(), cmd
            continue
        cmd = _command_from_payload(payload)
        if cmd:
            return text[:match.start()].strip(), cmd
    return "", ""


def _extract_loose_code_action(text: str) -> tuple[str, str]:
    match = re.search(
        r"```(?:bash|sh)?\s*\n(.*?)```",
        text or "",
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return "", ""
    return text[:match.start()].strip(), match.group(1).strip()


def _normalize_action(action: str) -> str:
    normalized = (action or "").strip()
    for _ in range(2):
        fence_match = re.fullmatch(
            r"```(?:bash|sh)?\s*\n?(.*?)\n?```",
            normalized,
            re.DOTALL | re.IGNORECASE,
        )
        if not fence_match:
            break
        normalized = fence_match.group(1).strip()
    normalized = re.sub(
        r"^```(?:bash|sh)?\s*\n",
        "",
        normalized,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    if normalized.endswith("```"):
        normalized = normalized[:-3].strip()
    shell_match = re.fullmatch(
        r"(?:bash|sh)\s+(?:-lc|-c|lc|c)\s+(.+)",
        normalized,
        re.DOTALL,
    )
    if shell_match:
        normalized = shell_match.group(1).strip()
    return normalized


def parse_response(
    content: str,
    reasoning: str = "",
    raw_completion_text: str = "",
) -> tuple[str, str, str]:
    """Parse model output into (thinking, thought, bash action)."""
    thinking, visible = strip_thinking(content)
    if reasoning and not thinking:
        thinking = reasoning.strip()

    thought = ""
    action = ""
    thought_match = re.search(
        r"THOUGHT:\s*(.*?)(?=ACTION:|$)", visible, re.DOTALL | re.IGNORECASE
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(
        r"ACTION:\s*```(?:bash|sh)?\s*\n(.*?)```",
        visible,
        re.DOTALL | re.IGNORECASE,
    )
    if action_match:
        action = action_match.group(1).strip()
    else:
        action_match = re.search(
            r"ACTION:\s*(.+?)$", visible, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip()

    if not action:
        loose_thought, loose_action = _extract_loose_code_action(visible)
        if loose_action:
            thought = thought or loose_thought
            action = loose_action
    if not action and reasoning:
        loose_thought, loose_action = _extract_loose_code_action(reasoning)
        if loose_action:
            thought = thought or loose_thought
            action = loose_action
    if not action and raw_completion_text:
        raw_thought, raw_action = _extract_json_command(raw_completion_text)
        if raw_action:
            thought = thought or raw_thought or visible.strip()
            action = raw_action

    return thinking, thought, _normalize_action(action) if action else action


@dataclass
class TokenLogprob:
    """Logprob info for a single generated token."""

    token: str
    token_id: int
    logprob: float
    top_logprobs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelResponse:
    """Full response from an OpenAI-compatible model endpoint."""

    content: str
    logprobs: list[TokenLogprob] | None = None
    prompt_logprobs: list[TokenLogprob] | None = None
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    inference_time: float = 0.0
    finish_reason: str | None = None
    reasoning: str = ""
    raw_logprobs: Any = None
    raw_prompt_logprobs: Any = None
    prompt_token_ids: list[int] | None = None
    routed_experts: list | None = None
    raw_completion_text: str = ""


class VLLMModel:
    """OpenAI-compatible vLLM chat-completions client used by the runner."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "YOUR_MODEL_NAME_OR_PATH",
        top_logprobs: int = 20,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int | None = None,
        max_tokens: int | None = None,
        request_timeout: float = 420.0,
        max_retries: int = 6,
        per_endpoint_max_concurrency: int | None = 8,
        record_prompt_tokens: bool = False,
        prompt_logprobs: int | None = None,
        return_token_ids: bool | None = None,
        reasoning_effort: str | None = None,
    ):
        urls = [item.strip() for item in str(base_url).split(",") if item.strip()]
        if not urls:
            raise ValueError("base_url must contain at least one endpoint")
        self.clients = [
            openai.OpenAI(base_url=url, api_key="dummy", timeout=request_timeout, max_retries=0)
            for url in urls
        ]
        self.base_urls = urls
        if per_endpoint_max_concurrency is not None and per_endpoint_max_concurrency <= 0:
            per_endpoint_max_concurrency = None
        self._endpoint_semaphores = [
            threading.BoundedSemaphore(per_endpoint_max_concurrency)
            if per_endpoint_max_concurrency is not None else None
            for _ in urls
        ]
        self._rr_idx = 0
        self._rr_lock = threading.Lock()
        self.model_name = model_name
        self.top_logprobs = top_logprobs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.record_prompt_tokens = record_prompt_tokens
        self.prompt_logprobs = prompt_logprobs
        self.return_token_ids = record_prompt_tokens if return_token_ids is None else return_token_ids
        self.reasoning_effort = reasoning_effort

    def query(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        top_logprobs: int | None = None,
    ) -> ModelResponse:
        temp = self.temperature if temperature is None else float(temperature)
        nucleus = self.top_p if top_p is None else float(top_p)
        max_tok = self.max_tokens if max_tokens is None else max_tokens
        top_lp = self.top_logprobs if top_logprobs is None else top_logprobs

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "top_p": nucleus,
        }
        extra_body: dict[str, Any] = {}
        if top_lp and top_lp > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = int(top_lp)
        if self.reasoning_effort:
            extra_body["reasoning_effort"] = self.reasoning_effort
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.return_token_ids:
            extra_body["return_token_ids"] = True
        if extra_body:
            kwargs["extra_body"] = extra_body
        if max_tok is not None:
            kwargs["max_tokens"] = int(max_tok)

        with self._rr_lock:
            start = self._rr_idx
            self._rr_idx += 1

        response = None
        started = time.time()
        endpoint_count = len(self.clients)
        for attempt in range(self.max_retries + 1):
            endpoint_idx = (start + attempt) % endpoint_count
            client = self.clients[endpoint_idx]
            semaphore = self._endpoint_semaphores[endpoint_idx]
            try:
                if semaphore is not None:
                    semaphore.acquire()
                try:
                    response = client.chat.completions.create(**kwargs)
                finally:
                    if semaphore is not None:
                        semaphore.release()
                break
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
            ) as exc:
                if attempt >= self.max_retries:
                    raise
                sleep_s = (
                    0.0
                    if endpoint_count > 1 and attempt < endpoint_count - 1
                    else min(60.0, 15.0 * (attempt + 1))
                )
                logger.warning(
                    "vLLM request failed (%s) on %s, retry %d/%d in %.0fs",
                    type(exc).__name__,
                    self.base_urls[endpoint_idx],
                    attempt + 1,
                    self.max_retries,
                    sleep_s,
                )
                if sleep_s:
                    time.sleep(sleep_s)

        if response is None:
            raise RuntimeError("vLLM request failed without a response")

        inference_time = time.time() - started
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", None) or ""
        raw_logprobs = getattr(choice, "logprobs", None)
        parsed_logprobs = self._parse_completion_logprobs(raw_logprobs)

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }

        return ModelResponse(
            content=content,
            logprobs=parsed_logprobs,
            request_id=getattr(response, "id", None),
            usage=usage,
            inference_time=inference_time,
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning=reasoning,
            raw_logprobs=raw_logprobs,
            routed_experts=getattr(choice, "routed_experts", None),
        )

    @staticmethod
    def _parse_completion_logprobs(raw_logprobs: Any) -> list[TokenLogprob] | None:
        content_logprobs = getattr(raw_logprobs, "content", None)
        if not content_logprobs:
            return None
        parsed: list[TokenLogprob] = []
        for token_lp in content_logprobs:
            top_lps = []
            for candidate in getattr(token_lp, "top_logprobs", None) or []:
                entry = {
                    "token": getattr(candidate, "token", ""),
                    "logprob": getattr(candidate, "logprob", -float("inf")),
                }
                token_id = getattr(candidate, "token_id", None)
                if token_id is not None:
                    entry["token_id"] = token_id
                top_lps.append(entry)
            parsed.append(
                TokenLogprob(
                    token=getattr(token_lp, "token", ""),
                    token_id=int(getattr(token_lp, "token_id", -1) or -1),
                    logprob=float(getattr(token_lp, "logprob", -float("inf"))),
                    top_logprobs=top_lps,
                )
            )
        return parsed


def _truncate(text: str, max_chars: int = 10000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - max_chars} characters] ...\n\n"
        + text[-half:]
    )


class DockerEnvironment:
    """Execute bash commands inside a SWE-bench Docker container."""

    def __init__(
        self,
        image: str,
        container_name: str,
        timeout: int = 30,
        max_output_chars: int = 10000,
    ):
        self.image = image
        self.container_name = container_name
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self._base_commit: str | None = None
        self._started = False
        self.last_raw_output = ""

    def start(self) -> None:
        if self._started:
            return

        wait_timeout = int(os.environ.get("TRACEGRAPH_WAIT_FOR_IMAGE_TIMEOUT", "0") or 0)
        if wait_timeout > 0:
            deadline = time.time() + wait_timeout
            while True:
                image_check = subprocess.run(
                    ["docker", "image", "inspect", self.image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if image_check.returncode == 0:
                    break
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Image not available after waiting {wait_timeout}s: {self.image}"
                    )
                logger.info("Waiting for Docker image to be built: %s", self.image)
                time.sleep(60)

        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                self.image,
                "tail",
                "-f",
                "/dev/null",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self._started = True
        out = self.execute("cd /testbed && git rev-parse HEAD~1", timeout=10)
        self._base_commit = out.strip().split("\n")[0] if out.strip() else None
        if self._base_commit:
            self.execute(f"cd /testbed && git checkout {self._base_commit}", timeout=10)

    def execute(self, command: str, timeout: int | None = None) -> str:
        if not self._started:
            self.start()
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/testbed",
                    self.container_name,
                    "bash",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = f"[Command timed out after {effective_timeout}s]: {command}"
        except Exception as exc:
            output = f"[Command failed]: {exc}"
        self.last_raw_output = output
        return _truncate(output, self.max_output_chars)

    def get_patch(self) -> str:
        if self._base_commit:
            return self.execute(f"cd /testbed && git diff {self._base_commit}", timeout=30)
        return self.execute("cd /testbed && git diff", timeout=30)

    def reset(self) -> None:
        if self._base_commit:
            self.execute(f"cd /testbed && git checkout {self._base_commit} -- .", timeout=30)
            self.execute("cd /testbed && git clean -fd", timeout=30)

    def stop(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
            self._started = False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


class Hook(Protocol):
    """Protocol for optional agent step hooks."""

    def on_step(
        self,
        step: int,
        response: ModelResponse,
        thinking: str,
        thought: str,
        action: str,
        observation: str,
        observation_full: str | None = None,
    ) -> None: ...

    def on_done(self, patch: str, reason: str) -> None: ...


@dataclass
class StepRecord:
    """Record of a single SWE agent step."""

    step_index: int
    thinking: str
    thought: str
    action: str
    observation: str
    response: ModelResponse
    observation_full: str | None = None


@dataclass
class AgentResult:
    """Final result of an agent run."""

    patch: str
    steps: list[StepRecord] = field(default_factory=list)
    exit_reason: str = ""


def _assistant_history_content(response: ModelResponse, thought: str, action: str) -> str:
    if response.content.strip():
        return response.content
    if action:
        return f"THOUGHT:\n{thought or response.reasoning}\n\nACTION:\n```bash\n{action}\n```"
    return response.reasoning or response.content


class Agent:
    """Minimal iterative LLM + bash agent used as a base class."""

    def __init__(
        self,
        model: VLLMModel,
        env: DockerEnvironment,
        hooks: list[Hook] | None = None,
        max_steps: int = 30,
    ):
        self.model = model
        self.env = env
        self.hooks = hooks or []
        self.max_steps = max_steps

    def run(self, problem_statement: str) -> AgentResult:
        messages = [make_system_message(), make_user_message(problem_statement)]
        steps: list[StepRecord] = []
        exit_reason = "max_steps"
        for step in range(self.max_steps):
            try:
                response = self.model.query(messages)
            except Exception as exc:
                exit_reason = f"error: {exc}"
                break

            thinking, thought, action = parse_response(
                response.content, response.reasoning, response.raw_completion_text
            )
            if not action:
                observation = (
                    "[ERROR] Could not parse an ACTION from your response. "
                    "Please respond with THOUGHT: and ACTION: sections, "
                    "with the action in a ```bash code block."
                )
                messages.append(
                    {"role": "assistant", "content": _assistant_history_content(response, thought, action)}
                )
                messages.append(make_observation_message(observation))
                steps.append(
                    StepRecord(
                        step_index=step,
                        thinking=thinking,
                        thought=thought,
                        action="",
                        observation=observation,
                        observation_full=observation,
                        response=response,
                    )
                )
                continue

            if action.strip().lower() == "submit":
                if not self.env.get_patch().strip():
                    observation = "[ERROR] Cannot submit because `git diff` is empty."
                    messages.append(
                        {"role": "assistant", "content": _assistant_history_content(response, thought, action)}
                    )
                    messages.append(make_observation_message(observation))
                    steps.append(
                        StepRecord(
                            step_index=step,
                            thinking=thinking,
                            thought=thought,
                            action="submit",
                            observation=observation,
                            observation_full=observation,
                            response=response,
                        )
                    )
                    continue
                exit_reason = "submit"
                steps.append(
                    StepRecord(
                        step_index=step,
                        thinking=thinking,
                        thought=thought,
                        action="submit",
                        observation="",
                        observation_full="",
                        response=response,
                    )
                )
                break

            observation = self.env.execute(action)
            observation_full = getattr(self.env, "last_raw_output", observation)
            steps.append(
                StepRecord(
                    step_index=step,
                    thinking=thinking,
                    thought=thought,
                    action=action,
                    observation=observation,
                    observation_full=observation_full,
                    response=response,
                )
            )
            for hook in self.hooks:
                hook.on_step(step, response, thinking, thought, action, observation, observation_full)
            messages.append(
                {"role": "assistant", "content": _assistant_history_content(response, thought, action)}
            )
            messages.append(make_observation_message(observation))

        patch = self.env.get_patch()
        for hook in self.hooks:
            hook.on_done(patch, exit_reason)
        return AgentResult(patch=patch, steps=steps, exit_reason=exit_reason)


__all__ = [
    "Agent",
    "AgentResult",
    "DockerEnvironment",
    "ModelResponse",
    "StepRecord",
    "TokenLogprob",
    "VLLMModel",
    "make_observation_message",
    "make_system_message",
    "make_user_message",
    "parse_response",
    "strip_thinking",
]
