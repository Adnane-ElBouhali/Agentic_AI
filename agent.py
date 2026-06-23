from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import List
from urllib.parse import urlparse

try:
    from hackathon_starter_kit.models import AnswerItem
    from hackathon_starter_kit.models import Input
    from hackathon_starter_kit.tools.gitlab.clone import clone
except ModuleNotFoundError as exc:
    if exc.name != "hackathon_starter_kit":
        raise
    from models import AnswerItem
    from models import Input
    from tools.gitlab.clone import clone


DEFAULT_MODEL = "mistral-medium-2508"
MAX_CONTEXT_CHARS = 120_000
MAX_FILE_CHARS = 12_000
MAX_TOOL_OUTPUT_CHARS = 18_000
MAX_TOOL_ROUNDS = 3

TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".cjs",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".dockerfile",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".ipynb",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lock",
    ".md",
    ".mjs",
    ".php",
    ".ps1",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".rst",
    ".scala",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

IMPORTANT_NAMES = {
    ".env.example",
    ".gitignore",
    "dockerfile",
    "makefile",
    "poetry.lock",
    "pyproject.toml",
    "readme",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}

SKIPPED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass
class RepoSnapshot:
    root: Path
    tree: str
    context: str


def predict(input: Input) -> List[AnswerItem]:
    """Answer all challenge questions for one submission."""

    repo_dir = Path(tempfile.mkdtemp(prefix=f"submission-{input.submission_id}-"))
    try:
        _materialize_repo(input, repo_dir)
        snapshot = _build_snapshot(repo_dir, input.template)
        raw_answers = _answer_with_tools(input, snapshot)
        return _coerce_answers(input, raw_answers)
    except Exception as exc:
        print(f"[agent] Fatal prediction error: {exc}")
        return [_unknown_answer(question.id, f"Agent error: {exc}") for question in input.template]
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)


def _materialize_repo(input: Input, dest: Path) -> None:
    """Clone a remote repo, or copy a local repo when tests provide one."""

    parsed = urlparse(input.repo_url)
    if parsed.scheme == "file":
        source = Path(parsed.path)
        _copy_local_repo(source, dest)
        return

    local_candidate = Path(input.repo_url).expanduser()
    if local_candidate.exists():
        _copy_local_repo(local_candidate, dest)
        return

    clone(input.token_gitlab, input.repo_url, str(dest))


def _copy_local_repo(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(*SKIPPED_DIRS))


def _build_snapshot(root: Path, questions: Iterable[Any]) -> RepoSnapshot:
    files = list(_iter_files(root))
    tree = _format_tree(root, files)
    question_text = " ".join(getattr(question, "question", "") for question in questions)
    ranked = sorted(files, key=lambda path: _file_priority(root, path, question_text), reverse=True)

    chunks: list[str] = []
    total = 0
    for path in ranked:
        if total >= MAX_CONTEXT_CHARS:
            break
        rel = _rel(root, path)
        remaining = MAX_CONTEXT_CHARS - total
        content = _read_text(path, min(MAX_FILE_CHARS, remaining))
        if not content:
            continue
        chunk = f"\n--- FILE: {rel} ---\n{content}\n"
        chunks.append(chunk)
        total += len(chunk)

    return RepoSnapshot(root=root, tree=tree, context="".join(chunks))


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIPPED_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > 700_000:
            continue
        if _looks_textual(path):
            yield path


def _looks_textual(path: Path) -> bool:
    name = path.name.lower()
    if name in IMPORTANT_NAMES or path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        sample = path.read_bytes()[:2048]
    except OSError:
        return False
    return b"\x00" not in sample


def _format_tree(root: Path, files: list[Path]) -> str:
    lines: list[str] = []
    for path in sorted(files, key=lambda item: _rel(root, item).lower())[:500]:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        lines.append(f"{_rel(root, path)} ({size} bytes)")
    extra = max(0, len(files) - len(lines))
    if extra:
        lines.append(f"... {extra} more files")
    return "\n".join(lines)


def _file_priority(root: Path, path: Path, question_text: str) -> tuple[int, int, int, int, str]:
    rel = _rel(root, path).lower()
    name = path.name.lower()
    tokens = {token for token in re.findall(r"[a-zA-Z0-9_]{3,}", question_text.lower())}
    token_hits = sum(1 for token in tokens if token in rel)
    important_name = int(name in IMPORTANT_NAMES or name.startswith("readme"))
    source_or_doc = int(path.suffix.lower() in {".py", ".md", ".txt", ".ipynb", ".toml", ".yaml", ".yml"})
    size_score = -min(path.stat().st_size, MAX_FILE_CHARS)
    return token_hits, important_name, source_or_doc, size_score, rel


def _read_text(path: Path, max_chars: int = MAX_FILE_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[could not read file: {exc}]"

    text = _clean_notebook_json(path, text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated: {len(text)} chars total]"


def _clean_notebook_json(path: Path, text: str) -> str:
    if path.suffix.lower() != ".ipynb":
        return text
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return text
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        if source.strip():
            cells.append(f"# Cell {index}\n{source}")
    return "\n\n".join(cells) if cells else text


def _answer_with_tools(input: Input, snapshot: RepoSnapshot) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": _system_prompt(input.code_execution)},
        {"role": "user", "content": _initial_prompt(input, snapshot)},
    ]

    parsed: dict[str, Any] = {}
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        content = _call_llm(input, messages)
        parsed = _parse_json(content)

        actions = parsed.get("actions") or []
        if round_index >= MAX_TOOL_ROUNDS or not actions:
            break

        tool_results = _run_actions(snapshot.root, actions, input.code_execution)
        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=True)})
        messages.append({"role": "user", "content": _tool_result_prompt(tool_results)})

    answers = parsed.get("answers")
    return answers if isinstance(answers, list) else []


def _system_prompt(code_execution: bool) -> str:
    execution_note = (
        "You may request code execution when it is needed."
        if code_execution
        else "Code execution is disabled for this submission; rely on files only."
    )
    return f"""
You are an autonomous hackathon agent answering repository/documentation questions.
Use the repository tree, file contents, and any tool results to answer accurately.
Questions may require source-code analysis, code execution, or recognizing that the
available data is insufficient. {execution_note}

Return JSON only. The JSON schema is:
{{
  "actions": [
    {{
      "tool": "read_file" | "search" | "run_python" | "run_command",
      "path": "relative/path for read_file",
      "query": "literal search query for search",
      "code": "python code for run_python",
      "args": ["command", "arguments"] for run_command,
      "reason": "short reason"
    }}
  ],
  "answers": [
    {{
      "question": 123,
      "answer": "direct, complete answer",
      "confidence": "low" | "medium" | "high",
      "evidence": ["files"] or ["files", "execution"] or ["execution"] or [],
      "not_known": false
    }}
  ]
}}

If more information is needed, return actions and provisional answers if useful.
If the answer cannot be found in the repository or tool results, set not_known true,
confidence low, evidence to [], and explain briefly that the available data does not
support the answer. Do not hallucinate.
Calibrate confidence: high only for direct evidence or successful execution, medium
for strong inference, low for partial or missing evidence.
""".strip()


def _initial_prompt(input: Input, snapshot: RepoSnapshot) -> str:
    questions = "\n".join(
        f"- id={question.id}: {question.question}" for question in input.template
    )
    return f"""
Template: {input.template_title}

Questions:
{questions}

Repository tree:
{snapshot.tree}

Selected repository contents:
{snapshot.context}

Answer every question id exactly once. Prefer concise but complete answers.
""".strip()


def _tool_result_prompt(results: list[dict[str, Any]]) -> str:
    return f"""
Tool results:
{json.dumps(results, ensure_ascii=True, indent=2)}

Use these results to either answer all questions or request only the next essential
actions. Return JSON only with the required schema.
""".strip()


def _call_llm(input: Input, messages: list[dict[str, str]]) -> str:
    client = _client(input)
    model = _env_first(
        "LLM_AS_A_SERVICE_MODEL",
        "LLM_MODEL",
        "OPENAI_MODEL",
        default=DEFAULT_MODEL,
    )

    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or "{}"
        except Exception as exc:
            last_error = exc
            print(f"[agent] LLM call failed: {exc}")
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def _client(input: Input) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("The openai package is required for LLM_as_a_service") from exc

    api_key = _env_first(
        "LLM_AS_A_SERVICE_API_TOKEN",
        "LLM_AS_A_SERVICE_API_KEY",
        "LLM_API_TOKEN",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "API_TOKEN",
        default=input.token_sparrow,
    )
    base_url = _env_first(
        "LLM_AS_A_SERVICE_API_URL",
        "LLM_AS_A_SERVICE_BASE_URL",
        "LLM_API_URL",
        "OPENAI_BASE_URL",
        "API_URL",
    )

    if not api_key:
        raise RuntimeError("Missing LLM service API token")
    if not base_url:
        raise RuntimeError(
            "Missing LLM service base URL. Set LLM_AS_A_SERVICE_API_URL or OPENAI_BASE_URL."
        )

    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _env_first(*names: str, default: str | None = None) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default or ""


def _parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"Could not parse LLM JSON response: {content[:500]}")


def _run_actions(root: Path, actions: list[Any], code_execution: bool) -> list[dict[str, Any]]:
    results = []
    for action in actions[:8]:
        if not isinstance(action, dict):
            continue
        tool = str(action.get("tool", "")).strip()
        try:
            if tool == "read_file":
                result = _tool_read_file(root, str(action.get("path", "")))
            elif tool == "search":
                result = _tool_search(root, str(action.get("query", "")))
            elif tool == "run_python" and code_execution:
                result = _tool_run_python(root, str(action.get("code", "")))
            elif tool == "run_command" and code_execution:
                result = _tool_run_command(root, action.get("args"))
            elif tool in {"run_python", "run_command"}:
                result = {"error": "code execution is disabled"}
            else:
                result = {"error": f"unknown tool: {tool}"}
        except Exception as exc:
            result = {"error": str(exc)}
        results.append(
            {
                "tool": tool,
                "reason": action.get("reason", ""),
                "result": result,
            }
        )
    return results


def _tool_read_file(root: Path, relative_path: str) -> dict[str, str]:
    path = _safe_path(root, relative_path)
    if not path.exists() or not path.is_file():
        return {"error": f"file not found: {relative_path}"}
    return {"path": _rel(root, path), "content": _read_text(path, MAX_TOOL_OUTPUT_CHARS)}


def _tool_search(root: Path, query: str) -> dict[str, Any]:
    query = query.strip()
    if not query:
        return {"error": "empty search query"}

    matches = []
    lowered = query.lower()
    for path in _iter_files(root):
        text = _read_text(path, MAX_TOOL_OUTPUT_CHARS)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if lowered in line.lower():
                matches.append(
                    {
                        "path": _rel(root, path),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                break
        if len(matches) >= 25:
            break
    return {"query": query, "matches": matches}


def _tool_run_python(root: Path, code: str) -> dict[str, str | int]:
    if not code.strip():
        return {"error": "empty python code"}
    return _run_subprocess(root, [sys.executable, "-c", code], timeout=40)


def _tool_run_command(root: Path, args: Any) -> dict[str, str | int]:
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return {"error": "args must be a list of strings"}
    if not args:
        return {"error": "empty command"}

    executable = Path(args[0]).name.lower()
    allowed = {
        "cat",
        "find",
        "grep",
        "ls",
        "node",
        "npm",
        "pytest",
        "python",
        "python3",
        "sed",
    }
    if executable not in allowed:
        return {"error": f"command not allowed: {args[0]}"}

    if executable in {"python", "python3"}:
        args = [sys.executable, *args[1:]]

    return _run_subprocess(root, args, timeout=80)


def _run_subprocess(root: Path, args: list[str], timeout: int) -> dict[str, str | int]:
    env = {**os.environ, "PYTHONPATH": str(root)}
    completed = subprocess.run(
        args,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout = _truncate(completed.stdout)
    stderr = _truncate(completed.stderr)
    return {"stdout": stdout, "stderr": stderr, "exit_code": completed.returncode}


def _safe_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def _coerce_answers(input: Input, raw_answers: list[dict[str, Any]]) -> List[AnswerItem]:
    by_question: dict[int, dict[str, Any]] = {}
    for answer in raw_answers:
        if not isinstance(answer, dict):
            continue
        try:
            question_id = int(answer.get("question"))
        except (TypeError, ValueError):
            continue
        by_question[question_id] = answer

    results = []
    for question in input.template:
        raw = by_question.get(question.id)
        if raw is None:
            results.append(_unknown_answer(question.id, "The available data did not support an answer."))
            continue

        confidence = _normalize_confidence(raw.get("confidence"))
        evidence = _normalize_evidence(raw.get("evidence"))
        not_known = bool(raw.get("not_known", False))
        answer_text = str(raw.get("answer") or "").strip()
        if not answer_text:
            answer_text = "The available data did not support an answer."
            not_known = True
            confidence = "low"
            evidence = []

        results.append(
            AnswerItem(
                question=question.id,
                answer=answer_text,
                confidence=confidence,
                evidence=evidence,
                not_known=not_known,
            )
        )
    return results


def _unknown_answer(question_id: int, reason: str) -> AnswerItem:
    return AnswerItem(
        question=question_id,
        answer=reason,
        confidence="low",
        evidence=[],
        not_known=True,
    )


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        if value >= 0.8:
            return "high"
        if value >= 0.45:
            return "medium"
        return "low"

    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    if text in {"certain", "strong"}:
        return "high"
    if text in {"unknown", "none", "weak"}:
        return "low"
    return "medium"


def _normalize_evidence(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    evidence = []
    for item in value:
        item_text = str(item).strip().lower()
        if item_text in {"files", "execution"} and item_text not in evidence:
            evidence.append(item_text)
    return evidence


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {len(text)} chars total]"


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
