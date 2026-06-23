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
MAX_QUESTION_CONTEXT_CHARS = 80_000
MAX_FILE_CHARS = 12_000
MAX_TOOL_OUTPUT_CHARS = 18_000
MAX_TOOL_ROUNDS = 3
MAX_SEARCH_FILE_CHARS = 220_000
MAX_SEARCH_MATCHES = 40
MAX_SNIPPETS_PER_FILE = 4
SEARCH_WINDOW_LINES = 2

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")
STOP_WORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "answer",
    "are",
    "can",
    "code",
    "does",
    "file",
    "find",
    "for",
    "from",
    "has",
    "how",
    "into",
    "its",
    "not",
    "question",
    "repo",
    "repository",
    "return",
    "should",
    "that",
    "the",
    "this",
    "use",
    "what",
    "when",
    "where",
    "which",
    "with",
}

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
    files: list[Path]
    all_files: list[Path]
    symbols: str
    metadata: str


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
    all_files = list(_iter_all_files(root))
    files = [path for path in all_files if _looks_textual(path) and path.stat().st_size <= 700_000]
    tree = _format_tree(root, all_files)
    symbols = _build_symbol_index(root, files)
    metadata = _repo_metadata(root)
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

    return RepoSnapshot(
        root=root,
        tree=tree,
        context="".join(chunks),
        files=files,
        all_files=all_files,
        symbols=symbols,
        metadata=metadata,
    )


def _iter_all_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in SKIPPED_DIRS for part in rel_parts):
            continue
        if path.is_file():
            yield path


def _iter_files(root: Path) -> Iterable[Path]:
    for path in _iter_all_files(root):
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
        kind = "text" if _looks_textual(path) else "binary"
        lines.append(f"{_rel(root, path)} ({size} bytes, {kind})")
    extra = max(0, len(files) - len(lines))
    if extra:
        lines.append(f"... {extra} more files")
    return "\n".join(lines)


def _repo_metadata(root: Path) -> str:
    fields = []
    for args, label in (
        (["git", "rev-parse", "--short", "HEAD"], "commit"),
        (["git", "branch", "--show-current"], "branch"),
        (["git", "remote", "-v"], "remotes"),
    ):
        try:
            result = subprocess.run(
                args,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            fields.append(f"{label}: {value[:1200]}")
    return "\n".join(fields) if fields else "No git metadata available."


def _build_symbol_index(root: Path, files: list[Path]) -> str:
    entries: list[str] = []
    for path in sorted(files, key=lambda item: _rel(root, item).lower()):
        if len(entries) >= 260:
            break
        suffix = path.suffix.lower()
        if suffix == ".py":
            entries.extend(_python_symbols(root, path))
        elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            entries.extend(_javascript_symbols(root, path))
    if not entries:
        return "No source symbols detected."
    return "\n".join(entries[:260])


def _python_symbols(root: Path, path: Path) -> list[str]:
    try:
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    entries = []
    rel = _rel(root, path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            entries.append(f"{rel}:{node.lineno} {kind} {node.name}")
    return entries


def _javascript_symbols(root: Path, path: Path) -> list[str]:
    text = _read_text(path, MAX_SEARCH_FILE_CHARS)
    entries = []
    rel = _rel(root, path)
    patterns = [
        (r"\b(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
        (r"\b(?:export\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
        (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", "function"),
    ]
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        for pattern, kind in patterns:
            for match in re.finditer(pattern, line):
                entries.append(f"{rel}:{index} {kind} {match.group(1)}")
    return entries


def _file_priority(root: Path, path: Path, question_text: str) -> tuple[int, int, int, int, str]:
    rel = _rel(root, path).lower()
    name = path.name.lower()
    tokens = _tokenize(question_text)
    token_hits = sum(1 for token in tokens if token in rel)
    important_name = int(name in IMPORTANT_NAMES or name.startswith("readme"))
    source_or_doc = int(path.suffix.lower() in {".py", ".md", ".txt", ".ipynb", ".toml", ".yaml", ".yml"})
    size_score = -min(path.stat().st_size, MAX_FILE_CHARS)
    return token_hits, important_name, source_or_doc, size_score, rel


def _question_file_priority(root: Path, path: Path, question_text: str) -> tuple[int, int, int, int, str]:
    rel = _rel(root, path).lower()
    name = path.name.lower()
    tokens = _tokenize(question_text)
    path_hits = sum(4 for token in tokens if token in rel)
    important_name = int(name in IMPORTANT_NAMES or name.startswith("readme"))
    source_or_doc = int(path.suffix.lower() in {".py", ".md", ".txt", ".ipynb", ".toml", ".yaml", ".yml"})
    text = _read_text(path, MAX_SEARCH_FILE_CHARS).lower()
    content_hits = sum(min(text.count(token), 8) for token in tokens)
    size_score = -min(path.stat().st_size, MAX_FILE_CHARS)
    return path_hits + content_hits, important_name, source_or_doc, size_score, rel


def _question_context(snapshot: RepoSnapshot, question_text: str) -> str:
    ranked = sorted(
        snapshot.files,
        key=lambda path: _question_file_priority(snapshot.root, path, question_text),
        reverse=True,
    )
    tokens = _tokenize(question_text)

    chunks: list[str] = []
    total = 0
    for index, path in enumerate(ranked):
        if total >= MAX_QUESTION_CONTEXT_CHARS:
            break

        rel = _rel(snapshot.root, path)
        snippets = _extract_relevant_snippets(path, tokens)
        if snippets:
            content = "\n\n".join(snippets)
            chunk = f"\n--- RELEVANT SNIPPETS: {rel} ---\n{content}\n"
        elif index < 8:
            content = _read_text(path, min(MAX_FILE_CHARS, MAX_QUESTION_CONTEXT_CHARS - total))
            if not content:
                continue
            chunk = f"\n--- FILE: {rel} ---\n{content}\n"
        else:
            continue

        remaining = MAX_QUESTION_CONTEXT_CHARS - total
        if len(chunk) > remaining:
            chunk = chunk[:remaining] + "\n... [question context truncated]"
        chunks.append(chunk)
        total += len(chunk)

    if chunks:
        return "".join(chunks)
    return snapshot.context


def _extract_relevant_snippets(path: Path, tokens: set[str]) -> list[str]:
    if not tokens:
        return []

    text = _read_text(path, MAX_SEARCH_FILE_CHARS)
    lines = text.splitlines()
    scored: list[tuple[int, int]] = []
    for line_number, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for token in tokens if token in line_lower)
        if score:
            scored.append((score, line_number))

    snippets: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    for _, line_number in sorted(scored, reverse=True):
        if len(snippets) >= MAX_SNIPPETS_PER_FILE:
            break
        start = max(0, line_number - SEARCH_WINDOW_LINES)
        end = min(len(lines), line_number + SEARCH_WINDOW_LINES + 1)
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
            continue
        used_ranges.append((start, end))
        body = "\n".join(
            f"{idx + 1}: {lines[idx][:600]}" for idx in range(start, end)
        )
        snippets.append(body)

    return snippets


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
    answers: list[dict[str, Any]] = []
    for question in input.template:
        answers.extend(_answer_one_question(input, snapshot, question))
    return answers


def _answer_one_question(input: Input, snapshot: RepoSnapshot, question: Any) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": _system_prompt(input.code_execution)},
        {"role": "user", "content": _question_prompt(input, snapshot, question)},
    ]

    parsed: dict[str, Any] = {}
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        parsed = _call_and_parse_json(input, messages)

        actions = parsed.get("actions") or []
        if round_index >= MAX_TOOL_ROUNDS or not actions:
            break

        tool_results = _run_actions(snapshot.root, actions, input.code_execution)
        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=True)})
        messages.append({"role": "user", "content": _tool_result_prompt(question, tool_results)})

    answers = parsed.get("answers")
    if not isinstance(answers, list):
        return []
    return [answer for answer in answers if _answer_matches_question(answer, question.id)]


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

Your score depends on both answer correctness and confidence calibration. Think like
an evidence-gathering agent:
- Use source_paths for files/snippets that directly support the answer.
- Request search/read_file when the targeted context is not enough.
- Request execution when the question asks for runtime output, computed values, or
  behavior that cannot be known confidently from static reading.
- If no evidence supports the requested fact, do not infer or invent it.

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
      "not_known": false,
      "source_paths": ["relative/path.py"]
    }}
  ]
}}

You will normally answer one question at a time. If more information is needed,
return actions and provisional answers if useful.
If the answer cannot be found in the repository or tool results, set not_known true,
confidence low, evidence to [], and explain briefly that the available data does not
support the answer. Do not hallucinate.
Calibrate confidence: high only for direct evidence or successful execution, medium
for strong inference, low for partial or missing evidence.
Do not use high confidence unless source_paths is non-empty or evidence includes
"execution". If you used an execution result, include "execution" in evidence.
Use source_paths for your own grounding; the final API may ignore it.
""".strip()


def _question_prompt(input: Input, snapshot: RepoSnapshot, question: Any) -> str:
    question_text = getattr(question, "question", "")
    context = _question_context(snapshot, question_text)
    hints = _question_hints(question_text, input.code_execution)
    search_hits = _question_search_context(snapshot, question_text)
    return f"""
Template: {input.template_title}

Question:
- id={question.id}: {question_text}

Likely approach:
{hints}

Challenge rubric:
- Some questions are answered by source-code/file analysis.
- Some questions require executing code to observe the answer.
- Some questions cannot be answered from the repository or available resources.
- For unanswerable questions: set not_known=true, confidence="low", evidence=[].
- Avoid both overconfidence and underconfidence; the judge penalizes miscalibration.

Repository metadata:
{snapshot.metadata}

Repository tree:
{snapshot.tree}

Source symbol index:
{snapshot.symbols}

Pre-search hits for this question:
{search_hits}

Targeted repository contents:
{context}

Answer question id {question.id} exactly once. Prefer concise but complete answers.
Use high confidence only when the answer is directly supported by file paths/snippets
or successful execution. If the question asks for a runtime value/output and execution
is enabled, request a run_python or run_command action before finalizing.
""".strip()


def _question_hints(question_text: str, code_execution: bool) -> str:
    lowered = question_text.lower()
    execution_markers = (
        "execute",
        "run",
        "output",
        "prints",
        "printed",
        "result",
        "returns",
        "value",
        "evaluate",
        "calculate",
        "compute",
        "shape",
    )
    missing_markers = (
        "author",
        "created",
        "email",
        "phone",
        "owner",
        "password",
        "secret",
        "token",
        "outside",
    )
    if any(marker in lowered for marker in execution_markers):
        if code_execution:
            return "Likely execution question. Use file analysis to locate code, then execute a minimal snippet or command before answering."
        return "Likely execution question, but execution is disabled. Answer from files only and lower confidence if runtime behavior is uncertain."
    if any(marker in lowered for marker in missing_markers):
        return "May be unanswerable if the requested fact is not in files or tool results. Prefer not_known=true over guessing."
    return "Likely source/documentation analysis question. Cite source_paths and answer from repository evidence."


def _question_search_context(snapshot: RepoSnapshot, question_text: str) -> str:
    tokens = sorted(_tokenize(question_text))
    if not tokens:
        return "No useful search terms."
    query = " ".join(tokens[:8])
    result = _tool_search(snapshot.root, query)
    matches = result.get("matches", []) if isinstance(result, dict) else []
    if not matches:
        return "No direct pre-search hits."
    lines = []
    for match in matches[:10]:
        path = match.get("path", "")
        line = match.get("line", "")
        snippet = str(match.get("snippet", "")).replace("\n", "\n  ")
        lines.append(f"- {path}:{line}\n  {snippet}")
    return "\n".join(lines)


def _tool_result_prompt(question: Any, results: list[dict[str, Any]]) -> str:
    return f"""
Question:
- id={question.id}: {question.question}

Tool results:
{json.dumps(results, ensure_ascii=True, indent=2)}

Use these results to either answer this question or request only the next essential
actions. Return JSON only with the required schema. The answers array must contain
only question id {question.id}.
""".strip()


def _answer_matches_question(answer: Any, question_id: int) -> bool:
    if not isinstance(answer, dict):
        return False
    try:
        return int(answer.get("question")) == int(question_id)
    except (TypeError, ValueError):
        return False


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


def _call_and_parse_json(input: Input, messages: list[dict[str, str]]) -> dict[str, Any]:
    content = _call_llm(input, messages)
    try:
        return _parse_json(content)
    except RuntimeError:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON for the required schema. "
                    "Return only corrected JSON. Do not add markdown or commentary."
                ),
            },
        ]
        repaired = _call_llm(input, repair_messages)
        return _parse_json(repaired)


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


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOP_WORDS and not token.isdigit()
    }


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
    path = _resolve_repo_file(root, relative_path)
    if not path.exists() or not path.is_file():
        return {"error": f"file not found: {relative_path}"}
    return {"path": _rel(root, path), "content": _read_text(path, MAX_TOOL_OUTPUT_CHARS)}


def _resolve_repo_file(root: Path, relative_path: str) -> Path:
    try:
        path = _safe_path(root, relative_path)
    except ValueError:
        raise
    if path.exists():
        return path

    cleaned = relative_path.strip().replace("\\", "/").strip("/")
    if not cleaned:
        return path
    candidates = []
    for candidate in _iter_files(root):
        rel = _rel(root, candidate)
        if rel == cleaned or rel.endswith(f"/{cleaned}") or candidate.name == cleaned:
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    return path


def _tool_search(root: Path, query: str) -> dict[str, Any]:
    query = query.strip()
    if not query:
        return {"error": "empty search query"}

    matches: list[dict[str, Any]] = []
    lowered = query.lower()
    tokens = _tokenize(query)
    for path in _iter_files(root):
        text = _read_text(path, MAX_SEARCH_FILE_CHARS)
        lines = text.splitlines()
        file_matches: list[dict[str, Any]] = []
        used_ranges: list[tuple[int, int]] = []

        for index, line in enumerate(lines):
            line_lower = line.lower()
            score = 0
            if lowered and lowered in line_lower:
                score += 12
            score += sum(1 for token in tokens if token in line_lower)
            if score == 0:
                continue

            start = max(0, index - SEARCH_WINDOW_LINES)
            end = min(len(lines), index + SEARCH_WINDOW_LINES + 1)
            if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
                continue
            used_ranges.append((start, end))
            snippet = "\n".join(
                f"{line_index + 1}: {lines[line_index][:600]}"
                for line_index in range(start, end)
            )
            file_matches.append(
                {
                    "score": score,
                    "path": _rel(root, path),
                    "line": index + 1,
                    "snippet": snippet,
                }
            )
            if len(file_matches) >= MAX_SNIPPETS_PER_FILE:
                break

        matches.extend(file_matches)

    matches.sort(key=lambda match: (match["score"], match["path"], -match["line"]), reverse=True)
    for match in matches:
        match.pop("score", None)
    return {"query": query, "matches": matches[:MAX_SEARCH_MATCHES]}


def _tool_run_python(root: Path, code: str) -> dict[str, str | int]:
    if not code.strip():
        return {"error": "empty python code"}
    return _run_subprocess(root, [_python_for_repo(root), "-c", code], timeout=40)


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
        args = [_python_for_repo(root), *args[1:]]
    elif executable == "pytest":
        args = [_python_for_repo(root), "-m", "pytest", *args[1:]]

    return _run_subprocess(root, args, timeout=80)


def _python_for_repo(root: Path) -> str:
    candidates = [
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
        root / "env" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        root / "env" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


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
        if _looks_unknown_answer(answer_text):
            not_known = True

        source_paths = _normalize_source_paths(raw.get("source_paths"))
        confidence, evidence = _calibrate(confidence, evidence, not_known, source_paths)

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


def _normalize_source_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _calibrate(
    confidence: str,
    evidence: list[str],
    not_known: bool,
    source_paths: list[str],
) -> tuple[str, list[str]]:
    if not_known:
        return "low", []
    if not evidence:
        return "low", []
    if "execution" in evidence and confidence == "low":
        return "medium", evidence
    if "files" in evidence and "execution" in evidence and confidence == "medium":
        return "high", evidence
    if confidence == "high" and "execution" not in evidence and not source_paths:
        return "medium", evidence
    return confidence, evidence


def _looks_unknown_answer(answer: str) -> bool:
    lowered = answer.lower()
    unknown_markers = (
        "available data did not support",
        "cannot be found",
        "can't be found",
        "not enough information",
        "not present in the repository",
        "unknown",
    )
    return any(marker in lowered for marker in unknown_markers)


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {len(text)} chars total]"


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
