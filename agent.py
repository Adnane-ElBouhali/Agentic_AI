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
    from hackathon_starter_kit.tools.code.env_setup import create_venv
except ModuleNotFoundError as exc:
    if exc.name != "hackathon_starter_kit":
        raise
    from models import AnswerItem
    from models import Input
    from tools.gitlab.clone import clone
    from tools.code.env_setup import create_venv

try:
    from hackathon_starter_kit.tools.code.code_execution import execute_notebook
    from hackathon_starter_kit.tools.code.code_execution import execute_python_file
    from hackathon_starter_kit.tools.code.code_execution import execute_python_snippet
except Exception:
    execute_notebook = None
    execute_python_file = None
    execute_python_snippet = None


DEFAULT_MODEL = "mistral-medium-2508"
ROLE_MODEL_ENV = {
    "classifier": ("LLM_MODEL_CLASSIFIER", "AGENT_MODEL_CLASSIFIER"),
    "sufficiency": ("LLM_MODEL_SUFFICIENCY", "AGENT_MODEL_SUFFICIENCY"),
    "execution": ("LLM_MODEL_EXECUTION", "AGENT_MODEL_EXECUTION"),
    "answer": ("LLM_MODEL_ANSWER", "AGENT_MODEL_ANSWER"),
    "verifier": ("LLM_MODEL_VERIFIER", "AGENT_MODEL_VERIFIER"),
    "calibrator": ("LLM_MODEL_CALIBRATOR", "AGENT_MODEL_CALIBRATOR"),
    "default": ("LLM_MODEL", "OPENAI_MODEL"),
}
_VENV_CACHE: dict[str, str] = {}

MAX_CONTEXT_CHARS = 120_000
MAX_QUESTION_CONTEXT_CHARS = 80_000
MAX_EVIDENCE_CHARS = 130_000
MAX_AGENT_CONTEXT_CHARS = 90_000
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
    classification = _classify_question(input, snapshot, question)
    evidence = _collect_evidence(snapshot, question, classification)
    sufficiency = _evidence_sufficiency_judge(input, snapshot, question, classification, evidence)
    evidence = _append_evidence_section(
        evidence,
        "Evidence Sufficiency Judgement",
        json.dumps(sufficiency, ensure_ascii=True, indent=2),
    )

    if _sufficiency_requests_more_evidence(sufficiency):
        evidence = _append_evidence_section(
            evidence,
            "Additional Evidence",
            json.dumps(
                _collect_requested_evidence(snapshot, sufficiency),
                ensure_ascii=True,
                indent=2,
            ),
        )
        sufficiency = _evidence_sufficiency_judge(input, snapshot, question, classification, evidence)
        evidence = _append_evidence_section(
            evidence,
            "Evidence Sufficiency Judgement After Additional Retrieval",
            json.dumps(sufficiency, ensure_ascii=True, indent=2),
        )

    execution_results = []
    if _sufficiency_requests_execution(sufficiency) or _should_execute(question.question, classification):
        execution_results = _execution_agent(input, snapshot, question, classification, evidence)
    if execution_results:
        evidence = _append_evidence_section(
            evidence,
            "Execution Results",
            json.dumps(execution_results, ensure_ascii=True, indent=2),
        )
        sufficiency = _evidence_sufficiency_judge(input, snapshot, question, classification, evidence)
        evidence = _append_evidence_section(
            evidence,
            "Evidence Sufficiency Judgement After Execution",
            json.dumps(sufficiency, ensure_ascii=True, indent=2),
        )

    if _sufficiency_marks_not_known(sufficiency):
        answer = _fallback_answer(question.id)
        answer["answer"] = str(
            sufficiency.get(
                "not_known_reason",
                "The available data did not support an answer.",
            )
        )
        answer["source_paths"] = []
        return [answer]

    draft = _draft_answer(input, snapshot, question, classification, evidence)
    verified = _adversarial_verify(input, snapshot, question, classification, evidence, draft)
    answer = _confidence_calibrator(input, snapshot, question, classification, evidence, verified)
    if _answer_matches_question(answer, question.id):
        return [answer]
    return []


def _classify_question(input: Input, snapshot: RepoSnapshot, question: Any) -> dict[str, Any]:
    question_text = getattr(question, "question", "")
    fallback = _fallback_classification(snapshot, question_text, input.code_execution)
    messages = [
        {
            "role": "system",
            "content": """
You are the Classifier Agent in a repository QA challenge.
Classify the question before anyone answers it. Choose the safest approach:
source_analysis, execution_required, or probably_not_known.
Return JSON only.
Schema:
{
  "question_type": "source_analysis" | "execution_required" | "probably_not_known",
  "execution_needed": true,
  "answerability": "likely_answerable" | "uncertain" | "likely_not_known",
  "files_to_read": ["relative/path.py"],
  "search_queries": ["specific query"],
  "risk_notes": "short note about hallucination/confidence risk"
}
Prefer execution_required for runtime output, computed values, tests, shapes, printed
results, or behavior that static reading may mispredict. Prefer probably_not_known
when the requested fact sounds external to the repo.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Template: {input.template_title}
Question id={question.id}: {question_text}

Heuristic hint:
{_question_hints(question_text, input.code_execution)}

Repository metadata:
{snapshot.metadata}

Repository tree:
{snapshot.tree}

Source symbol index:
{snapshot.symbols}

Pre-search hits:
{_question_search_context(snapshot, question_text)}

Targeted snippets:
{_question_context(snapshot, question_text)}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, fallback, "classifier")
    return {**fallback, **parsed}


def _collect_evidence(snapshot: RepoSnapshot, question: Any, classification: dict[str, Any]) -> str:
    question_text = getattr(question, "question", "")
    sections = ""
    sections = _append_evidence_section(sections, "Question", f"id={question.id}: {question_text}")
    sections = _append_evidence_section(
        sections,
        "Classifier Result",
        json.dumps(classification, ensure_ascii=True, indent=2),
    )
    sections = _append_evidence_section(sections, "Approach Hint", _question_hints(question_text, True))
    sections = _append_evidence_section(sections, "Repository Metadata", snapshot.metadata)
    sections = _append_evidence_section(sections, "Repository Tree", snapshot.tree)
    sections = _append_evidence_section(sections, "Source Symbol Index", snapshot.symbols)
    sections = _append_evidence_section(
        sections,
        "Pre-search Hits",
        _question_search_context(snapshot, question_text),
    )

    queries = _evidence_queries(question_text, classification)
    search_results = []
    for query in queries[:6]:
        search_results.append(_tool_search(snapshot.root, query))
    sections = _append_evidence_section(
        sections,
        "Search Results",
        json.dumps(search_results, ensure_ascii=True, indent=2),
    )

    file_reads = []
    for rel_path in _evidence_files(snapshot, question_text, classification):
        result = _tool_read_file(snapshot.root, rel_path)
        file_reads.append(result)
    sections = _append_evidence_section(
        sections,
        "Selected File Contents",
        json.dumps(file_reads, ensure_ascii=True, indent=2),
    )

    sections = _append_evidence_section(
        sections,
        "Targeted Repository Snippets",
        _question_context(snapshot, question_text),
    )
    return sections


def _evidence_sufficiency_judge(
    input: Input,
    snapshot: RepoSnapshot,
    question: Any,
    classification: dict[str, Any],
    evidence: str,
) -> dict[str, Any]:
    fallback = {
        "enough_evidence": False,
        "should_answer": False,
        "should_execute": _should_execute(question.question, classification),
        "should_mark_not_known": False,
        "not_known_reason": "",
        "additional_files": [],
        "additional_search_queries": [],
        "confidence_ceiling": "medium",
        "rationale": "Fallback sufficiency judgement.",
    }
    messages = [
        {
            "role": "system",
            "content": """
You are the Evidence Sufficiency Judge. Decide whether the current evidence is
enough to answer one question, whether more retrieval/execution is needed, or
whether the question should be marked not_known.
Return JSON only:
{
  "enough_evidence": true,
  "should_answer": true,
  "should_execute": false,
  "should_mark_not_known": false,
  "not_known_reason": "",
  "additional_files": ["relative/path.py"],
  "additional_search_queries": ["query"],
  "confidence_ceiling": "low" | "medium" | "high",
  "rationale": "brief reason"
}
Mark not_known only when the evidence and repository inventory indicate the
requested fact is absent or external. Request execution for runtime outputs,
computed values, tests, or behavior that static evidence cannot settle.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Question id={question.id}: {question.question}

Classifier:
{json.dumps(classification, ensure_ascii=True, indent=2)}

Repository tree:
{snapshot.tree}

Evidence packet:
{evidence}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, fallback, "sufficiency")
    return {**fallback, **parsed}


def _collect_requested_evidence(snapshot: RepoSnapshot, sufficiency: dict[str, Any]) -> dict[str, Any]:
    collected: dict[str, Any] = {"files": [], "searches": []}
    files = sufficiency.get("additional_files", [])
    if isinstance(files, str):
        files = [files]
    if isinstance(files, list):
        for path in files[:8]:
            if str(path).strip():
                collected["files"].append(_tool_read_file(snapshot.root, str(path).strip()))

    queries = sufficiency.get("additional_search_queries", [])
    if isinstance(queries, str):
        queries = [queries]
    if isinstance(queries, list):
        for query in queries[:6]:
            if str(query).strip():
                collected["searches"].append(_tool_search(snapshot.root, str(query).strip()))
    return collected


def _sufficiency_requests_more_evidence(sufficiency: dict[str, Any]) -> bool:
    if bool(sufficiency.get("enough_evidence")) or bool(sufficiency.get("should_mark_not_known")):
        return False
    files = sufficiency.get("additional_files", [])
    queries = sufficiency.get("additional_search_queries", [])
    return bool(files) or bool(queries)


def _sufficiency_requests_execution(sufficiency: dict[str, Any]) -> bool:
    return bool(sufficiency.get("should_execute")) and not bool(sufficiency.get("should_mark_not_known"))


def _sufficiency_marks_not_known(sufficiency: dict[str, Any]) -> bool:
    return bool(sufficiency.get("should_mark_not_known")) and not bool(sufficiency.get("should_answer"))


def _execution_agent(
    input: Input,
    snapshot: RepoSnapshot,
    question: Any,
    classification: dict[str, Any],
    evidence: str,
) -> list[dict[str, Any]]:
    if not input.code_execution:
        return []
    if not _should_execute(question.question, classification):
        return []

    messages = [
        {
            "role": "system",
            "content": """
You are the Execution Agent. Decide the minimal safe execution needed to answer
one repository question. Return JSON only:
{
  "actions": [
    {
      "tool": "run_python" | "run_python_file" | "run_notebook" | "run_command",
      "code": "python snippet for run_python",
      "path": "relative/path.py or notebook.ipynb for file/notebook execution",
      "args": ["python", "script.py"] for run_command,
      "reason": "why this execution is necessary"
    }
  ]
}
Use at most two actions. Prefer small Python snippets that import/call the relevant
code or inspect data. Use run_notebook for notebook-output questions. Do not run
destructive commands or long services.
Return {"actions": []} if execution is not necessary after reading the evidence.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Question id={question.id}: {question.question}

Classifier:
{json.dumps(classification, ensure_ascii=True, indent=2)}

Evidence:
{evidence}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, {"actions": []}, "execution")
    actions = parsed.get("actions")
    if not isinstance(actions, list):
        return []
    executable_actions = [
        action
        for action in actions[:2]
        if isinstance(action, dict)
        and action.get("tool") in {"run_python", "run_python_file", "run_notebook", "run_command"}
    ]
    return _run_actions(snapshot.root, executable_actions, input.code_execution)


def _draft_answer(
    input: Input,
    snapshot: RepoSnapshot,
    question: Any,
    classification: dict[str, Any],
    evidence: str,
) -> dict[str, Any]:
    fallback = _fallback_answer(question.id)
    messages = [
        {
            "role": "system",
            "content": """
You are the Answer Agent. Write the best final answer from the evidence only.
Return JSON only:
{
  "answers": [
    {
      "question": 123,
      "answer": "concise complete answer",
      "confidence": "low" | "medium" | "high",
      "evidence": ["files"] or ["execution"] or ["files", "execution"] or [],
      "not_known": false,
      "source_paths": ["relative/path.py"]
    }
  ]
}
Do not invent missing facts. If evidence does not support the answer, return
not_known=true, confidence="low", evidence=[], and explain that the repository
does not contain enough information.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Question id={question.id}: {question.question}

Classifier:
{json.dumps(classification, ensure_ascii=True, indent=2)}

Evidence packet:
{evidence}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, {"answers": [fallback]}, "answer")
    answer = _extract_single_answer(parsed, question.id)
    return answer or fallback


def _adversarial_verify(
    input: Input,
    snapshot: RepoSnapshot,
    question: Any,
    classification: dict[str, Any],
    evidence: str,
    draft: dict[str, Any],
) -> dict[str, Any]:
    fallback = draft or _fallback_answer(question.id)
    messages = [
        {
            "role": "system",
            "content": """
You are the Adversarial Verifier Agent. Your task is to try to prove the draft
answer wrong or unsupported using only the evidence packet. Be strict.
Return one answer object as JSON only:
{
  "question": 123,
  "answer": "verified or corrected answer",
  "confidence": "low" | "medium" | "high",
  "evidence": ["files"] or ["execution"] or ["files", "execution"] or [],
  "not_known": false,
  "source_paths": ["relative/path.py"],
  "verification_notes": "brief note"
}
If the draft contains any fact not supported by evidence, rewrite it or mark
not_known=true. Do not add new facts. Prefer not_known over a plausible guess.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Question id={question.id}: {question.question}

Classifier:
{json.dumps(classification, ensure_ascii=True, indent=2)}

Draft answer:
{json.dumps(draft, ensure_ascii=True, indent=2)}

Evidence packet:
{evidence}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, fallback, "verifier")
    answer = _extract_single_answer(parsed, question.id)
    if answer is None and _answer_matches_question(parsed, question.id):
        answer = parsed
    return answer or fallback


def _confidence_calibrator(
    input: Input,
    snapshot: RepoSnapshot,
    question: Any,
    classification: dict[str, Any],
    evidence: str,
    verified: dict[str, Any],
) -> dict[str, Any]:
    fallback = _deterministic_answer_calibration(verified or _fallback_answer(question.id))
    messages = [
        {
            "role": "system",
            "content": """
You are the final Confidence Calibrator Agent. You may only adjust confidence,
evidence, not_known, and wording to match the evidence. Return JSON only:
{
  "question": 123,
  "answer": "final answer",
  "confidence": "low" | "medium" | "high",
  "evidence": ["files"] or ["execution"] or ["files", "execution"] or [],
  "not_known": false,
  "source_paths": ["relative/path.py"],
  "calibration_notes": "brief reason"
}
Calibration rules:
- not_known=true always means confidence low and evidence [].
- high requires exact source_paths or successful execution output.
- runtime/computed answers should be high only when execution output is present.
- static facts directly shown in files can be high.
- indirect but plausible source inference is medium.
- weak or ambiguous support is low.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Question id={question.id}: {question.question}

Classifier:
{json.dumps(classification, ensure_ascii=True, indent=2)}

Verified answer:
{json.dumps(verified, ensure_ascii=True, indent=2)}

Evidence packet:
{evidence}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, fallback, "calibrator")
    answer = _extract_single_answer(parsed, question.id)
    if answer is None and _answer_matches_question(parsed, question.id):
        answer = parsed
    return _deterministic_answer_calibration(answer or fallback)


def _deterministic_answer_calibration(answer: dict[str, Any]) -> dict[str, Any]:
    calibrated = dict(answer)
    confidence = _normalize_confidence(calibrated.get("confidence"))
    evidence = _normalize_evidence(calibrated.get("evidence"))
    not_known = bool(calibrated.get("not_known", False))
    answer_text = str(calibrated.get("answer") or "")
    if _looks_unknown_answer(answer_text):
        not_known = True
    source_paths = _normalize_source_paths(calibrated.get("source_paths"))
    confidence, evidence = _calibrate(confidence, evidence, not_known, source_paths)
    calibrated["confidence"] = confidence
    calibrated["evidence"] = evidence
    calibrated["not_known"] = not_known
    calibrated["source_paths"] = source_paths
    if not str(calibrated.get("answer") or "").strip():
        calibrated["answer"] = "The available data did not support an answer."
        calibrated["not_known"] = True
        calibrated["confidence"] = "low"
        calibrated["evidence"] = []
    return calibrated


def _critic_review(
    input: Input,
    snapshot: RepoSnapshot,
    question: Any,
    classification: dict[str, Any],
    evidence: str,
    draft: dict[str, Any],
) -> dict[str, Any]:
    fallback = draft or _fallback_answer(question.id)
    messages = [
        {
            "role": "system",
            "content": """
You are the Critic and Confidence Calibrator Agent for a judged hackathon.
Your job is to prevent wrong, unsupported, overconfident, or underconfident answers.
Return one corrected answer object as JSON only:
{
  "question": 123,
  "answer": "corrected answer",
  "confidence": "low" | "medium" | "high",
  "evidence": ["files"] or ["execution"] or ["files", "execution"] or [],
  "not_known": false,
  "source_paths": ["relative/path.py"],
  "critic_notes": "brief validation note"
}
Rules:
- If the answer is not directly supported by evidence, set not_known=true.
- High confidence requires direct source_paths or successful execution output.
- Execution output plus matching source evidence can be high confidence.
- Strong source evidence without execution is high only for static facts.
- Partial inference is medium. Missing/ambiguous evidence is low.
- Do not penalize correct answers with low confidence; calibrate honestly.
""".strip(),
        },
        {
            "role": "user",
            "content": _limit_text(
                f"""
Question id={question.id}: {question.question}

Classifier:
{json.dumps(classification, ensure_ascii=True, indent=2)}

Draft answer:
{json.dumps(draft, ensure_ascii=True, indent=2)}

Evidence packet:
{evidence}
""".strip(),
                MAX_AGENT_CONTEXT_CHARS,
            ),
        },
    ]
    parsed = _safe_call_and_parse_json(input, messages, fallback)
    answer = _extract_single_answer(parsed, question.id)
    if answer is None and _answer_matches_question(parsed, question.id):
        answer = parsed
    return answer or fallback


def _safe_call_and_parse_json(
    input: Input,
    messages: list[dict[str, str]],
    fallback: dict[str, Any],
    agent_role: str = "default",
) -> dict[str, Any]:
    try:
        parsed = _call_and_parse_json(input, messages, agent_role)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception as exc:
        print(f"[agent] JSON agent failed: {exc}")
        return fallback


def _extract_single_answer(parsed: dict[str, Any], question_id: int) -> dict[str, Any] | None:
    if _answer_matches_question(parsed, question_id):
        return parsed
    answer = parsed.get("answer")
    if isinstance(answer, dict) and _answer_matches_question(answer, question_id):
        return answer
    answers = parsed.get("answers")
    if isinstance(answers, list):
        for item in answers:
            if _answer_matches_question(item, question_id):
                return item
    return None


def _fallback_answer(question_id: int) -> dict[str, Any]:
    return {
        "question": question_id,
        "answer": "The available data did not support an answer.",
        "confidence": "low",
        "evidence": [],
        "not_known": True,
        "source_paths": [],
    }


def _fallback_classification(
    snapshot: RepoSnapshot,
    question_text: str,
    code_execution: bool,
) -> dict[str, Any]:
    question_type = "source_analysis"
    execution_needed = False
    hint = _question_hints(question_text, code_execution)
    if hint.startswith("Likely execution"):
        question_type = "execution_required"
        execution_needed = code_execution
    elif "unanswerable" in hint:
        question_type = "probably_not_known"

    return {
        "question_type": question_type,
        "execution_needed": execution_needed,
        "answerability": "uncertain",
        "files_to_read": _ranked_question_paths(snapshot, question_text, 5),
        "search_queries": _default_queries(question_text),
        "risk_notes": hint,
    }


def _should_execute(question_text: str, classification: dict[str, Any]) -> bool:
    if bool(classification.get("execution_needed")):
        return True
    if str(classification.get("question_type", "")).lower() == "execution_required":
        return True
    return _question_hints(question_text, True).startswith("Likely execution")


def _evidence_queries(question_text: str, classification: dict[str, Any]) -> list[str]:
    queries = []
    raw_queries = classification.get("search_queries", [])
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if isinstance(raw_queries, list):
        queries.extend(str(query).strip() for query in raw_queries if str(query).strip())
    queries.extend(_default_queries(question_text))

    deduped = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped


def _default_queries(question_text: str) -> list[str]:
    tokens = sorted(_tokenize(question_text))
    queries = []
    if question_text.strip():
        queries.append(question_text.strip())
    if tokens:
        queries.append(" ".join(tokens[:8]))
    for token in tokens[:5]:
        queries.append(token)
    return queries


def _evidence_files(
    snapshot: RepoSnapshot,
    question_text: str,
    classification: dict[str, Any],
) -> list[str]:
    paths = []
    raw_paths = classification.get("files_to_read", [])
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    if isinstance(raw_paths, list):
        paths.extend(str(path).strip() for path in raw_paths if str(path).strip())
    paths.extend(_ranked_question_paths(snapshot, question_text, 8))

    deduped = []
    for path in paths:
        if path and path not in deduped:
            deduped.append(path)
    return deduped[:12]


def _ranked_question_paths(snapshot: RepoSnapshot, question_text: str, limit: int) -> list[str]:
    ranked = sorted(
        snapshot.files,
        key=lambda path: _question_file_priority(snapshot.root, path, question_text),
        reverse=True,
    )
    return [_rel(snapshot.root, path) for path in ranked[:limit]]


def _append_evidence_section(existing: str, title: str, body: str) -> str:
    section = f"\n\n## {title}\n{body.strip() if body else '[empty]'}"
    if len(existing) + len(section) <= MAX_EVIDENCE_CHARS:
        return existing + section
    remaining = MAX_EVIDENCE_CHARS - len(existing)
    if remaining <= 80:
        return existing
    return existing + section[:remaining] + "\n... [evidence truncated]"


def _limit_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {len(text)} chars total]"


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


def _call_llm(input: Input, messages: list[dict[str, str]], agent_role: str = "default") -> str:
    client = _client(input)
    model = _model_for_role(agent_role)

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


def _call_and_parse_json(
    input: Input,
    messages: list[dict[str, str]],
    agent_role: str = "default",
) -> dict[str, Any]:
    content = _call_llm(input, messages, agent_role)
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
        repaired = _call_llm(input, repair_messages, agent_role)
        return _parse_json(repaired)


def _model_for_role(agent_role: str) -> str:
    role = agent_role if agent_role in ROLE_MODEL_ENV else "default"
    names = ROLE_MODEL_ENV.get(role, ()) + (
        "LLM_AS_A_SERVICE_MODEL",
        "LLM_MODEL",
        "OPENAI_MODEL",
    )
    return _env_first(*names, default=DEFAULT_MODEL)


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
            elif tool == "run_python_file" and code_execution:
                result = _tool_run_python_file(root, str(action.get("path", "")))
            elif tool == "run_notebook" and code_execution:
                result = _tool_run_notebook(root, str(action.get("path", "")))
            elif tool == "run_command" and code_execution:
                result = _tool_run_command(root, action.get("args"))
            elif tool in {"run_python", "run_python_file", "run_notebook", "run_command"}:
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
    venv_python = _python_for_repo(root)
    if execute_python_snippet is not None:
        output = execute_python_snippet(code, venv_python_path=venv_python)
        return {"stdout": _truncate(output), "stderr": "", "exit_code": 0}
    return _run_subprocess(root, [venv_python, "-c", code], timeout=40)


def _tool_run_python_file(root: Path, relative_path: str) -> dict[str, str | int]:
    path = _resolve_repo_file(root, relative_path)
    if not path.exists() or not path.is_file():
        return {"error": f"file not found: {relative_path}"}
    if path.suffix.lower() != ".py":
        return {"error": f"not a Python file: {relative_path}"}
    venv_python = _python_for_repo(root)
    if execute_python_file is not None:
        output = execute_python_file(str(path), venv_python_path=venv_python)
        return {"stdout": _truncate(output), "stderr": "", "exit_code": 0}
    return _run_subprocess(root, [venv_python, str(path)], timeout=80)


def _tool_run_notebook(root: Path, relative_path: str) -> dict[str, str | int]:
    path = _resolve_repo_file(root, relative_path)
    if not path.exists() or not path.is_file():
        return {"error": f"file not found: {relative_path}"}
    if path.suffix.lower() != ".ipynb":
        return {"error": f"not a notebook file: {relative_path}"}
    if execute_notebook is None:
        return {"error": "notebook execution helper is unavailable"}
    output = execute_notebook(str(path), venv_python_path=_python_for_repo(root))
    return {"stdout": _truncate(output), "stderr": "", "exit_code": 0}


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
    cached = _VENV_CACHE.get(str(root))
    if cached and Path(cached).exists():
        return cached

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
            _VENV_CACHE[str(root)] = str(candidate)
            return str(candidate)

    should_build_env = os.getenv("AGENT_BUILD_REPO_ENV", "0").lower() in {"1", "true", "yes"}
    if should_build_env and ((root / "requirements.txt").exists() or (root / "pyproject.toml").exists()):
        try:
            created = create_venv(str(root))
        except Exception as exc:
            print(f"[agent] Could not create repo venv: {exc}")
        else:
            if created and not str(created).startswith("[error]") and Path(created).exists():
                _VENV_CACHE[str(root)] = str(created)
                return str(created)
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
