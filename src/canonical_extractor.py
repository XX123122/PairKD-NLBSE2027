"""Vendored exact function-level excerpt from the project canonical extractor.

Source: scripts/analyze_code_comment_evidence.py
The source SHA-256 is recorded in bundle_manifest.json and README.md.
"""
from __future__ import annotations

import re


def normalize_comment_text(text: str) -> str:
    """Normalize extracted comment delimiters while preserving content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^//+\s?", "", stripped)
        stripped = re.sub(r"^/\*+\s?", "", stripped)
        stripped = re.sub(r"\s*\*/$", "", stripped)
        stripped = re.sub(r"^\*+\s?", "", stripped)
        stripped = re.sub(r"^#\s?", "", stripped)
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def extract_code_and_comments(text: str) -> tuple[str, str, dict[str, int]]:
    """Canonical lightweight C-like lexical extraction used in NLBSE training.

    Recognizes //, /* */ and logical-line # comments, while preserving quote
    and escape handling so comment-looking text in strings remains code.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    code_out: list[str] = []
    comment_out: list[str] = []
    n = len(text)
    i = 0
    state = "code"
    quote = ""
    escaped = False
    line_comment_count = 0
    block_comment_count = 0
    hash_comment_count = 0
    logical_line_only_ws = True

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if quote:
                code_out.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = ""
                if ch == "\n":
                    logical_line_only_ws = True
                elif not ch.isspace():
                    logical_line_only_ws = False
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                code_out.append(ch)
                logical_line_only_ws = False
                i += 1
                continue
            if ch == "/" and nxt == "/":
                line_comment_count += 1
                comment_out.append("//")
                code_out.append(" ")
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                block_comment_count += 1
                comment_out.append("/*")
                code_out.append(" ")
                state = "block_comment"
                i += 2
                continue
            if ch == "#" and logical_line_only_ws:
                rest = text[i + 1 : i + 32]
                if re.match(
                    r"\s*(include|define|if|ifdef|ifndef|elif|else|endif|pragma|"
                    r"error|warning|line|undef)\b",
                    rest,
                ):
                    code_out.append(ch)
                    logical_line_only_ws = False
                    i += 1
                    continue
                hash_comment_count += 1
                comment_out.append("#")
                code_out.append(" ")
                state = "hash_comment"
                i += 1
                continue
            code_out.append(ch)
            if ch == "\n":
                logical_line_only_ws = True
            elif not ch.isspace():
                logical_line_only_ws = False
            i += 1
            continue
        if state in ("line_comment", "hash_comment"):
            comment_out.append(ch)
            if ch == "\n":
                code_out.append("\n")
                state = "code"
                logical_line_only_ws = True
            i += 1
            continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                comment_out.append("*/")
                code_out.append(" ")
                state = "code"
                i += 2
                continue
            comment_out.append(ch)
            if ch == "\n":
                code_out.append("\n")
                logical_line_only_ws = True
            i += 1

    code_only = "".join(code_out)
    code_only = re.sub(r"[ \t]+", " ", code_only)
    code_only = re.sub(r" *\n *", "\n", code_only)
    code_only = re.sub(r"\n{4,}", "\n\n\n", code_only).strip()
    comment_only = normalize_comment_text("".join(comment_out))
    meta = {
        "line_comment_count": line_comment_count,
        "block_comment_count": block_comment_count,
        "hash_comment_count": hash_comment_count,
        "comment_count": line_comment_count + block_comment_count + hash_comment_count,
    }
    return code_only, comment_only, meta
