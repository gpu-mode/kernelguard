#!/usr/bin/env python3
"""
KernelGuard — Rule-based GPU kernel hack detector.

Detects cheating/hacking patterns in GPU kernel competition submissions.

Usage:
    kernelguard --jsonl /path/to/pairs.jsonl
    kernelguard --parquet /path/to/submissions.parquet
    kernelguard --audit-rules
"""

import argparse
import ast
import copy
from dataclasses import dataclass, field
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import tomllib
import tokenize
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Optional


CONFIG_VERSION = 1
DEFAULT_PROFILE_NAME = "default"

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Pattern 1: Timer monkey-patching
RE_CUDA_EVENT_ASSIGN = re.compile(r'torch\.cuda\.Event\s*=\s*')
RE_CUDA_EVENT_SETATTR = re.compile(r'setattr\s*\(\s*torch\.cuda\s*,\s*[\'"]Event[\'"]\s*,')
RE_ELAPSED_TIME_METHOD_PATCH = re.compile(r'torch\.cuda\.Event\.elapsed_time\s*=')
RE_CUDA_SYNCHRONIZE_PATCH = re.compile(r'torch\.cuda\.synchronize\s*=')
RE_DUMMY_FAKE_EVENT = re.compile(r'class\s+_?(?:Dummy|Fake|Mock|Stub|Noop)\w*Event', re.IGNORECASE)
RE_ELAPSED_TIME_FAKE = re.compile(r'def\s+elapsed_time\s*\([^)]*\)\s*:')

# Pattern 2: Print/stdout injection
RE_EMIT_FAKE_BENCH = re.compile(r'def\s+_?emit_fake_benchmark')
RE_BUILTINS_PRINT_ASSIGN = re.compile(r'builtins\.print\s*=')
RE_PATCHED_PRINT = re.compile(r'_(?:patched|real|orig|original)_print')
RE_SYS_STDOUT_REPLACE = re.compile(r'sys\.stdout\s*=')
RE_SYS_STDERR_REPLACE = re.compile(r'sys\.stderr\s*=')
RE_RANKED_BENCHMARK_PRINT = re.compile(
    r'(?:print|sys\.stdout\.write)\s*\([^)]*(?:Ranked\s+Benchmark)',
    re.DOTALL,
)
RE_RANKED_BENCHMARK_STR = re.compile(r'["\'].*?Ranked\s+Benchmark.*?["\']', re.DOTALL)

# Pattern 3: Evaluator exploitation
RE_MAIN_TIME_KERNEL = re.compile(r'main\.time_kernel\s*=')
RE_MAIN_CLONE_DATA = re.compile(r'main\._clone_data\s*=')
RE_MAIN_CLEAR_L2 = re.compile(r'main\.clear_l2_cache\s*=')
RE_SYS_MODULES_MAIN = re.compile(
    r'sys\.modules(?:\s*\[\s*[\'"]__main__[\'"]\s*\]|\.get\s*\(\s*[\'"]__main__[\'"]\s*\))',
)
RE_MAIN_ALIAS_FROM_GET = re.compile(
    r'(\w+)\s*=\s*sys\.modules\.get\s*\(\s*[\'"]__main__[\'"]\s*\)',
)
RE_MAIN_ALIAS_FROM_INDEX = re.compile(
    r'(\w+)\s*=\s*sys\.modules\s*\[\s*[\'"]__main__[\'"]\s*\]',
)
RE_MAIN_ATTR_PATCH = re.compile(
    r'(\w+)\.(time_kernel|_clone_data|clear_l2_cache)\s*=',
)
RE_MAIN_SETATTR_PATCH = re.compile(
    r'setattr\s*\(\s*(\w+)\s*,\s*[\'"](time_kernel|_clone_data|clear_l2_cache)[\'"]\s*,',
)

# Pattern 4: torch._scaled_mm reference
RE_SCALED_MM_PYTHON = re.compile(r'torch\._scaled_mm\s*\(')
RE_SCALED_MM_ALIAS = re.compile(r'=\s*torch\._scaled_mm\s*(?!\()$', re.MULTILINE)  # _mm = torch._scaled_mm (not a call)
RE_SCALED_MM_ALIAS_ASSIGN = re.compile(r'^\s*(\w+)\s*=\s*torch\._scaled_mm\s*$', re.MULTILINE)
RE_SCALED_MM_CPP = re.compile(r'aten::_scaled_mm')
RE_SCALED_MM_SCHEMA = re.compile(r'findSchemaOrThrow\s*\(\s*["\']aten::_scaled_mm')
RE_SCALED_MM_ANY = re.compile(r'_scaled_mm')

# Pattern 5: torch.mm/bmm as main compute (Python-only check)
RE_TORCH_MM = re.compile(r'torch\.mm\s*\(')
RE_TORCH_BMM = re.compile(r'torch\.bmm\s*\(')
RE_TORCH_MATMUL = re.compile(r'torch\.matmul\s*\(')

# Pattern 6: Result caching by pointer
RE_WEAK_VALUE_DICT = re.compile(r'WeakValueDictionary')
RE_DECODED_CACHE = re.compile(r'_decoded_cache')
RE_PREPROCESS_CACHE = re.compile(r'_PREPROCESS_CACHE')
RE_RESULT_REUSE = re.compile(r'_result_reuse')
RE_SCALE_CACHE = re.compile(r'_scale_cache')
RE_OUTPUT_CACHE = re.compile(r'_OUTPUT_CACHE')
RE_RESULT_CACHE_GENERAL = re.compile(r'_(?:RESULT|GROUPED_RESULT|COMPUTE|GEMM)_CACHE')
RE_ID_DATA_CACHE = re.compile(r'id\s*\(\s*data\s*\)')
RE_DATA_PTR_CACHE_KEY = re.compile(
    r'(?:\w+\s*\[\s*int\s*\(\s*\w+\.data_ptr\s*\(\)\s*\)\s*\])|'  # cache[int(A.data_ptr())]
    r'(?:\.data_ptr\s*\(\)\s*[,\)].*?(?:cache|key|dict|hash))',
    re.IGNORECASE,
)
RE_VERSION_CACHE = re.compile(r'\._version\b')
RE_CACHE_GET_ASSIGN = re.compile(
    r'(\w+)\s*=\s*(\w*(?:cache|reuse)\w*)\.get\s*\(',
    re.IGNORECASE,
)
# "return cache[...]" but exclude compiled-kernel / module caches (legitimate)
RE_RETURN_CACHE_INDEX = re.compile(
    r'return\s+(?!_?(?:compiled|kernel|module|func|op)_?\w*cache)'
    r'\w*(?:cache|reuse)\w*\s*\[',
    re.IGNORECASE,
)
RE_CACHE_STORE_OUTPUT = re.compile(
    r'\w*(?:cache|reuse)\w*\s*\[[^\]]+\]\s*=\s*(?:result|output|out|c_ref)\b',
    re.IGNORECASE,
)
RE_CPP_CACHE_FAST_RETURN = re.compile(
    r'if\s*\([^)]*(?:cache\.last|cache\.prev|lastA|lastB|prevA|prevB)[^)]*\)\s*(?:\{[^{}]{0,200}?\breturn\b|return\b)',
    re.DOTALL,
)

# Pattern 7: CUDA graph replay
RE_CTYPES_LIBCUDART = re.compile(r'ctypes\.CDLL\s*\(\s*["\']libcudart')
RE_CUDA_GRAPH_CREATE = re.compile(r'cudaGraphCreate')
RE_CUDA_GRAPH_LAUNCH = re.compile(r'cudaGraphLaunch')
RE_CUDA_GRAPH_INSTANTIATE = re.compile(r'cudaGraphInstantiate')
RE_CUDA_STREAM_CAPTURE = re.compile(r'cudaStreamBeginCapture')

# Pattern 8: Silent fallback — limit scope to ~500 chars after except (roughly 10-15 lines)
RE_TORCH_MATMUL_FALLBACK = re.compile(
    r'except\s+(?:Exception|RuntimeError|BaseException|TypeError)[^:]*:'
    r'[^\n]{0,500}?'
    r'(?:torch\.matmul|torch\.mm\s*\(|torch\.bmm\s*\()',
    re.DOTALL,
)

# Pattern 9: Trivial/probe
RE_RETURN_EMPTY = re.compile(r'return\s+\[\s*\]')
RE_RETURN_NONE = re.compile(r'return\s+None\b')

# Pattern 11: torch.compile
RE_TORCH_COMPILE = re.compile(r'torch\.compile\s*\(')

# Pattern 12: Python CUDAGraph caching (not ctypes — uses torch.cuda.CUDAGraph API)
RE_CUDA_GRAPH_PYTHON = re.compile(r'torch\.cuda\.CUDAGraph\s*\(')
RE_CUDA_GRAPH_REPLAY = re.compile(r'\.replay\s*\(')
RE_CUDA_GRAPH_CACHE_KEY = re.compile(
    r'data_ptr\s*\(\).*?(?:graph|cache|dict)|'
    r'(?:graph|cache|dict).*?data_ptr\s*\(\)',
    re.IGNORECASE | re.DOTALL,
)
RE_GRAPH_CACHE_NAME = re.compile(r'\b(?:_graph_cache|graph_cache)\b', re.IGNORECASE)

# Pattern 13: Hardcoded benchmark shapes (KERNEL_CONFIGS gated by exact dimensions)
RE_KERNEL_CONFIGS_DICT = re.compile(r'KERNEL_CONFIGS\s*=\s*\{')
RE_SHAPE_TUPLE_KEY = re.compile(r'\(\s*\d{2,5}\s*,\s*\d{2,5}\s*(?:,\s*\d{1,5}\s*)?\)\s*:')
RE_SHAPE_IF_GATE = re.compile(
    r'if\s+.*?(?:==|in)\s*[\[(]?\s*\(?\s*\d{3,5}\s*,\s*\d{3,5}',
)

# Pattern 14: Unsynchronized multi-stream dispatch
RE_GET_STREAM_FROM_POOL = re.compile(r'getStreamFromPool|get_stream_from_pool|torch\.cuda\.Stream\s*\(')
RE_NO_SYNC_STREAM = re.compile(r'(?:stream|s)\d*\.synchronize\s*\(\)')
RE_STREAM_WAIT_EVENT = re.compile(r'\.wait_event\s*\(')
RE_STREAM_WAIT_STREAM = re.compile(r'\.wait_stream\s*\(')
RE_TORCH_CUDA_SYNCHRONIZE = re.compile(r'torch\.cuda\.synchronize\s*\(')
RE_CPP_STREAM_SYNC = re.compile(
    r'(?:cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cudaStreamWaitEvent)\s*\(',
)
RE_CPP_METHOD_SYNC = re.compile(r'\.(?:synchronize|wait_event|wait_stream)\s*\(')

# Pattern 15: cudaEventDisableTiming
RE_CUDA_EVENT_DISABLE_TIMING = re.compile(r'cudaEventDisableTiming|disable_timing\s*=\s*True')

# C++/CUDA block markers for stripping
RE_CPP_MARKERS = re.compile(r'#include|__global__|__device__|__host__|extern\s+"C"|asm\s+volatile')
RE_TRIPLE_QUOTED = re.compile(r'(?:r)?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', re.DOTALL)


# ---------------------------------------------------------------------------
# Utility: strip C++/CUDA inline source from Python code
# ---------------------------------------------------------------------------

_LARGE_LITERAL_THRESHOLD = 2000  # chars; blobs larger than this are stripped


def strip_cpp_cuda_blocks(code: str) -> str:
    """Remove C/CUDA source literals and large data blobs from Python code.

    Strips triple-quoted strings that either:
    - Contain C++/CUDA source markers (__global__, #include, etc.), OR
    - Are larger than _LARGE_LITERAL_THRESHOLD characters (e.g. base64 blobs,
      embedded source code, lookup tables).  These blobs are never hacking
      indicators and their long all-word-char content causes catastrophic
      backtracking in patterns like (\\w+)\\s*=\\s*literal.
    """
    def replacer(match):
        content = match.group(1)
        if len(content) > _LARGE_LITERAL_THRESHOLD or RE_CPP_MARKERS.search(content):
            return '""""""'  # empty triple-quoted string
        return match.group(0)
    return RE_TRIPLE_QUOTED.sub(replacer, code)


def strip_python_comments(code: str) -> str:
    """Remove Python comments without touching string literals."""
    try:
        tokens = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(code).readline)
            if tok.type != tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    return tokenize.untokenize(tokens)


def extract_function_block(code: str, func_name: str) -> str:
    """Best-effort extraction of a Python function block from source text."""
    lines = code.splitlines()
    func_re = re.compile(rf'^\s*def\s+{re.escape(func_name)}\s*\(')
    for i, line in enumerate(lines):
        if not func_re.match(line):
            continue
        base_indent = len(line) - len(line.lstrip())
        block = [line]
        for nxt in lines[i + 1:]:
            stripped = nxt.strip()
            cur_indent = len(nxt) - len(nxt.lstrip())
            if stripped and cur_indent <= base_indent and re.match(r'^\s*(def|class)\s+\w+', nxt):
                break
            block.append(nxt)
        return "\n".join(block)
    return ""


RE_IMPORT_MAIN_AS = re.compile(r'import\s+__main__\s+as\s+(\w+)')
RE_FROM_IMPORT_MAIN = re.compile(r'from\s+__main__\s+import\s+')


def find_main_aliases(code: str) -> set[str]:
    """Collect local variable aliases bound to sys.modules['__main__']."""
    aliases = {"main"}
    for m in RE_MAIN_ALIAS_FROM_GET.finditer(code):
        aliases.add(m.group(1))
    for m in RE_MAIN_ALIAS_FROM_INDEX.finditer(code):
        aliases.add(m.group(1))
    for m in RE_IMPORT_MAIN_AS.finditer(code):
        aliases.add(m.group(1))
    return aliases


def find_scaled_mm_aliases(code: str) -> set[str]:
    """Collect local aliases bound directly to torch._scaled_mm."""
    aliases = set()
    for m in re.finditer(r'^\s*(\w+)\s*=\s*torch\._scaled_mm\s*$', code, re.MULTILINE):
        aliases.add(m.group(1))
    return aliases


def function_uses_scaled_mm(func_body: str, alias_names: set[str]) -> bool:
    """Return True when a function body calls torch._scaled_mm or one of its aliases."""
    if RE_SCALED_MM_PYTHON.search(func_body):
        return True
    for alias in alias_names:
        if re.search(rf'\b{re.escape(alias)}\s*\(', func_body):
            return True
    return False


BASE_ENTRYPOINT_NAMES = ("custom_kernel",)
ENTRYPOINT_NAMES = BASE_ENTRYPOINT_NAMES


def _entrypoint_candidates() -> tuple[str, ...]:
    return ENTRYPOINT_NAMES


def is_entrypoint_name(name: str) -> bool:
    return name in ENTRYPOINT_NAMES


def entrypoint_label(name: Optional[str] = None) -> str:
    return name or (ENTRYPOINT_NAMES[0] if ENTRYPOINT_NAMES else "entrypoint")


@dataclass
class SubmissionFacts:
    """Shared normalized views and AST summaries for one submission."""

    raw_code: str
    python_only: str
    python_active: str
    ast_tree: Optional[ast.AST]
    main_aliases: set[str]
    scaled_mm_aliases: set[str]
    trusted_aliases: dict[str, str]
    entrypoint_name: Optional[str]
    custom_kernel_pos: Optional[int]
    code_before_custom_kernel: str
    code_from_custom_kernel: str
    custom_kernel_code: str
    custom_kernel_active: str
    _function_blocks: dict[str, str] = field(default_factory=dict)
    _active_function_blocks: dict[str, str] = field(default_factory=dict)

    # --- Pre-computed AST indices (populated by _build_ast_index) ---
    # Nodes that contain a .data_ptr() call anywhere in their subtree
    _nodes_with_data_ptr: set[int] = field(default_factory=set)
    # Nodes that contain ._version attribute access
    _nodes_with_version: set[int] = field(default_factory=set)
    # Function names (non-entrypoint) whose body contains data_ptr / _version
    _data_ptr_helpers: set[str] = field(default_factory=set)
    _version_helpers: set[str] = field(default_factory=set)
    # Module-level vars initialized to None
    _none_inited: set[str] = field(default_factory=set)
    # All assignments: {target_name: [value_node, ...]}
    _assignments_by_target: dict[str, list] = field(default_factory=dict)
    # All import statements
    _imports: list = field(default_factory=list)
    _import_froms: list = field(default_factory=list)
    # Class definitions
    _class_defs: list = field(default_factory=list)

    def get_function_block(self, func_name: str) -> str:
        block = self._function_blocks.get(func_name)
        if block is None:
            block = extract_function_block(self.raw_code, func_name)
            self._function_blocks[func_name] = block
        return block

    def get_active_function_block(self, func_name: str) -> str:
        block = self._active_function_blocks.get(func_name)
        if block is None:
            block = strip_python_comments(self.get_function_block(func_name))
            self._active_function_blocks[func_name] = block
        return block


def build_submission_facts(code: str) -> SubmissionFacts:
    """Parse and normalize a submission once for reuse across all detectors."""
    python_only = strip_cpp_cuda_blocks(code)
    python_active = strip_python_comments(python_only)
    tree = _safe_ast_parse(code)
    entrypoint_name = None
    custom_kernel_match = None
    for candidate_name in _entrypoint_candidates():
        match = re.search(rf'^\s*def\s+{re.escape(candidate_name)}\s*\(', code, re.MULTILINE)
        if match is not None:
            entrypoint_name = candidate_name
            custom_kernel_match = match
            break
    custom_kernel_pos = custom_kernel_match.start() if custom_kernel_match else None
    code_before_custom_kernel = code[:custom_kernel_pos] if custom_kernel_pos is not None else code
    code_from_custom_kernel = code[custom_kernel_pos:] if custom_kernel_pos is not None else code
    custom_kernel_code = extract_function_block(code, entrypoint_name or entrypoint_label())
    custom_kernel_active = strip_python_comments(custom_kernel_code)
    trusted_aliases = _collect_trusted_aliases(tree) if tree is not None else {}

    facts = SubmissionFacts(
        raw_code=code,
        python_only=python_only,
        python_active=python_active,
        ast_tree=tree,
        main_aliases=find_main_aliases(python_only),
        scaled_mm_aliases=find_scaled_mm_aliases(code_before_custom_kernel),
        trusted_aliases=trusted_aliases,
        entrypoint_name=entrypoint_name,
        custom_kernel_pos=custom_kernel_pos,
        code_before_custom_kernel=code_before_custom_kernel,
        code_from_custom_kernel=code_from_custom_kernel,
        custom_kernel_code=custom_kernel_code,
        custom_kernel_active=custom_kernel_active,
    )
    if entrypoint_name:
        facts._function_blocks[entrypoint_name] = custom_kernel_code
        facts._active_function_blocks[entrypoint_name] = custom_kernel_active
    facts._function_blocks["custom_kernel"] = custom_kernel_code
    facts._active_function_blocks["custom_kernel"] = custom_kernel_active
    _build_ast_index(facts)
    return facts


def _build_ast_index(facts: SubmissionFacts) -> None:
    """Single-pass AST walk to populate all index fields on facts."""
    tree = facts.ast_tree
    if tree is None:
        return

    nodes_with_data_ptr: set[int] = set()
    nodes_with_version: set[int] = set()
    data_ptr_helpers: set[str] = set()
    version_helpers: set[str] = set()
    none_inited: set[str] = set()
    imports: list = []
    import_froms: list = []
    class_defs: list = []

    # Single walk: tag every node that is a data_ptr call or _version access
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "data_ptr":
                nodes_with_data_ptr.add(id(node))
        if isinstance(node, ast.Attribute) and node.attr == "_version":
            nodes_with_version.add(id(node))
        if isinstance(node, ast.Import):
            imports.append(node)
        elif isinstance(node, ast.ImportFrom):
            import_froms.append(node)
        elif isinstance(node, ast.ClassDef):
            class_defs.append(node)

    # Module-level None-initialized vars
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                for t in stmt.targets:
                    n = _ast_root_name(t)
                    if n:
                        none_inited.add(n)

    # Find helper functions (non-entrypoint) that contain data_ptr / _version
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if is_entrypoint_name(node.name):
            continue
        for child in ast.walk(node):
            if id(child) in nodes_with_data_ptr:
                data_ptr_helpers.add(node.name)
            if id(child) in nodes_with_version:
                version_helpers.add(node.name)
            if node.name in data_ptr_helpers and node.name in version_helpers:
                break

    # Propagate: mark ancestor expressions as containing data_ptr / _version
    # We need this for _expr_has_data_ptr / _expr_has_tensor_version replacements
    # Walk each assignment value and check if any descendant has the tag
    # This is still O(n) total since we do one walk and check set membership

    facts._nodes_with_data_ptr = nodes_with_data_ptr
    facts._nodes_with_version = nodes_with_version
    facts._data_ptr_helpers = data_ptr_helpers
    facts._version_helpers = version_helpers
    facts._none_inited = none_inited
    facts._imports = imports
    facts._import_froms = import_froms
    facts._class_defs = class_defs


def _expr_has_data_ptr_fast(expr: ast.AST | None, index: set[int]) -> bool:
    """O(subtree) check using pre-computed index — avoids full ast.walk per call."""
    if expr is None:
        return False
    for node in ast.walk(expr):
        if id(node) in index:
            return True
    return False


def _expr_has_version_fast(expr: ast.AST | None, index: set[int]) -> bool:
    if expr is None:
        return False
    for node in ast.walk(expr):
        if id(node) in index:
            return True
    return False


def ensure_submission_facts(code_or_facts: str | SubmissionFacts) -> SubmissionFacts:
    """Accept a raw code string or a pre-built SubmissionFacts object."""
    if isinstance(code_or_facts, SubmissionFacts):
        return code_or_facts
    return build_submission_facts(code_or_facts)


def _ast_root_name(expr: ast.AST | None) -> Optional[str]:
    """Return the left-most name that owns an expression, when present."""
    cur = expr
    while cur is not None:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.Attribute):
            cur = cur.value
            continue
        if isinstance(cur, ast.Subscript):
            cur = cur.value
            continue
        break
    return None


def _ast_dotted_name(expr: ast.AST | None) -> Optional[str]:
    """Return a dotted name such as torch.linalg.householder_product."""
    parts: list[str] = []
    cur = expr
    while cur is not None:
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
            continue
        break
    return None


def _expr_names(expr: ast.AST | None) -> set[str]:
    if expr is None:
        return set()
    return {
        node.id
        for node in ast.walk(expr)
        if isinstance(node, ast.Name)
    }


def _target_names(target: ast.AST | None) -> set[str]:
    """Return all simple names assigned by a target expression."""
    if target is None:
        return set()
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names.update(_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    root = _ast_root_name(target)
    return {root} if root else set()


def _expr_has_data_ptr(expr: ast.AST | None) -> bool:
    if expr is None:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "data_ptr"
        for node in ast.walk(expr)
    )


def _expr_has_tensor_version(expr: ast.AST | None) -> bool:
    if expr is None:
        return False
    return any(
        isinstance(node, ast.Attribute) and node.attr == "_version"
        for node in ast.walk(expr)
    )


_TRIVIAL_GPU_OPS = frozenset({
    "fill_", "zero_", "copy_", "fill", "zero", "record",
})


def _body_has_calls(body: list[ast.stmt]) -> bool:
    """Return True if the body contains non-trivial function calls.

    Tiny GPU ops like ``_tiny.fill_(0)`` or ``_anchor.copy_(_anchor)`` are
    common dummy work used to keep CUDA timers non-zero; they don't count
    as real compute and should not prevent replay detection.
    """
    for stmt in body:
        for nested in ast.walk(stmt):
            if not isinstance(nested, ast.Call):
                continue
            # Allow trivial method calls: obj.fill_(0), obj.copy_(obj), etc.
            if (isinstance(nested.func, ast.Attribute)
                    and nested.func.attr in _TRIVIAL_GPU_OPS):
                continue
            return True
    return False


def _looks_stateful_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("last", "prev", "cache", "saved", "memo"))


_ENTRYPOINT_METHOD_NAMES = ("__call__", "forward", "run", "solve")


def _iter_non_nested_nodes(node: ast.AST):
    """Yield descendants without descending into nested function/class scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield child
            continue
        yield child
        yield from _iter_non_nested_nodes(child)


def _function_input_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
    if fn.name in _ENTRYPOINT_METHOD_NAMES and args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    names = {arg.arg for arg in args}
    if fn.args.vararg is not None:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg is not None:
        names.add(fn.args.kwarg.arg)
    return names


def _method_from_class(cls: ast.ClassDef, preferred: tuple[str, ...] = _ENTRYPOINT_METHOD_NAMES):
    methods = {
        child.name: child
        for child in cls.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in preferred:
        if name in methods:
            return methods[name]
    return None


def _factory_returned_function(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    nested = {
        child.name: child
        for child in fn.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for stmt in fn.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            returned = nested.get(stmt.value.id)
            if returned is not None:
                return returned
    return None


def _entrypoint_function_nodes(facts: SubmissionFacts) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Resolve simple Python callable exports for entrypoint-scoped detectors."""
    tree = facts.ast_tree
    if tree is None:
        return []

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    classes: dict[str, ast.ClassDef] = {}
    instances: dict[str, str] = {}
    aliases: dict[str, str] = {}
    resolved: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    seen: set[int] = set()

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[stmt.name] = stmt
        elif isinstance(stmt, ast.ClassDef):
            classes[stmt.name] = stmt

    def add(fn: ast.FunctionDef | ast.AsyncFunctionDef | None) -> None:
        if fn is None or id(fn) in seen:
            return
        seen.add(id(fn))
        resolved.append(fn)

    def resolve_name(name: str) -> str:
        while name in aliases and aliases[name] != name:
            name = aliases[name]
        return name

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_entrypoint_name(stmt.name):
            add(stmt)
        elif isinstance(stmt, ast.ClassDef) and is_entrypoint_name(stmt.name):
            add(_method_from_class(stmt))

        if not isinstance(stmt, ast.Assign):
            continue
        target_names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        if not target_names:
            continue

        value = stmt.value
        if isinstance(value, ast.Name):
            value_name = resolve_name(value.id)
            for target in target_names:
                aliases[target] = value_name
                if is_entrypoint_name(target):
                    add(functions.get(value_name))
                    add(_method_from_class(classes[value_name]) if value_name in classes else None)
                elif value_name in classes:
                    instances[target] = value_name
        elif isinstance(value, ast.Call):
            callee = value.func
            if isinstance(callee, ast.Name):
                callee_name = resolve_name(callee.id)
                if callee_name in classes:
                    for target in target_names:
                        instances[target] = callee_name
                        if is_entrypoint_name(target):
                            add(_method_from_class(classes[callee_name]))
                elif callee_name == "partial" and value.args and isinstance(value.args[0], ast.Name):
                    fn = functions.get(resolve_name(value.args[0].id))
                    for target in target_names:
                        if is_entrypoint_name(target):
                            add(fn)
                elif callee_name in functions:
                    for target in target_names:
                        if is_entrypoint_name(target):
                            add(_factory_returned_function(functions[callee_name]) or functions[callee_name])
            elif isinstance(callee, ast.Attribute):
                if callee.attr == "partial" and value.args and isinstance(value.args[0], ast.Name):
                    fn = functions.get(resolve_name(value.args[0].id))
                    for target in target_names:
                        if is_entrypoint_name(target):
                            add(fn)
                owner = callee.value
                if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name):
                    class_name = resolve_name(owner.func.id)
                    if class_name in classes and callee.attr in _ENTRYPOINT_METHOD_NAMES:
                        for target in target_names:
                            if is_entrypoint_name(target):
                                add(_method_from_class(classes[class_name], (callee.attr,)))
        elif isinstance(value, ast.Attribute):
            owner = value.value
            if value.attr in _ENTRYPOINT_METHOD_NAMES and isinstance(owner, ast.Name):
                owner_name = resolve_name(owner.id)
                class_name = instances.get(owner_name, owner_name if owner_name in classes else "")
                if class_name in classes:
                    for target in target_names:
                        if is_entrypoint_name(target):
                            add(_method_from_class(classes[class_name], (value.attr,)))

    return resolved


def _expr_is_none(expr: ast.AST | None) -> bool:
    return isinstance(expr, ast.Constant) and expr.value is None


def _static_string(expr: ast.AST | None) -> Optional[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        parts: list[str] = []
        for value in expr.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return None
            parts.append(value.value)
        return "".join(parts)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _static_string(expr.left)
        right = _static_string(expr.right)
        if left is not None and right is not None:
            return left + right
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "join"
        and _static_string(expr.func.value) is not None
        and len(expr.args) == 1
        and isinstance(expr.args[0], (ast.List, ast.Tuple))
    ):
        parts = [_static_string(elt) for elt in expr.args[0].elts]
        if all(part is not None for part in parts):
            return _static_string(expr.func.value).join(parts)  # type: ignore[arg-type]
    return None


def _expr_has_benchmark_literal(expr: ast.AST | None) -> bool:
    if expr is None:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r'Ranked\s+Benchmark|BENCHMARK_PASSED|\bbenchmark\b\s*[:=]?|score\s*[:=]', node.value, re.IGNORECASE):
                return True
    return False


def _expr_has_decode_like_call(expr: ast.AST | None, helper_names: set[str] | None = None) -> bool:
    if expr is None:
        return False
    helper_names = helper_names or set()
    decode_names = {
        "decode", "decompress", "b64decode", "b32decode", "b16decode",
        "urlsafe_b64decode", "decodebytes", "decodestring", "unhexlify",
        "a2b_hex", "a2b_base64", "bytes", "bytearray", "chr",
    }
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and (node.func.id in decode_names or node.func.id in helper_names):
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in decode_names:
            return True
    return False


def _expr_contains_input_derived_call(expr: ast.AST | None, input_names: set[str]) -> bool:
    if expr is None:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and _ast_root_name(node.func.value) in input_names:
                return True
            if isinstance(node.func, ast.Name) and any(_expr_names(arg) & input_names for arg in node.args):
                return True
    return False


def _is_input_float_call(expr: ast.AST | None, input_names: set[str]) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "float"
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id in input_names
        and not expr.args
        and not expr.keywords
    )


def _is_input_attr_float_call(expr: ast.AST | None, owner_name: str) -> Optional[str]:
    if not (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "float"
        and isinstance(expr.func.value, ast.Attribute)
        and isinstance(expr.func.value.value, ast.Name)
        and expr.func.value.value.id == owner_name
        and not expr.args
        and not expr.keywords
    ):
        return None
    return expr.func.value.attr


def _lambda_input_names(expr: ast.Lambda) -> set[str]:
    args = list(expr.args.posonlyargs) + list(expr.args.args) + list(expr.args.kwonlyargs)
    if args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    return {arg.arg for arg in args}


def _lambda_returns_input_float(expr: ast.AST | None) -> bool:
    if not isinstance(expr, ast.Lambda):
        return False
    input_names = _lambda_input_names(expr)
    return bool(input_names) and _is_input_float_call(expr.body, input_names)


def _torch_alias_sets(facts: SubmissionFacts) -> tuple[set[str], dict[str, str]]:
    torch_aliases = {"torch"}
    imported: dict[str, str] = {}
    tree = facts.ast_tree
    if tree is None:
        return torch_aliases, imported
    for node in facts._imports:
        for alias in node.names:
            if alias.name == "torch":
                torch_aliases.add(alias.asname or alias.name)
    for node in facts._import_froms:
        if node.module != "torch":
            continue
        for alias in node.names:
            imported[alias.asname or alias.name] = alias.name
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        dotted = _ast_dotted_name(stmt.value)
        if dotted and dotted.startswith(tuple(f"{alias}." for alias in torch_aliases)):
            attr = dotted.rsplit(".", 1)[-1]
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    imported[target.id] = attr
    return torch_aliases, imported


def _torch_call_name(call: ast.Call, torch_aliases: set[str], imported: dict[str, str]) -> Optional[str]:
    if isinstance(call.func, ast.Name):
        return imported.get(call.func.id)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        if call.func.value.id in torch_aliases:
            return call.func.attr
    return None


def _is_transpose_of(expr: ast.AST | None, name: str) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr in {"t", "transpose", "transpose_"}
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == name
    )


def _looks_output_value_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {"h", "q", "r", "y", "z"}
        or any(token in lowered for token in ("out", "output", "result", "res", "ans"))
    )


def _cache_name_allows_output_replay(name: Optional[str]) -> bool:
    if not name:
        return False
    lowered = name.lower()
    if any(token in lowered for token in (
        "compile", "compiled", "module", "plan", "workspace", "preprocess",
        "scale", "decoded", "meta", "sort", "view", "shape", "quant", "pad",
    )):
        return False
    return any(token in lowered for token in (
        "cache", "reuse", "saved", "memo", "result", "out", "compute", "gemm", "bank",
    ))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_timer_monkeypatch(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 1: torch.cuda.Event replaced with fake class."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    matches = []
    if RE_CUDA_EVENT_ASSIGN.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "torch.cuda.Event reassigned",
        })
    if RE_CUDA_EVENT_SETATTR.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "setattr(torch.cuda, 'Event', ...) patching",
        })
    if RE_ELAPSED_TIME_METHOD_PATCH.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "torch.cuda.Event.elapsed_time method patched",
        })
    if RE_CUDA_SYNCHRONIZE_PATCH.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "torch.cuda.synchronize patched",
        })
    if RE_DUMMY_FAKE_EVENT.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "Dummy/Fake Event class defined",
        })
    tree = facts.ast_tree
    if tree is not None:
        time_aliases = {"time"}
        patch_aliases = {"patch"}
        mock_module_aliases = {"mock"}
        for node in facts._imports:
            for alias in node.names:
                if alias.name == "time":
                    time_aliases.add(alias.asname or alias.name)
                elif alias.name == "unittest.mock":
                    mock_module_aliases.add(alias.asname or alias.name)
        for node in facts._import_froms:
            if node.module == "unittest.mock":
                for alias in node.names:
                    if alias.name == "patch":
                        patch_aliases.add(alias.asname or alias.name)
            elif node.module == "unittest":
                for alias in node.names:
                    if alias.name == "mock":
                        mock_module_aliases.add(alias.asname or alias.name)

        clock_attrs = {
            "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
            "process_time", "process_time_ns", "time", "time_ns",
        }
        seen_time_patches: set[str] = set()
        seen_cuda_timer_patches: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in time_aliases
                        and target.attr in clock_attrs
                    ):
                        seen_time_patches.add(f"{target.value.id}.{target.attr}")
                    if isinstance(target, ast.Subscript):
                        key = _static_string(target.slice)
                        owner = _ast_dotted_name(target.value)
                        if owner == "torch.cuda.__dict__" and key in {"Event", "synchronize"}:
                            seen_cuda_timer_patches.add(f"torch.cuda.__dict__[{key!r}]")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in time_aliases
            ):
                attr = _static_string(node.args[1])
                if attr in clock_attrs:
                    seen_time_patches.add(f"setattr({node.args[0].id}, {attr!r}, ...)")
            elif isinstance(node, ast.Call):
                func_name = _ast_dotted_name(node.func)
                is_patch_call = (
                    (isinstance(node.func, ast.Name) and node.func.id in patch_aliases)
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "patch"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in mock_module_aliases
                    )
                    or func_name == "unittest.mock.patch"
                )
                target = _static_string(node.args[0]) if is_patch_call and node.args else None
                if target in {"torch.cuda.Event", "torch.cuda.synchronize", "torch.cuda.Event.elapsed_time"}:
                    seen_cuda_timer_patches.add(f"unittest.mock.patch({target!r})")
        if seen_time_patches:
            matches.append({
                "pattern": "TIMER_MONKEYPATCH",
                "severity": "critical",
                "evidence": "host timing function patched: " + ", ".join(sorted(seen_time_patches)),
            })
        if seen_cuda_timer_patches:
            matches.append({
                "pattern": "TIMER_MONKEYPATCH",
                "severity": "critical",
                "evidence": "CUDA timing primitive patched: " + ", ".join(sorted(seen_cuda_timer_patches)),
            })
    return matches


def detect_print_injection(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 2: separate real benchmark spoofing from benign stdio handling."""
    facts = ensure_submission_facts(code)
    python_active = facts.python_active
    matches = []
    seen: set[tuple[str, str]] = set()

    def add(pattern: str, severity: str, evidence: str):
        key = (pattern, evidence)
        if key in seen:
            return
        seen.add(key)
        matches.append({
            "pattern": pattern,
            "severity": severity,
            "evidence": evidence,
        })

    fake_emit = False
    if RE_EMIT_FAKE_BENCH.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "_emit_fake_benchmark function defined")
    if RE_BUILTINS_PRINT_ASSIGN.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "builtins.print reassigned")
    if RE_RANKED_BENCHMARK_PRINT.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "Ranked Benchmark string printed directly")
    elif RE_PATCHED_PRINT.search(python_active) and RE_RANKED_BENCHMARK_STR.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "print patching plus Ranked Benchmark payload")

    # Stdio redirection is common as a compatibility shim around compilation and
    # should not be treated as benchmark spoofing on its own.
    if RE_SYS_STDOUT_REPLACE.search(python_active):
        if re.search(r'if\s+sys\.stdout\s+is\s+None\s*:', python_active):
            add("STDIO_REDIRECT", "low", "guarded sys.stdout compatibility redirect")
        else:
            add("STDIO_REDIRECT", "medium", "sys.stdout reassigned")
    if RE_SYS_STDERR_REPLACE.search(python_active):
        if re.search(r'if\s+sys\.stderr\s+is\s+None\s*:', python_active):
            add("STDIO_REDIRECT", "low", "guarded sys.stderr compatibility redirect")
        else:
            add("STDIO_REDIRECT", "medium", "sys.stderr reassigned")

    tree = facts.ast_tree
    if tree is not None:
        sys_aliases = {"sys"}
        os_aliases = {"os"}
        os_write_aliases = set()
        for node in facts._imports:
            for alias in node.names:
                if alias.name == "sys":
                    sys_aliases.add(alias.asname or alias.name)
                elif alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
        for node in facts._import_froms:
            if node.module == "os":
                for alias in node.names:
                    if alias.name == "write":
                        os_write_aliases.add(alias.asname or alias.name)

        print_fns = list(_entrypoint_function_nodes(facts))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "kernel":
                print_fns.append(node)

        for fn in print_fns:
            direct_emits: set[str] = set()
            for node in _iter_non_nested_nodes(fn):
                if not isinstance(node, ast.Call):
                    continue
                payload = " ".join(
                    part
                    for arg in node.args
                    if (part := _static_string(arg)) is not None
                )
                spoof_payload = bool(
                    re.search(r'Ranked\s+Benchmark|BENCHMARK_PASSED|\bbenchmark\b\s*[:=]?|score\s*[:=]', payload, re.IGNORECASE)
                    or any(_expr_has_benchmark_literal(arg) for arg in node.args)
                )
                if isinstance(node.func, ast.Name):
                    if node.func.id == "print" and spoof_payload:
                        direct_emits.add("print(...) in entrypoint")
                    elif node.func.id in os_write_aliases and node.args:
                        fd = node.args[0]
                        if isinstance(fd, ast.Constant) and fd.value in (1, 2) and spoof_payload:
                            direct_emits.add(f"{node.func.id}({fd.value}, ...) in entrypoint")
                elif isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "write"
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr in {"stdout", "stderr"}
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id in sys_aliases
                        and spoof_payload
                    ):
                        direct_emits.add(f"{node.func.value.value.id}.{node.func.value.attr}.write(...) in entrypoint")
                    elif (
                        node.func.attr == "write"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in os_aliases
                        and node.args
                    ):
                        fd = node.args[0]
                        if isinstance(fd, ast.Constant) and fd.value in (1, 2) and spoof_payload:
                            direct_emits.add(f"{node.func.value.id}.write({fd.value}, ...) in entrypoint")
            if direct_emits:
                fake_emit = True
                add("FAKE_BENCHMARK_EMIT", "critical", "; ".join(sorted(direct_emits)))
                break

    # Do not keep the old broad PRINT_INJECTION marker; the split rules carry
    # the action semantics now.
    return matches


def detect_evaluator_exploit(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 3: Patching evaluator internals (main.time_kernel, etc.)."""
    facts = ensure_submission_facts(code)
    matches = []
    seen = set()

    def add_match(evidence: str):
        if evidence in seen:
            return
        seen.add(evidence)
        matches.append({
            "pattern": "EVALUATOR_EXPLOIT",
            "severity": "critical",
            "evidence": evidence,
        })

    # Strip large data blobs (base64, lookup tables) before pattern matching.
    # Blobs cause O(n²) backtracking in patterns like (\w+)\s*=\s*literal when
    # they form very long single-line word runs (e.g. 129KB base64 strings).
    code = facts.python_only
    main_patch_targets = sorted(TRUSTED_FUNCTIONS)

    for attr in main_patch_targets:
        if re.search(rf'\bmain\.{re.escape(attr)}\s*=', code):
            add_match(f"main.{attr} patched")

    aliases = facts.main_aliases
    # Check for alias-based patching regardless of how __main__ was obtained
    for obj in sorted(aliases):
        if obj == "main":
            continue
        for attr in main_patch_targets:
            if re.search(rf'\b{re.escape(obj)}\.{re.escape(attr)}\s*=', code):
                add_match(f"{obj}.{attr} patched via __main__ alias")
    for obj in sorted(aliases):
        for attr in main_patch_targets:
            if re.search(
                rf'setattr\s*\(\s*{re.escape(obj)}\s*,\s*[\'"]{re.escape(attr)}[\'"]\s*,',
                code,
            ):
                add_match(f"setattr({obj}, '{attr}', ...) on __main__ alias")
    # Also flag if __main__ is imported directly (unusual in a kernel submission)
    if RE_FROM_IMPORT_MAIN.search(code):
        add_match("from __main__ import ... (direct harness access)")
    return matches


def detect_scaled_mm_ref(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 4: Using torch._scaled_mm as primary compute.

    Scope-aware: if the file has a configured entrypoint function and
    `_scaled_mm` only appears BEFORE that function, it's likely a
    reference implementation (not the submission's compute path) and
    should not be flagged.
    """
    facts = ensure_submission_facts(code)
    matches = []
    code = facts.raw_code
    entrypoint_name = entrypoint_label(facts.entrypoint_name)
    custom_kernel_pos = facts.custom_kernel_pos or 0

    # For scope-aware check: code from the configured entrypoint onward.
    # If no entrypoint is found, check the entire file (conservative).
    if facts.custom_kernel_pos is not None:
        # Check if _scaled_mm is used at or after the configured entrypoint,
        # or if _scaled_mm is aliased to a variable that the entrypoint could
        # call indirectly.
        code_from_ck = facts.code_from_custom_kernel
        code_before_ck = facts.code_before_custom_kernel

        has_python_after = bool(RE_SCALED_MM_PYTHON.search(code_from_ck))
        has_alias_after = bool(RE_SCALED_MM_ALIAS.search(code_from_ck))
        alias_names_before = facts.scaled_mm_aliases
        has_alias_before = bool(alias_names_before)
        has_cpp = bool(RE_SCALED_MM_CPP.search(code))
        has_schema = bool(RE_SCALED_MM_SCHEMA.search(code))

        # _scaled_mm only before the configured entrypoint — check if it actually
        # reaches that code path through a helper or a direct alias call.
        has_python_before = bool(RE_SCALED_MM_PYTHON.search(code_before_ck))
        if ((has_python_before or has_alias_before) and not has_python_after and
                not has_alias_after and
                not has_cpp and not has_schema):
            # Find function names defined before the entrypoint that use _scaled_mm
            helper_funcs_with_mm = set()
            for func_m in re.finditer(r'^def\s+(\w+)\s*\(', code_before_ck, re.MULTILINE):
                func_name = func_m.group(1)
                func_body = facts.get_function_block(func_name)
                if function_uses_scaled_mm(func_body, alias_names_before):
                    helper_funcs_with_mm.add(func_name)
            # Strip Python comments from the entrypoint body before call-site analysis
            # so that commented-out calls (e.g. # result = ref_kernel(data)) are ignored.
            ck_active = facts.custom_kernel_active
            # Check if the entrypoint output comes from a helper that uses _scaled_mm.
            # Validation-only calls (result not returned) are NOT the submission's compute.
            compute_helpers = set()
            for fn in helper_funcs_with_mm:
                # Direct return: return helper(...)
                if re.search(rf'\breturn\s+{re.escape(fn)}\s*\(', ck_active):
                    compute_helpers.add(fn)
                    continue
                # Assigned return: var = helper(...) … return var
                assign_m = re.search(rf'\b(\w+)\s*=\s*{re.escape(fn)}\s*\(', ck_active)
                if assign_m:
                    var = assign_m.group(1)
                    after = ck_active[assign_m.end():]
                    if re.search(rf'\breturn\s+{re.escape(var)}\b', after):
                        compute_helpers.add(fn)
            for alias in alias_names_before:
                if re.search(rf'\breturn\s+{re.escape(alias)}\s*\(', ck_active):
                    compute_helpers.add(alias)
                    continue
                assign_m = re.search(rf'\b(\w+)\s*=\s*{re.escape(alias)}\s*\(', ck_active)
                if assign_m:
                    var = assign_m.group(1)
                    after = ck_active[assign_m.end():]
                    if re.search(rf'\breturn\s+{re.escape(var)}\b', after):
                        compute_helpers.add(alias)
            if compute_helpers:
                # Distinguish pure delegation from benchmark-shape if/else fallback.
                # Two conditions must BOTH hold to downgrade to medium:
                # 1. There is a primary non-ref compute branch (more return paths than
                #    just the ref-helper returns) → ref call is a shape-gated fallback.
                # 2. All compute helpers have "ref" in their name — only the evaluator's
                #    own ref_kernel (or similar) counts as a true correctness fallback.
                #    A custom-named helper (e.g. cublas_kernel, fast_path) that wraps
                #    _scaled_mm is still a hack regardless of branching.
                total_returns = len(re.findall(r'\breturn\b', ck_active))
                # Count return paths that go through a ref helper — both direct
                # ("return ref_kernel(data)") and assign-then-return
                # ("results = ref_kernel(data); return results").
                ref_path_returns = 0
                for fn in compute_helpers:
                    ref_path_returns += len(
                        re.findall(rf'\breturn\s+{re.escape(fn)}\s*\(', ck_active)
                    )
                    for am in re.finditer(rf'\b(\w+)\s*=\s*{re.escape(fn)}\s*\(', ck_active):
                        var = am.group(1)
                        if re.search(rf'\breturn\s+{re.escape(var)}\b', ck_active[am.end():]):
                            ref_path_returns += 1
                has_primary_custom_path = total_returns > ref_path_returns
                all_helpers_are_ref_named = all(
                    'ref' in fn.lower() for fn in compute_helpers
                )
                if has_primary_custom_path and all_helpers_are_ref_named:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "medium",
                        "evidence": (
                            f"_scaled_mm helper(s) used as shape-gated fallback in custom_kernel: "
                            f"{', '.join(sorted(compute_helpers))} "
                            f"(primary path is custom, ref only in conditional branch)"
                        ).replace("custom_kernel", entrypoint_name),
                    })
                else:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "high",
                        "evidence": (
                            f"{entrypoint_name} returns output of _scaled_mm helper(s): "
                            f"{', '.join(sorted(compute_helpers))} "
                        ),
                    })
                return matches
            # Helper called for validation only (result not returned).
            # If called as a bare statement (no assignment), the helper writes
            # into an in-place output argument — treat as low severity.
            # Otherwise (assigned but not returned) it is a medium signal.
            validation_calls = {fn for fn in helper_funcs_with_mm
                                if re.search(rf'\b{re.escape(fn)}\s*\(', ck_active)}
            validation_calls.update(
                alias for alias in alias_names_before
                if re.search(rf'\b{re.escape(alias)}\s*\(', ck_active)
            )
            if validation_calls:
                inplace = {fn for fn in validation_calls
                           if re.search(rf'^\s*{re.escape(fn)}\s*\(', ck_active, re.MULTILINE)}
                non_inplace = validation_calls - inplace
                if non_inplace:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "medium",
                        "evidence": (
                            f"_scaled_mm helper(s) called in {entrypoint_name} but result not returned: "
                            f"{', '.join(sorted(non_inplace))}"
                        ),
                    })
                if inplace:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "low",
                        "evidence": f"_scaled_mm helper(s) called as in-place mutation (result discarded): {', '.join(sorted(inplace))}",
                    })
                return matches
            # Helper defined but not called from the entrypoint (e.g. commented-out) → medium
            if helper_funcs_with_mm:
                matches.append({
                    "pattern": "SCALED_MM_REF",
                    "severity": "medium",
                    "evidence": (
                        f"_scaled_mm in helper(s) before {entrypoint_name}: "
                        f"{', '.join(sorted(helper_funcs_with_mm))} (not called in active code)"
                    ),
                })
                return matches
            # _scaled_mm or an alias only appears before the entrypoint and is never
            # reached from it → likely reference/dead code, not the submission path.
            return []

        has_python = has_python_after
        has_alias = has_alias_after
    else:
        has_python = bool(RE_SCALED_MM_PYTHON.search(code))
        has_alias = bool(RE_SCALED_MM_ALIAS.search(code))
        has_cpp = bool(RE_SCALED_MM_CPP.search(code))
        has_schema = bool(RE_SCALED_MM_SCHEMA.search(code))

    if has_python or has_alias or has_cpp or has_schema:
        evidence_parts = []
        if has_python:
            evidence_parts.append("torch._scaled_mm() called in Python")
        if has_alias:
            evidence_parts.append("torch._scaled_mm aliased to variable")
        if has_cpp:
            evidence_parts.append("aten::_scaled_mm in C++ code")
        if has_schema:
            evidence_parts.append("findSchemaOrThrow for _scaled_mm")
        matches.append({
            "pattern": "SCALED_MM_REF",
            "severity": "high",
            "evidence": "; ".join(evidence_parts),
        })
    return matches


def detect_decode_mm_ref(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 5: FP4 decode + torch.mm/bmm as main compute path.

    Only flags when mm/bmm/matmul appears to feed the output (near return
    or assigned to a result-like variable), not when used for small
    pre/post processing steps.
    """
    facts = ensure_submission_facts(code)
    python_only = facts.python_only
    custom_kernel_code = facts.custom_kernel_code
    entrypoint_name = entrypoint_label(facts.entrypoint_name)
    matches = []

    # Tighten scope: only count mm/bmm/matmul used in the submission entrypoint.
    if not custom_kernel_code:
        return matches

    has_mm = bool(RE_TORCH_MM.search(custom_kernel_code))
    has_bmm = bool(RE_TORCH_BMM.search(custom_kernel_code))
    has_matmul = bool(RE_TORCH_MATMUL.search(custom_kernel_code))

    if has_mm or has_bmm or has_matmul:
        ops = []
        if has_mm:
            ops.append("torch.mm")
        if has_bmm:
            ops.append("torch.bmm")
        if has_matmul:
            ops.append("torch.matmul")

        # Check if mm/bmm feeds the output path (return, result assignment, or
        # is the dominant compute — more than one call suggests it IS the kernel)
        mm_call_count = (
            len(RE_TORCH_MM.findall(custom_kernel_code)) +
            len(RE_TORCH_BMM.findall(custom_kernel_code)) +
            len(RE_TORCH_MATMUL.findall(custom_kernel_code))
        )
        feeds_output = bool(re.search(
            r'(?:result|output|out|c_ref|c|ret)\s*=\s*torch\.(?:mm|bmm|matmul)\s*\('
            r'|return\s+.*?torch\.(?:mm|bmm|matmul)\s*\(',
            custom_kernel_code,
        ))

        # If only 1 call and doesn't feed output, likely a small helper step
        if mm_call_count == 1 and not feeds_output:
            return matches

        has_decode_context = bool(
            re.search(r'\b(?:fp4|decode|sfa|sfb|scale_a|scale_b|to_blocked)\b', python_only, re.IGNORECASE),
        )
        severity = "high" if has_decode_context else "medium"
        evidence = f"{', '.join(ops)} used inside {entrypoint_name}"
        if not has_decode_context:
            evidence += " (no explicit FP4 decode context)"
        if feeds_output:
            evidence += " [feeds output path]"
        matches.append({
            "pattern": "DECODE_MM_REF",
            "severity": severity,
            "evidence": evidence,
        })
    return matches


def detect_result_caching(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 6: split output replay from benign workspace/preprocess caches."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    python_only = facts.python_only
    python_active = facts.python_active
    matches = []

    output_replay_signals = []
    preprocess_signals = []
    workspace_signals = []
    runner_plan_signals = []

    if RE_WEAK_VALUE_DICT.search(python_only):
        workspace_signals.append("WeakValueDictionary")
    if RE_DECODED_CACHE.search(python_only):
        preprocess_signals.append("_decoded_cache")
    if RE_PREPROCESS_CACHE.search(python_only):
        preprocess_signals.append("_PREPROCESS_CACHE")
    if RE_SCALE_CACHE.search(python_only):
        preprocess_signals.append("_scale_cache")
    if RE_RESULT_CACHE_GENERAL.search(python_only):
        workspace_signals.append("_RESULT/_GROUPED_RESULT/_COMPUTE/_GEMM_CACHE")
    if RE_VERSION_CACHE.search(python_only):
        workspace_signals.append("tensor._version cache check")

    # Scope all strong-signal checks to the configured entrypoint body.
    # Helper functions that cache compilation artifacts (TensorMap, compiled kernels,
    # plan descriptors, etc.) return those objects from caches, but that is legitimate —
    # only a cache inside the entrypoint itself indicates result caching.
    cache_scope = facts.custom_kernel_active if facts.custom_kernel_active else python_active

    stores_output = bool(RE_CACHE_STORE_OUTPUT.search(cache_scope))
    if RE_RESULT_REUSE.search(cache_scope):
        output_replay_signals.append("_result_reuse")
    if RE_OUTPUT_CACHE.search(cache_scope):
        workspace_signals.append("_OUTPUT_CACHE")
    if RE_ID_DATA_CACHE.search(cache_scope):
        workspace_signals.append("id(data) cache key")
    if RE_DATA_PTR_CACHE_KEY.search(cache_scope):
        workspace_signals.append("data_ptr() cache key")
    if RE_RETURN_CACHE_INDEX.search(cache_scope):
        output_replay_signals.append("direct return from cache[...]")
    if stores_output and output_replay_signals:
        output_replay_signals.append("cache[...] stores output/result tensor")
    elif stores_output:
        workspace_signals.append("cache[...] stores reusable output/result tensor")

    for var, cache_name in RE_CACHE_GET_ASSIGN.findall(cache_scope):
        cache_lower = cache_name.lower()
        if any(token in cache_lower for token in ("plan", "dispatch", "runner", "config")):
            runner_plan_signals.append(f"{cache_name}.get(...) runner/plan cache")
        elif any(token in cache_lower for token in ("decoded", "preprocess", "scale", "sort", "view", "shape", "quant", "meta", "pad")):
            preprocess_signals.append(f"{cache_name}.get(...) preprocess cache")
        elif re.search(
            rf'if\s+{re.escape(var)}\s+is\s+not\s+None\s*:[\s\S]{{0,300}}?\breturn\s+{re.escape(var)}\b',
            cache_scope,
        ):
            output_replay_signals.append(f"{cache_name}.get(...) then return cached value")
        elif re.search(
            rf'if\s+{re.escape(var)}\s*:[\s\S]{{0,300}}?\breturn\s+{re.escape(var)}\b',
            cache_scope,
        ):
            output_replay_signals.append(f"{cache_name}.get(...) guarded return")
        else:
            workspace_signals.append(f"{cache_name}.get(...) workspace cache")

    # C++ thread_local cache keyed by data_ptr (e.g., lastA = A.data_ptr())
    cpp_thread_local_cache = re.findall(
        r'thread_local\s+(?:\w+\s+)?(?:cache|Cache)\b',
        code,
    )
    cpp_last_ptr_store = re.findall(
        r'(?:cache\.last|cache\.prev|lastA|lastB|prevA|prevB)\w*\s*(?:\[.*?\])?\s*=\s*.*?'
        r'(?:data_ptr|\.ptr\b)',
        code,
    )
    cpp_data_ptr_compare = re.findall(
        r'(?:cache\.last|cache\.prev|lastA|lastB)\w*\s*(?:\[.*?\])?\s*!='
        r'|'
        r'(?:uint64_t|uintptr_t)\s+\w+p?\s*=\s*\((?:uint64_t|uintptr_t)\)\s*\w+\.data_ptr',
        code,
    )
    if cpp_thread_local_cache and (cpp_last_ptr_store or cpp_data_ptr_compare):
        if RE_CPP_CACHE_FAST_RETURN.search(code):
            output_replay_signals.append("C++ pointer cache fast-return path")
        else:
            workspace_signals.append("C++ thread_local cache keyed by data_ptr()")

    # CUDA graph replay (CUDAGraph + .replay()) explains the cache-and-return
    # pattern — the graph replays actual GPU computation, not cached results.
    # Downgrade cache-return signals when CUDA graph replay is present.
    has_cuda_graph_replay = (
        RE_CUDA_GRAPH_PYTHON.search(python_only) and RE_CUDA_GRAPH_REPLAY.search(python_only)
    )
    if has_cuda_graph_replay:
        workspace_signals.append("CUDA graph replay cache")
        output_replay_signals = [
            s for s in output_replay_signals
            if s not in {
                "direct return from cache[...]",
                "cache[...] stores output/result tensor",
            }
        ]

    output_replay_signals = list(dict.fromkeys(output_replay_signals))
    preprocess_signals = list(dict.fromkeys(preprocess_signals))
    workspace_signals = list(dict.fromkeys(workspace_signals))
    runner_plan_signals = list(dict.fromkeys(runner_plan_signals))

    if output_replay_signals:
        matches.append({
            "pattern": "OUTPUT_REPLAY_CACHE",
            "severity": "high",
            "evidence": "Output replay signals: " + ", ".join(output_replay_signals),
        })
    if preprocess_signals:
        matches.append({
            "pattern": "PREPROCESS_CACHE",
            "severity": "low",
            "evidence": "Preprocess cache signals: " + ", ".join(preprocess_signals),
        })
    if workspace_signals:
        matches.append({
            "pattern": "WORKSPACE_CACHE",
            "severity": "low",
            "evidence": "Workspace cache signals: " + ", ".join(workspace_signals),
        })
    if runner_plan_signals:
        matches.append({
            "pattern": "RUNNER_PLAN_CACHE",
            "severity": "low",
            "evidence": "Runner/plan cache signals: " + ", ".join(runner_plan_signals),
        })
    return matches


def detect_helper_output_replay_cache(code: str | SubmissionFacts) -> list[dict]:
    """Detect helper-level output caches returned by the submitted entrypoint."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not functions:
        return []

    torch_aliases, imported_torch = _torch_alias_sets(facts)
    tensor_return_ops = {
        "matmul", "mm", "bmm", "zeros", "empty", "ones", "full",
        "zeros_like", "empty_like", "ones_like", "full_like",
    }

    def tensor_output_expr(expr: ast.AST | None, input_names: set[str], output_names: set[str]) -> bool:
        if expr is None:
            return False
        if _expr_names(expr) & (input_names | output_names):
            return True
        for call in [n for n in ast.walk(expr) if isinstance(n, ast.Call)]:
            if _torch_call_name(call, torch_aliases, imported_torch) in tensor_return_ops:
                return True
            for kw in call.keywords:
                if kw.arg == "device" and _static_string(kw.value) == "cuda":
                    return True
        return False

    def cache_lookup(value: ast.AST | None) -> tuple[Optional[str], bool]:
        if isinstance(value, ast.Subscript):
            cache_name = _ast_root_name(value.value)
            return cache_name, _cache_name_allows_output_replay(cache_name)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr in {"get", "pop"}:
                cache_name = _ast_root_name(value.func.value)
                return cache_name, _cache_name_allows_output_replay(cache_name)
        return None, False

    def helper_has_cache_replay(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
        input_names = _function_input_names(fn)
        output_names: set[str] = set()
        cache_hits: set[str] = set()
        stored_caches: set[str] = set()

        for stmt in _iter_non_nested_nodes(fn):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            value = stmt.value
            cache_name, is_cache_lookup = cache_lookup(value)
            if is_cache_lookup:
                for target in targets:
                    cache_hits.update(_target_names(target))
                    if cache_name:
                        cache_hits.add(cache_name)
                continue
            if tensor_output_expr(value, input_names, output_names):
                for target in targets:
                    output_names.update(_target_names(target))

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    cache_name = _ast_root_name(target.value)
                    if not _cache_name_allows_output_replay(cache_name):
                        continue
                    if tensor_output_expr(stmt.value, input_names, output_names):
                        stored_caches.add(cache_name or "")
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Subscript):
                cache_name = _ast_root_name(stmt.target.value)
                if _cache_name_allows_output_replay(cache_name) and tensor_output_expr(stmt.value, input_names, output_names):
                    stored_caches.add(cache_name or "")

        if not stored_caches:
            return None

        for stmt in _iter_non_nested_nodes(fn):
            if not isinstance(stmt, ast.If):
                continue
            for inner in stmt.body:
                if isinstance(inner, ast.Return) and inner.value is not None:
                    cache_name, is_cache_lookup = cache_lookup(inner.value)
                    if is_cache_lookup and (not cache_name or cache_name in stored_caches):
                        return cache_name or next(iter(stored_caches))
                    if _expr_names(inner.value) & cache_hits:
                        return next(iter(stored_caches))
        return None

    def helper_has_lru_tensor_return(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        has_lru = False
        for deco in fn.decorator_list:
            dotted = _ast_dotted_name(deco.func if isinstance(deco, ast.Call) else deco)
            if dotted and dotted.endswith("lru_cache"):
                has_lru = True
                break
        if not has_lru:
            return False
        input_names = _function_input_names(fn)
        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Return) and tensor_output_expr(stmt.value, input_names, set()):
                return True
        return False

    def returned_helper_calls(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        assigned: dict[str, str] = {}
        returned: set[str] = set()
        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                callee = stmt.value.func.id
                if callee in functions:
                    for target in stmt.targets:
                        for name in _target_names(target):
                            assigned[name] = callee
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                value = stmt.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in functions:
                    returned.add(value.func.id)
                elif isinstance(value, ast.Name) and value.id in assigned:
                    returned.add(assigned[value.id])
            elif isinstance(stmt, ast.If):
                for inner in stmt.body + stmt.orelse:
                    if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Name) and inner.value.id in assigned:
                        returned.add(assigned[inner.value.id])
        return returned

    pending: list[str] = []
    seen: set[str] = set()
    for entrypoint in _entrypoint_function_nodes(facts):
        pending.extend(sorted(returned_helper_calls(entrypoint)))

    while pending:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        fn = functions.get(name)
        if fn is None:
            continue
        cache_name = helper_has_cache_replay(fn)
        if cache_name is not None:
            return [{
                "pattern": "OUTPUT_REPLAY_CACHE",
                "severity": "critical",
                "evidence": f"{entrypoint_label(facts.entrypoint_name)} returns helper-level cached output from {name} via {cache_name}",
            }]
        if helper_has_lru_tensor_return(fn):
            return [{
                "pattern": "OUTPUT_REPLAY_CACHE",
                "severity": "critical",
                "evidence": f"{entrypoint_label(facts.entrypoint_name)} returns tensor output from lru_cache helper {name}",
            }]
        pending.extend(sorted(returned_helper_calls(fn)))

    return []


def detect_value_keyed_output_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect cached output replay guarded by full tensor-value equality.

    This catches helper-mediated replay patterns such as:
    cache[key] = (input.clone(), output.clone())
    if torch.equal(current_input, cached_input): return cached_output.clone()

    It intentionally requires both a stored input snapshot and an equality-guarded
    cached-output return so normal workspace, preprocess, and graph caches do not
    become hard replay findings.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None or facts.entrypoint_name is None:
        return []

    function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    entrypoint = function_defs.get(facts.entrypoint_name)
    if entrypoint is None:
        return []

    def _looks_cache_name(name: Optional[str]) -> bool:
        if not name:
            return False
        lowered = name.lower()
        return any(token in lowered for token in ("cache", "memo"))

    def _reachable_functions() -> set[str]:
        reachable = {facts.entrypoint_name}
        pending = [facts.entrypoint_name]
        while pending:
            current = pending.pop()
            fn = function_defs.get(current)
            if fn is None:
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                    continue
                callee = call.func.id
                if callee in function_defs and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        return reachable

    def _assign_parts(stmt: ast.AST) -> tuple[list[ast.AST], ast.AST | None]:
        if isinstance(stmt, ast.Assign):
            return list(stmt.targets), stmt.value
        if isinstance(stmt, ast.AnnAssign):
            return [stmt.target], stmt.value
        return [], None

    def _assignment_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
        return [
            node
            for node in ast.walk(fn)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]

    def _is_cache_lookup(value: ast.AST | None) -> bool:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            return value.func.attr == "get" and _looks_cache_name(_ast_root_name(value.func.value))
        if isinstance(value, ast.Subscript):
            return _looks_cache_name(_ast_root_name(value))
        return False

    def _is_cache_store_target(target: ast.AST | None) -> bool:
        return isinstance(target, ast.Subscript) and _looks_cache_name(_ast_root_name(target))

    def _clone_roots(expr: ast.AST | None) -> set[str]:
        roots: set[str] = set()
        if expr is None:
            return roots
        for call in ast.walk(expr):
            if not isinstance(call, ast.Call):
                continue
            if isinstance(call.func, ast.Attribute) and call.func.attr == "clone":
                root = _ast_root_name(call.func.value)
                if root:
                    roots.add(root)
        return roots

    def _is_tensor_equal_guard(
        expr: ast.AST | None,
        input_names: set[str],
        cached_input_names: set[str],
    ) -> bool:
        if expr is None:
            return False
        for call in ast.walk(expr):
            if not isinstance(call, ast.Call):
                continue
            args: list[ast.AST] = []
            dotted = _ast_dotted_name(call.func)
            if dotted == "torch.equal":
                args = list(call.args[:2])
            elif isinstance(call.func, ast.Attribute) and call.func.attr == "equal":
                args = [call.func.value, *call.args[:1]]
            if len(args) < 2:
                continue

            left_names = _expr_names(args[0])
            right_names = _expr_names(args[1])
            left_is_input = bool(left_names & input_names)
            right_is_input = bool(right_names & input_names)
            left_is_cached = bool(left_names & cached_input_names)
            right_is_cached = bool(right_names & cached_input_names)
            if (left_is_input and right_is_cached) or (right_is_input and left_is_cached):
                return True
        return False

    def _cache_store_has_input_snapshot_and_output(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        input_names: set[str],
    ) -> bool:
        input_snapshot_vars: set[str] = set()
        output_snapshot_vars: set[str] = set()
        computed_output_vars: set[str] = set()
        assignments = _assignment_nodes(fn)

        for stmt in assignments:
            targets, value = _assign_parts(stmt)
            if value is None:
                continue
            clone_roots = _clone_roots(value)
            assigned_names: set[str] = set()
            for target in targets:
                assigned_names.update(_target_names(target))
            if clone_roots & input_names:
                input_snapshot_vars.update(assigned_names)
            if any(root not in input_names for root in clone_roots):
                output_snapshot_vars.update(assigned_names)
            if isinstance(value, ast.Call) and not _is_cache_lookup(value):
                for assigned in assigned_names:
                    lowered = assigned.lower()
                    if not any(token in lowered for token in ("key", "cache", "cached", "shape")):
                        computed_output_vars.add(assigned)

        for stmt in assignments:
            targets, value = _assign_parts(stmt)
            if value is None or not any(_is_cache_store_target(target) for target in targets):
                continue
            value_names = _expr_names(value)
            clone_roots = _clone_roots(value)
            has_input_snapshot = bool(
                (clone_roots & input_names) or (value_names & input_snapshot_vars)
            )
            has_output_payload = bool(
                any(root not in input_names for root in clone_roots)
                or (value_names & output_snapshot_vars)
                or (value_names & computed_output_vars)
            )
            if has_input_snapshot and has_output_payload:
                return True
        return False

    def _cache_symbols(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[set[str], set[str], set[str]]:
        cache_hit_names: set[str] = set()
        cached_input_names: set[str] = set()
        cached_output_names: set[str] = set()

        for stmt in _assignment_nodes(fn):
            targets, value = _assign_parts(stmt)
            if value is None:
                continue
            assigned_names: set[str] = set()
            for target in targets:
                assigned_names.update(_target_names(target))

            if _is_cache_lookup(value):
                cache_hit_names.update(assigned_names)

            value_root = _ast_root_name(value)
            if value_root in cache_hit_names:
                for target in targets:
                    if isinstance(target, (ast.Tuple, ast.List)) and target.elts:
                        cached_input_names.update(_target_names(target.elts[0]))
                        for elt in target.elts[1:]:
                            cached_output_names.update(_target_names(elt))
                    else:
                        cache_hit_names.update(_target_names(target))
            elif value_root in cached_output_names:
                cached_output_names.update(assigned_names)

        return cache_hit_names, cached_input_names, cached_output_names

    def _has_guarded_cached_return(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        input_names: set[str],
        cached_input_names: set[str],
        cached_output_names: set[str],
    ) -> bool:
        if not cached_input_names or not cached_output_names:
            return False
        for if_node in ast.walk(fn):
            if not isinstance(if_node, ast.If):
                continue
            if not _is_tensor_equal_guard(if_node.test, input_names, cached_input_names):
                continue
            for stmt in if_node.body:
                for nested in ast.walk(stmt):
                    if not isinstance(nested, ast.Return) or nested.value is None:
                        continue
                    if _expr_names(nested.value) & cached_output_names:
                        return True
        return False

    for name in sorted(_reachable_functions()):
        fn = function_defs.get(name)
        if fn is None:
            continue
        input_names = {arg.arg for arg in fn.args.args}
        if not input_names:
            continue
        _, cached_input_names, cached_output_names = _cache_symbols(fn)
        if not _cache_store_has_input_snapshot_and_output(fn, input_names):
            continue
        if _has_guarded_cached_return(fn, input_names, cached_input_names, cached_output_names):
            return [{
                "pattern": "OUTPUT_REPLAY_CACHE",
                "severity": "critical",
                "evidence": (
                    "Value-keyed output replay in reachable helper "
                    f"{name}: stores cloned input/output and returns cached output "
                    "after torch.equal input check"
                ),
            }]

    return []


def detect_cuda_graph_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 7: CUDA graph replay via ctypes libcudart."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    matches = []
    signals = []
    has_ctypes = bool(RE_CTYPES_LIBCUDART.search(code))
    has_create = bool(RE_CUDA_GRAPH_CREATE.search(code))
    has_launch = bool(RE_CUDA_GRAPH_LAUNCH.search(code))
    has_instantiate = bool(RE_CUDA_GRAPH_INSTANTIATE.search(code))
    has_capture = bool(RE_CUDA_STREAM_CAPTURE.search(code))

    if has_ctypes:
        signals.append("ctypes.CDLL(libcudart)")
    if has_create:
        signals.append("cudaGraphCreate")
    if has_launch:
        signals.append("cudaGraphLaunch")
    if has_instantiate:
        signals.append("cudaGraphInstantiate")
    if has_capture:
        signals.append("cudaStreamBeginCapture")

    if signals:
        severity = "medium"
        reasons = []
        if has_launch and has_instantiate:
            severity = "high"
            reasons.append("instantiate+launch replay path")
        if has_ctypes and (has_launch or has_instantiate):
            severity = "high"
            reasons.append("ctypes graph launch path")
        evidence = "CUDA graph API: " + ", ".join(signals)
        if reasons:
            evidence += " (" + ", ".join(reasons) + ")"
        matches.append({
            "pattern": "CUDA_GRAPH_REPLAY",
            "severity": severity,
            "evidence": evidence,
        })
    return matches


def detect_silent_fallback(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 8: try/except falling back to reference matmul."""
    facts = ensure_submission_facts(code)
    code = facts.python_active
    matches = []
    if RE_TORCH_MATMUL_FALLBACK.search(code):
        matches.append({
            "pattern": "SILENT_FALLBACK",
            "severity": "medium",
            "evidence": "try/except fallback to torch.matmul/mm/bmm",
        })
    return matches


def detect_trivial_probe(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 9: Trivial/probe submission (empty return, debug only)."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    matches = []
    lines = [l.strip() for l in code.strip().splitlines() if l.strip() and not l.strip().startswith('#')]
    if len(lines) < 35:
        # Check for return [] or return None without real compute
        has_return_empty = bool(RE_RETURN_EMPTY.search(code))
        has_return_none = bool(RE_RETURN_NONE.search(code))
        # No tensor operations
        has_compute = any(kw in code for kw in [
            'torch.mm', 'torch.bmm', 'torch.matmul', '_scaled_mm',
            'triton', 'cute.kernel', '__global__', 'load_inline',
            'cutlass', 'tl.load', 'tl.store',
        ])
        if (has_return_empty or has_return_none) and not has_compute:
            matches.append({
                "pattern": "TRIVIAL_PROBE",
                "severity": "high",
                "evidence": f"Trivial submission ({len(lines)} non-empty lines, returns empty/None)",
            })
    return matches


def _input_float_return_from_body(body: list[ast.stmt], input_names: set[str]) -> bool:
    aliases: set[str] = set()
    saw_return = False
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if _is_input_float_call(stmt.value, input_names):
                aliases.add(stmt.targets[0].id)
                continue
        if (
            isinstance(stmt, ast.With)
            and stmt.items
            and all(_is_nullcontext_call(item.context_expr) for item in stmt.items)
        ):
            if _input_float_return_from_body(stmt.body, input_names):
                saw_return = True
                continue
            return False
        if isinstance(stmt, ast.Return):
            if _is_input_float_call(stmt.value, input_names):
                saw_return = True
                continue
            if isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                saw_return = True
                continue
            if isinstance(stmt.value, ast.IfExp):
                branches = [stmt.value.body, stmt.value.orelse]
                if all(
                    _is_input_float_call(branch, input_names)
                    or (isinstance(branch, ast.Name) and branch.id in aliases)
                    for branch in branches
                ):
                    saw_return = True
                    continue
            return False
        return False
    return saw_return


def _none_guarded_return(stmt: ast.stmt, names: set[str]) -> bool:
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id in names
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and len(test.comparators) == 1
        and _expr_is_none(test.comparators[0])
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.Return)
        and isinstance(stmt.body[0].value, ast.Name)
        and stmt.body[0].value.id == test.left.id
        and not stmt.orelse
    ):
        return False
    return not _body_has_calls(stmt.body)


def _optional_cache_input_float_return(
    body: list[ast.stmt],
    data_names: set[str],
    cache_names: set[str],
) -> bool:
    aliases: set[str] = set()
    saw_guard = False
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if not saw_guard and _none_guarded_return(stmt, cache_names):
            saw_guard = True
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if _is_input_float_call(stmt.value, data_names):
                aliases.add(stmt.targets[0].id)
                continue
        if isinstance(stmt, ast.Return):
            return saw_guard and (
                _is_input_float_call(stmt.value, data_names)
                or (isinstance(stmt.value, ast.Name) and stmt.value.id in aliases)
            )
        return False
    return False


def _global_ifexp_input_float_return(
    body: list[ast.stmt],
    input_names: set[str],
    global_names: set[str],
    none_inited: set[str],
) -> bool:
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass, ast.Global)):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.IfExp):
            return False
        expr = stmt.value
        if not (
            isinstance(expr.body, ast.Name)
            and expr.body.id in global_names
            and expr.body.id in none_inited
            and _is_input_float_call(expr.orelse, input_names)
        ):
            return False
        test = expr.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == expr.body.id
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and _expr_is_none(test.comparators[0])
        )
    return False


def _nonlocal_input_float_state(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    local_none: set[str],
) -> Optional[str]:
    input_names = _function_input_names(fn)
    nonlocal_names = {
        name
        for stmt in fn.body
        if isinstance(stmt, ast.Nonlocal)
        for name in stmt.names
    }
    state_names = nonlocal_names & local_none
    for state_name in state_names:
        has_guard = False
        has_store = False
        has_return = False
        for stmt in fn.body:
            if isinstance(stmt, ast.Nonlocal):
                continue
            if _none_guarded_return(stmt, {state_name}):
                has_guard = True
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id == state_name
                    and _is_input_float_call(stmt.value, input_names)
                ):
                    has_store = True
                    continue
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name) and stmt.value.id == state_name:
                has_return = True
                continue
            return None
        if has_guard and has_store and has_return:
            return state_name
    return None


def _is_nullcontext_call(expr: ast.AST | None) -> bool:
    return (
        isinstance(expr, ast.Call)
        and (
            (isinstance(expr.func, ast.Name) and expr.func.id == "nullcontext")
            or (
                isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "nullcontext"
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == "contextlib"
            )
        )
    )


def detect_input_passthrough_output(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: entrypoint fabricates output by returning input.float()."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)
    function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        if input_names and _input_float_return_from_body(fn.body, input_names):
            return [{
                "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                "severity": "critical",
                "evidence": f"{entrypoint_name} returns an input tensor cast to float without compute",
            }]

        positional_args = list(fn.args.posonlyargs) + list(fn.args.args)
        if fn.name in _ENTRYPOINT_METHOD_NAMES and positional_args and positional_args[0].arg in {"self", "cls"}:
            positional_args = positional_args[1:]
        default_args = positional_args[-len(fn.args.defaults):] if fn.args.defaults else []
        cache_names = {
            arg.arg
            for arg, default in zip(default_args, fn.args.defaults)
            if _expr_is_none(default)
        } & input_names
        data_names = input_names - cache_names
        if cache_names and data_names and _optional_cache_input_float_return(fn.body, data_names, cache_names):
            return [{
                "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                "severity": "critical",
                "evidence": f"{entrypoint_name} returns optional cached output or input.float()",
            }]

        global_names = {
            name
            for stmt in fn.body
            if isinstance(stmt, ast.Global)
            for name in stmt.names
        }
        if global_names and _global_ifexp_input_float_return(
            fn.body, input_names, global_names, facts._none_inited
        ):
            return [{
                "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                "severity": "critical",
                "evidence": f"{entrypoint_name} returns global sentinel output or input.float()",
            }]

        local_none = {
            target.id
            for stmt in fn.body
            if isinstance(stmt, ast.Assign)
            and _expr_is_none(stmt.value)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        if local_none:
            nonlocal_helpers = {
                stmt.name
                for stmt in fn.body
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _nonlocal_input_float_state(stmt, local_none) is not None
            }
            for stmt in fn.body:
                if (
                    isinstance(stmt, ast.Return)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in nonlocal_helpers
                    and len(stmt.value.args) == 1
                    and isinstance(stmt.value.args[0], ast.Name)
                    and stmt.value.args[0].id in input_names
                    and not stmt.value.keywords
                ):
                    return [{
                        "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} delegates to nonlocal state storing input.float()",
                    }]

        none_slots = {
            target.id
            for stmt in fn.body
            if isinstance(stmt, ast.Assign)
            and _expr_is_none(stmt.value)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        for stmt in fn.body:
            if not (
                isinstance(stmt, ast.If)
                and isinstance(stmt.test, ast.Compare)
                and isinstance(stmt.test.left, ast.Name)
                and stmt.test.left.id in none_slots
                and len(stmt.test.ops) == 1
                and len(stmt.test.comparators) == 1
            ):
                continue
            slot = stmt.test.left.id
            assigned_float = any(
                isinstance(inner, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == slot for t in inner.targets)
                and _is_input_float_call(inner.value, input_names)
                for inner in stmt.body + stmt.orelse
            )
            returned_slot = any(
                isinstance(inner, ast.Return)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == slot
                for inner in stmt.body + stmt.orelse
            )
            if assigned_float and returned_slot:
                return [{
                    "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                    "severity": "critical",
                    "evidence": f"{entrypoint_name} lazily returns cached input.float()",
                }]

    return []


def detect_indirect_input_passthrough_output(code: str | SubmissionFacts) -> list[dict]:
    """Detect indirect entrypoints that still only return input.float()."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        stmt.name: stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes: dict[str, ast.ClassDef] = {
        stmt.name: stmt for stmt in tree.body if isinstance(stmt, ast.ClassDef)
    }

    def function_returns_input_float(name: str | None) -> bool:
        fn = functions.get(name or "")
        return fn is not None and _input_float_return_from_body(fn.body, _function_input_names(fn))

    def callable_value_returns_input_float(expr: ast.AST | None) -> bool:
        if _lambda_returns_input_float(expr):
            return True
        if isinstance(expr, ast.Name):
            return function_returns_input_float(expr.id)
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in {"staticmethod", "classmethod"}
            and expr.args
        ):
            return callable_value_returns_input_float(expr.args[0])
        return False

    def dynamic_type_returns_input_float(expr: ast.AST | None) -> bool:
        if not (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "type"
            and len(expr.args) >= 3
            and isinstance(expr.args[2], ast.Dict)
        ):
            return False
        for key, value in zip(expr.args[2].keys, expr.args[2].values):
            if _static_string(key) == "__call__":
                return callable_value_returns_input_float(value)
        return False

    # K = type("K", (), {"__call__": lambda self, x: x.float()}); custom_kernel = K()
    dynamic_classes: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not dynamic_type_returns_input_float(stmt.value):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                dynamic_classes.add(target.id)
    if dynamic_classes:
        dynamic_instances: set[str] = set()
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id in dynamic_classes
                and not stmt.value.args
                and not stmt.value.keywords
            ):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        dynamic_instances.add(target.id)
                        if is_entrypoint_name(target.id):
                            return [{
                                "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                                "severity": "critical",
                                "evidence": f"{entrypoint_name} is a dynamic type() callable returning input.float()",
                            }]
            if any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
                if isinstance(stmt.value, ast.Name) and stmt.value.id in dynamic_instances:
                    return [{
                        "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                        "severity": "critical",
                    "evidence": f"{entrypoint_name} aliases a dynamic type() callable returning input.float()",
                }]

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
            continue
        if (
            isinstance(stmt.value, ast.Call)
            and not stmt.value.args
            and not stmt.value.keywords
            and dynamic_type_returns_input_float(stmt.value.func)
        ):
            return [{
                "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                "severity": "critical",
                "evidence": f"{entrypoint_name} directly instantiates a dynamic type() callable returning input.float()",
            }]

    # Base.__init_subclass__ installs Child.custom_kernel = lambda x: x.float()
    subclass_export_bases: set[str] = set()
    for cls in classes.values():
        for item in cls.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name != "__init_subclass__":
                continue
            cls_arg = item.args.args[0].arg if item.args.args else "cls"
            for stmt in _iter_non_nested_nodes(item):
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr in ENTRYPOINT_NAMES
                        and isinstance(target.value, ast.Name)
                        and target.value.id == cls_arg
                        and callable_value_returns_input_float(stmt.value)
                    ):
                        subclass_export_bases.add(cls.name)
    if subclass_export_bases:
        subclass_exports = {
            cls.name
            for cls in classes.values()
            if any(isinstance(base, ast.Name) and base.id in subclass_export_bases for base in cls.bases)
        }
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
                continue
            if (
                isinstance(stmt.value, ast.Attribute)
                and stmt.value.attr in ENTRYPOINT_NAMES
                and isinstance(stmt.value.value, ast.Name)
                and stmt.value.value.id in subclass_exports
            ):
                return [{
                    "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                    "severity": "critical",
                    "evidence": f"{entrypoint_name} is installed by __init_subclass__ as input.float()",
                }]

    # class D: def __get__(self, obj, cls): return obj.x.float()
    # class W: out = D(); def __init__(self, x): self.x = x
    # def custom_kernel(x): return W(x).out
    descriptor_slots: dict[str, str] = {}
    for cls in classes.values():
        get_method = _method_from_class(cls, ("__get__",))
        if get_method is None or len(get_method.args.args) < 2:
            continue
        obj_name = get_method.args.args[1].arg
        for stmt in get_method.body:
            if isinstance(stmt, ast.Return):
                slot = _is_input_attr_float_call(stmt.value, obj_name)
                if slot:
                    descriptor_slots[cls.name] = slot
                    break

    if descriptor_slots:
        wrapper_descriptor_slots: dict[tuple[str, str], str] = {}
        wrapper_init_inputs: dict[str, set[str]] = {}
        for cls in classes.values():
            attr_slots: dict[str, str] = {}
            for stmt in cls.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if (
                    isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in descriptor_slots
                ):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            attr_slots[target.id] = descriptor_slots[stmt.value.func.id]
            if not attr_slots:
                continue
            init_method = _method_from_class(cls, ("__init__",))
            if init_method is None or len(init_method.args.args) < 2:
                continue
            self_name = init_method.args.args[0].arg
            assigned_slots: set[str] = set()
            for stmt in _iter_non_nested_nodes(init_method):
                if not isinstance(stmt, ast.Assign):
                    continue
                if not (_expr_names(stmt.value) & _function_input_names(init_method)):
                    continue
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == self_name
                    ):
                        assigned_slots.add(target.attr)
            wrapper_init_inputs[cls.name] = assigned_slots
            for public_attr, source_slot in attr_slots.items():
                if source_slot in assigned_slots:
                    wrapper_descriptor_slots[(cls.name, public_attr)] = source_slot

        for fn in _entrypoint_function_nodes(facts):
            input_names = _function_input_names(fn)
            for stmt in _iter_non_nested_nodes(fn):
                if not isinstance(stmt, ast.Return):
                    continue
                value = stmt.value
                if not (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Call)
                    and isinstance(value.value.func, ast.Name)
                    and value.value.args
                    and isinstance(value.value.args[0], ast.Name)
                    and value.value.args[0].id in input_names
                    and (value.value.func.id, value.attr) in wrapper_descriptor_slots
                ):
                    continue
                return [{
                    "pattern": "INPUT_PASSTHROUGH_OUTPUT",
                    "severity": "critical",
                    "evidence": f"{entrypoint_name} returns descriptor-backed input.float()",
                }]

    return []


def detect_input_reduction_output(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: entrypoint returns torch.any/all(input) as fake output."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    torch_aliases, imported = _torch_alias_sets(facts)
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    def reduction_input(expr: ast.AST | None, input_names: set[str]) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        if _torch_call_name(expr, torch_aliases, imported) not in {"any", "all"}:
            return False
        return (
            len(expr.args) == 1
            and isinstance(expr.args[0], ast.Name)
            and expr.args[0].id in input_names
            and not expr.keywords
        )

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        aliases: set[str] = set()
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                if reduction_input(stmt.value, input_names):
                    aliases.add(stmt.targets[0].id)
                    continue
            if isinstance(stmt, ast.Return):
                if reduction_input(stmt.value, input_names):
                    return [{
                        "pattern": "INPUT_REDUCTION_OUTPUT",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns torch.any/all(input) as output",
                    }]
                if isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                    return [{
                        "pattern": "INPUT_REDUCTION_OUTPUT",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns torch.any/all(input) alias as output",
                    }]
    return []


def detect_self_matmul_output(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: entrypoint returns an input multiplied by itself as fake output."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    torch_aliases, imported = _torch_alias_sets(facts)
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    def is_self_product(expr: ast.AST | None, input_names: set[str], aliases: set[str]) -> bool:
        names = input_names | aliases
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.MatMult):
            if isinstance(expr.left, ast.Name) and expr.left.id in names:
                return _is_transpose_of(expr.right, expr.left.id) or (
                    isinstance(expr.right, ast.Name) and expr.right.id == expr.left.id
                )
        if not isinstance(expr, ast.Call):
            return False
        call_name = _torch_call_name(expr, torch_aliases, imported)
        if call_name in {"mm", "matmul"} and len(expr.args) >= 2:
            first, second = expr.args[:2]
            if isinstance(first, ast.Name) and first.id in names:
                return _is_transpose_of(second, first.id) or (
                    isinstance(second, ast.Name) and second.id == first.id
                )
        if call_name == "bmm" and len(expr.args) >= 2:
            first, second = expr.args[:2]
            if isinstance(first, ast.Name) and first.id in aliases:
                return _is_transpose_of(second, first.id)
        return False

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        aliases: set[str] = set()
        result_aliases: set[str] = set()
        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target = stmt.targets[0].id
                if (
                    isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "unsqueeze"
                    and isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.value.id in input_names
                ):
                    aliases.add(target)
                if is_self_product(stmt.value, input_names, aliases):
                    result_aliases.add(target)
            elif isinstance(stmt, ast.Return):
                if is_self_product(stmt.value, input_names, aliases):
                    return [{
                        "pattern": "SELF_MATMUL_OUTPUT",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns input self-matmul output",
                    }]
                if isinstance(stmt.value, ast.Name) and stmt.value.id in result_aliases:
                    return [{
                        "pattern": "SELF_MATMUL_OUTPUT",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns input self-matmul alias",
                    }]
    return []


def detect_torch_compile_cache(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 11: torch.compile for pipeline graph caching."""
    facts = ensure_submission_facts(code)
    code = facts.python_only
    matches = []
    if RE_TORCH_COMPILE.search(code):
        matches.append({
            "pattern": "TORCH_COMPILE_CACHE",
            "severity": "low",
            "evidence": "torch.compile() used",
        })
    return matches


def detect_cuda_graph_python(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 12: Python CUDAGraph caching with data_ptr keys + .replay()."""
    facts = ensure_submission_facts(code)
    python_only = facts.python_only
    matches = []
    signals = []

    has_graph = bool(RE_CUDA_GRAPH_PYTHON.search(python_only))
    has_replay = bool(RE_CUDA_GRAPH_REPLAY.search(python_only))
    has_cache_key = bool(RE_CUDA_GRAPH_CACHE_KEY.search(python_only))
    has_graph_cache = bool(RE_GRAPH_CACHE_NAME.search(python_only))

    if has_graph:
        signals.append("torch.cuda.CUDAGraph()")
    if has_replay:
        signals.append(".replay()")
    if has_cache_key:
        signals.append("data_ptr() as graph cache key")
    if has_graph_cache:
        signals.append("graph cache dict")

    # Need at least CUDAGraph + replay or CUDAGraph + cache key.
    # CUDA graph replay is a legitimate optimization (reduces kernel launch
    # overhead).  Keep at medium — exploits using CUDA graphs would also
    # have MODULE_MUTATION / GLOBALS_MUTATION which are critical/high.
    if has_graph and (has_replay or has_cache_key):
        matches.append({
            "pattern": "CUDA_GRAPH_PYTHON",
            "severity": "medium",
            "evidence": "Python CUDAGraph caching: " + ", ".join(signals),
        })
    return matches


def detect_hardcoded_shapes(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 13: Hardcoded benchmark shapes in KERNEL_CONFIGS or shape-gated branches.

    Checks both Python and C++ inline code, including macro-based dispatch.
    """
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    python_only = facts.python_only
    matches = []
    signals = []

    # Python checks
    has_configs = bool(RE_KERNEL_CONFIGS_DICT.search(python_only))
    shape_tuples = RE_SHAPE_TUPLE_KEY.findall(python_only)
    has_shape_gate = bool(RE_SHAPE_IF_GATE.search(python_only))

    if has_configs and len(shape_tuples) >= 3:
        signals.append(f"KERNEL_CONFIGS dict with {len(shape_tuples)} shape keys")
    if has_shape_gate and len(shape_tuples) >= 3:
        signals.append(f"Python shape-gated branches ({len(shape_tuples)} shape tuples)")

    # Python shape-gated if statements: `if m == 256 and n == 4096` or `if self.m == 256 and self.n == 4096`
    py_shape_ifs = re.findall(
        r'if\s+(?:self\.)?[a-zA-Z_]\w*\s*==\s*\d{3,5}\s+and\s+(?:self\.)?[a-zA-Z_]\w*\s*==\s*\d{3,5}',
        python_only,
    )
    if len(py_shape_ifs) >= 2:
        signals.append(f"Python shape-gated if-statements ({len(py_shape_ifs)} branches)")

    # C++ checks: `if (K == 7168)` or `if (G==8 && N0==7168 && K0==2048)`
    cpp_shape_ifs = re.findall(
        r'if\s*\([^)]*(?:M|N|K|L|G|N0|K0|L0|M0)\s*==\s*\d{3,5}[^)]*\)',
        code,
    )
    if len(cpp_shape_ifs) >= 2:
        signals.append(f"C++ shape-gated dispatch ({len(cpp_shape_ifs)} branches)")

    # C++ macro-based shape dispatch: LAUNCH(7168, ...) or similar
    macro_shape_dispatch = re.findall(
        r'(?:LAUNCH|DISPATCH|KERNEL_CASE|SPECIALIZE)\s*\(\s*\d{3,5}',
        code,
    )
    if len(macro_shape_dispatch) >= 3:
        signals.append(f"C++ macro-based shape dispatch ({len(macro_shape_dispatch)} calls)")

    # Shape-conditional Python parameter tuning: if m == 256, if self.m == 512
    py_single_shape_checks = re.findall(
        r'(?:self\.)?(?:m|n|k|M|N|K)\s*==\s*\d{3,5}',
        python_only,
    )
    if len(py_single_shape_checks) >= 3:
        signals.append(f"Python shape dimension checks ({len(py_single_shape_checks)})")
    elif len(py_single_shape_checks) >= 2 and py_shape_ifs:
        signals.append(f"Python shape dimension checks ({len(py_single_shape_checks)} + {len(py_shape_ifs)} combined)")

    if signals:
        # Shape specialization is a weak signal (legitimate kernels tune per
        # shape). Keep at low — KERNEL_CONFIGS alone does not indicate hacking.
        severity = "low"
        matches.append({
            "pattern": "HARDCODED_SHAPES",
            "severity": severity,
            "evidence": "; ".join(signals),
        })
    return matches


def detect_benchmark_unsafe_algo_dispatch(code: str | SubmissionFacts) -> list[dict]:
    """Shape/data-family dispatch into unsafe specialized algorithms.

    Per-shape tuning is legitimate. This detector requires the stronger
    combination: benchmark-shaped branches plus data-family classifiers and
    fast-path terms that imply reduced precision, truncated work, or a generic
    exact fallback for cases outside the public fixture.
    """
    facts = ensure_submission_facts(code)
    raw_code = facts.raw_code
    active_code = facts.python_active
    combined = f"{active_code}\n{raw_code}"
    low = combined.lower()
    matches = []

    shape_gate_count = len(re.findall(
        r'(?:\(\s*[Bb]\s*,\s*[Nn]\s*\)\s*==\s*\(\s*\d{1,5}\s*,\s*\d{1,5}\s*\)|'
        r'\b(?:B|b|batch|N|n)\s*(?:==|!=|<=|>=|<|>)\s*\d{1,5})',
        combined,
    ))
    benchmark_constant_count = len(re.findall(
        r'\b(?:60|176|352|512|640|768|1024|2048|4096|1280)\b',
        combined,
    ))
    family_terms = re.findall(
        r'\b(?:classify_512|classify_1024|detect_label|rankdef|rank_def|clustered|'
        r'nearrank|near_rank|nearcol|near_col|mixed|rowscale|colnorm|zerofrac|'
        r'colrange|tail_max|neardiff|inactive_tail)\b',
        low,
    )
    unsafe_terms = re.findall(
        r'\b(?:zero_tail|trunc|truncate|stop_at|switch_at|ncap|ncols|limit|'
        r'_trunc_stop|tf32|fp16|bf16|fast_path|unsafe|rankdef)\b',
        low,
    )
    fallback_terms = re.findall(
        r'\b(?:torch\.geqrf|torch\.linalg\.(?:qr|lu)|fallback|fall back|exact|generic)\b',
        combined,
        re.IGNORECASE,
    )
    raw_native_cache = (
        bool(re.search(r'\bnvcc\b', low))
        and bool(re.search(r'\b(?:ctypes|CDLL|subprocess\.run|Popen|\.so|\.cu)\b', combined))
        and not bool(re.search(r'\bload_inline\s*\(', active_code))
        and len(re.findall(r'\b(?:data_ptr|ctypes\.c_void_p|uint64_t|void\s*\*)\b', combined)) >= 4
    )

    strong_shape_fixture = shape_gate_count >= 8 and benchmark_constant_count >= 20
    data_family_routing = len(family_terms) >= 4 and len(unsafe_terms) >= 4
    exact_escape_hatch = len(fallback_terms) >= 2

    if raw_native_cache and strong_shape_fixture and data_family_routing:
        signals = [
            "raw native compile/load cache",
            f"{shape_gate_count} shape gates",
            f"{len(set(family_terms))} data-family terms",
            f"{len(set(unsafe_terms))} unsafe/truncation terms",
        ]
        if exact_escape_hatch:
            signals.append("exact/generic fallback path")
        matches.append({
            "pattern": "BENCHMARK_UNSAFE_ALGO_DISPATCH",
            "severity": "high",
            "evidence": "; ".join(signals),
        })

    return matches


def detect_input_structure_truncation_dispatch(code: str | SubmissionFacts) -> list[dict]:
    """Input probes routing to partial computation or exact subset repair.

    This is not a ban on algorithm selection. It targets classifiers/probes of
    input structure that steer only a detected subset into truncated fast paths,
    tail-zeroing, or reference/exact repair, which is the static shape of a
    public-fixture assumption.
    """
    facts = ensure_submission_facts(code)
    active_code = facts.python_active
    raw_code = facts.raw_code
    combined = f"{active_code}\n{raw_code}"
    low = combined.lower()
    matches = []

    probe_terms = re.findall(
        r'\b(?:probe_cond|probe_n|_probe|_probe_conditioning|_s3_collinear|'
        r'spread_thresh|sparse_thresh|corr_thresh|bit0|bit1|classify_512|'
        r'classify_1024|detect_label|colnorm|neardiff|zerofrac|tail_max)\b',
        low,
    )
    subset_terms = re.findall(
        r'\b(?:bad_idx|good_idx|bad_count|good_count|index_select|index_copy|'
        r'scatter_exact|compact_label|panel_labels|label_bad_cache|gather_.*bad|'
        r'compact_.*bad)\b',
        low,
    )
    partial_terms = re.findall(
        r'(?:tau\s*\[[^\]]*limit\s*:\s*\]\s*=\s*0|H\s*\[[^\]]*limit\s*:\s*\]\s*=\s*0|'
        r'\b(?:zero_tail|trunc|truncate|inactive_tail|stop_at|switch_at|ncap|ncols|limit\s*<\s*n)\b)',
        combined,
        re.IGNORECASE,
    )
    repair_terms = re.findall(
        r'\b(?:repair|exact|torch\.geqrf|torch\.linalg\.(?:qr|lu)|index_copy|index_select)\b',
        combined,
        re.IGNORECASE,
    )
    probe_set = set(probe_terms)
    subset_set = set(subset_terms)
    partial_set = set(partial_terms)
    repair_set = set(repair_terms)

    probe_driven_repair = (
        bool(probe_set & {"probe_cond", "probe_n", "_probe", "_s3_collinear"})
        and len(probe_set) >= 4
        and len(subset_set) >= 3
        and len(repair_set) >= 3
    )
    classifier_driven_partial = (
        bool(probe_set & {"classify_512", "classify_1024", "detect_label"})
        and len(probe_set) >= 4
        and len(partial_set) >= 3
        and len(subset_set) >= 2
    )

    if probe_driven_repair or classifier_driven_partial:
        signals = [
            f"{len(probe_set)} structure/probe terms",
            f"{len(subset_set)} subset-routing terms",
        ]
        if partial_set:
            signals.append(f"{len(partial_set)} partial/truncation terms")
        if repair_set:
            signals.append(f"{len(repair_set)} exact-repair terms")
        matches.append({
            "pattern": "INPUT_STRUCTURE_TRUNCATION_DISPATCH",
            "severity": "high",
            "evidence": "; ".join(signals),
        })

    return matches


def detect_unsync_multistream(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 14: Unsynchronized multi-stream dispatch to exploit timing.

    Checks both Python code AND C++ inline code, since getStreamFromPool
    is often called from C++/CUDA inline sources.
    """
    # Check Python code
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    python_only = facts.python_only
    py_stream_creates = len(RE_GET_STREAM_FROM_POOL.findall(python_only))
    py_sync_calls = (
        len(RE_NO_SYNC_STREAM.findall(python_only)) +
        len(RE_STREAM_WAIT_EVENT.findall(python_only)) +
        len(RE_STREAM_WAIT_STREAM.findall(python_only)) +
        len(RE_TORCH_CUDA_SYNCHRONIZE.findall(python_only))
    )

    candidates = []
    if py_stream_creates >= 2:
        if py_sync_calls == 0:
            candidates.append((
                "high",
                f"Python: {py_stream_creates} stream creates, no sync/wait primitives",
            ))
        elif py_sync_calls * 2 < py_stream_creates:
            # Significantly under-synchronized: fewer than half as many syncs as streams
            candidates.append((
                "medium",
                f"Python: {py_stream_creates} stream creates, only {py_sync_calls} sync/wait calls",
            ))

    # Also check C++ inline code for getStreamFromPool (c10::cuda::getStreamFromPool)
    cpp_stream_pattern = re.compile(r'getStreamFromPool\s*\(')
    cpp_stream_creates = len(cpp_stream_pattern.findall(code))
    cpp_sync_calls = len(RE_CPP_STREAM_SYNC.findall(code)) + len(RE_CPP_METHOD_SYNC.findall(code))

    if cpp_stream_creates >= 2:
        if cpp_sync_calls == 0:
            candidates.append((
                "high",
                f"C++: {cpp_stream_creates} getStreamFromPool calls, no sync/wait primitives",
            ))
        elif cpp_sync_calls * 2 < cpp_stream_creates:
            candidates.append((
                "medium",
                f"C++: {cpp_stream_creates} getStreamFromPool calls, only {cpp_sync_calls} sync/wait calls",
            ))

    if not candidates:
        return []

    severity_rank = {"high": 2, "medium": 1}
    top_severity = max(candidates, key=lambda c: severity_rank[c[0]])[0]
    evidence = " | ".join(msg for sev, msg in candidates if sev == top_severity)
    return [{
        "pattern": "UNSYNC_MULTISTREAM",
        "severity": top_severity,
        "evidence": evidence,
    }]


def detect_cuda_event_disable_timing(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 15: cudaEventDisableTiming to mess with benchmark timing."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    matches = []
    if RE_CUDA_EVENT_DISABLE_TIMING.search(code):
        matches.append({
            "pattern": "CUDA_EVENT_DISABLE_TIMING",
            "severity": "medium",
            "evidence": "cudaEventDisableTiming or disable_timing=True flag",
        })
    return matches


# C preprocessor token-paste obfuscation of CUDA API names.
# Detects patterns like:
#   #define CAT(a,b) a##b
#   #define Q NVFP4_CAT2(cudaSt, reamCreateWithFlags)
# where the author explicitly splits CUDA runtime identifiers to evade
# string-based detectors.
RE_TOKEN_PASTE_DEF = re.compile(
    r'#\s*define\s+\w+\s*\([^)]*\)\s*\w+\s*##\s*\w+'
)
RE_SPLIT_CUDA_RUNTIME = re.compile(
    r'\b\w+\(\s*cuda(?:St|Ev|Me|La|De|Dr|Gr)'
    r'\s*,\s*(?:ream|ent|mory|unch|vice|iver|aph)'
    r'(?:_t|CreateWithFlags|WaitEvent|NonBlocking|Record|'
    r'Synchronize|Destroy|Launch|GetLastError|Alloc|Free)\s*\)',
    re.IGNORECASE,
)


def detect_token_paste_cuda_api(code: str | SubmissionFacts) -> list[dict]:
    """Detect C preprocessor token-paste obfuscation of CUDA API names.

    This is explicit filter evasion: the author splits CUDA runtime
    identifiers (e.g. cudaStreamCreateWithFlags → cudaSt + reamCreateWithFlags)
    using ## token-pasting to bypass string-based detectors.

    Confirmed 3,700x enrichment on nvfp4_group_gemm (17/26 hacked vs 2/8749 legit).
    """
    facts = ensure_submission_facts(code)
    raw = facts.raw_code
    if not RE_TOKEN_PASTE_DEF.search(raw):
        return []
    if not RE_SPLIT_CUDA_RUNTIME.search(raw):
        return []
    return [{
        "pattern": "TOKEN_PASTE_CUDA_API",
        "severity": "critical",
        "evidence": (
            "C preprocessor ## token-paste reconstructs CUDA runtime API names "
            "(explicit filter evasion)"
        ),
    }]


# Sequence-aware batch graph exploit: learns repeated call sequences
# and builds a batch CUDA graph that launches all calls at once.
RE_CALL_SEQUENCE = re.compile(r'\b_?call_sequence\b|\b_?known_seq\b|\bbatch_graph\b|\bbatch_handle\b')
RE_CTYPES_CUDART = re.compile(r'ctypes\.CDLL\s*\(\s*["\'].*libcudart|ctypes\.CDLL\s*\(\s*\w+\.__file__')
RE_NATIVE_GRAPH_LAUNCH = re.compile(r'\bcudaGraphLaunch\b|\binstant_launch\b|\bm_changed_launch\b')


def detect_sequence_batch_graph(code: str | SubmissionFacts) -> list[dict]:
    """Detect sequence-learning graph replay exploits.

    Pattern: submission learns the benchmark's call sequence, then builds
    a batch CUDA graph that executes all calls in the sequence at once,
    returning cached outputs for subsequent calls without per-call launches.

    Also detects native graph launch via ctypes (bypassing Python overhead).
    """
    facts = ensure_submission_facts(code)
    raw = facts.raw_code
    matches = []

    # Sequence-batch graph
    if RE_CALL_SEQUENCE.search(raw) and RE_NATIVE_GRAPH_LAUNCH.search(raw):
        matches.append({
            "pattern": "SEQUENCE_BATCH_GRAPH",
            "severity": "high",
            "evidence": (
                "Learns benchmark call sequence and builds batch CUDA graph "
                "(benchmark-order exploitation)"
            ),
        })

    # Partial graph key with native launch (weak identity cache)
    if RE_CTYPES_CUDART.search(raw) and RE_NATIVE_GRAPH_LAUNCH.search(raw):
        if not matches:  # don't double-fire
            matches.append({
                "pattern": "PARTIAL_GRAPH_KEY",
                "severity": "high",
                "evidence": (
                    "Native CUDA graph launch via ctypes with partial-identity "
                    "cache key (bypasses Python overhead, weak cache identity)"
                ),
            })

    return matches


# Runtime package installation: pip install inside submission code.
RE_RUNTIME_PIP_INSTALL = re.compile(
    r'(?:subprocess|os\.system|Popen)\s*\([^)]*pip\s+install'
    r'|check_call\s*\([^)]*pip.*install'
    r'|check_output\s*\([^)]*pip.*install',
    re.DOTALL,
)


def detect_runtime_package_install(code: str | SubmissionFacts) -> list[dict]:
    """Detect runtime package installation (sandbox violation).

    Submissions should not install packages during benchmark execution.
    This is a runner attack surface, not necessarily the exploit mechanism,
    but it should be blocked.
    """
    facts = ensure_submission_facts(code)
    raw = facts.raw_code
    if RE_RUNTIME_PIP_INSTALL.search(raw):
        return [{
            "pattern": "RUNTIME_PACKAGE_INSTALL",
            "severity": "critical",
            "evidence": "Runtime pip install inside submission code (sandbox violation)",
        }]
    tree = facts.ast_tree
    if tree is not None:
        os_aliases = {"os"}
        subprocess_aliases = {"subprocess"}
        socket_aliases = {"socket"}
        imported_calls: dict[str, str] = {}
        for node in facts._imports:
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
                elif alias.name == "subprocess":
                    subprocess_aliases.add(alias.asname or alias.name)
                elif alias.name == "socket":
                    socket_aliases.add(alias.asname or alias.name)
        for node in facts._import_froms:
            if node.module == "subprocess":
                for alias in node.names:
                    if alias.name in {"run", "call", "check_call", "check_output", "Popen"}:
                        imported_calls[alias.asname or alias.name] = f"subprocess.{alias.name}"
            elif node.module == "socket":
                for alias in node.names:
                    if alias.name in {"socket", "create_connection"}:
                        imported_calls[alias.asname or alias.name] = f"socket.{alias.name}"

        def risky_process_call(node: ast.Call) -> bool:
            if any(kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords):
                return True
            static_parts = []
            for arg in node.args[:2]:
                if (value := _static_string(arg)) is not None:
                    static_parts.append(value)
                elif isinstance(arg, (ast.List, ast.Tuple)):
                    for elt in arg.elts:
                        if (value := _static_string(elt)) is not None:
                            static_parts.append(value)
            command = " ".join(static_parts).lower()
            return bool(re.search(r'\b(?:pip|install|curl|wget|nc|netcat|ssh|scp|bash|sh|powershell|python\s+-m\s+pip)\b', command))

        risky_calls: set[str] = set()
        for fn in _entrypoint_function_nodes(facts):
            for node in _iter_non_nested_nodes(fn):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in imported_calls:
                    target = imported_calls[node.func.id]
                    if target.startswith("socket.") or risky_process_call(node):
                        risky_calls.add(target)
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    owner = node.func.value.id
                    attr = node.func.attr
                    if owner in os_aliases and attr in {"popen", "system"}:
                        if risky_process_call(node):
                            risky_calls.add(f"{owner}.{attr}")
                    elif owner in subprocess_aliases and attr in {"run", "call", "check_call", "check_output", "Popen"}:
                        if risky_process_call(node):
                            risky_calls.add(f"{owner}.{attr}")
                    elif owner in socket_aliases and attr in {"socket", "create_connection"}:
                        risky_calls.add(f"{owner}.{attr}")
        if risky_calls:
            return [{
                "pattern": "RUNTIME_PACKAGE_INSTALL",
                "severity": "critical",
                "evidence": "entrypoint process/network call: " + ", ".join(sorted(risky_calls)),
            }]
    return []


# ---------------------------------------------------------------------------
# AST-based detectors (Layer 2: structural analysis beyond regex)
# ---------------------------------------------------------------------------

# Modules that are trusted by the evaluator and must not be mutated by submissions.
BASE_TRUSTED_MODULES = frozenset({
    "reference", "utils", "eval", "__main__",
})
TRUSTED_MODULES = BASE_TRUSTED_MODULES

# Evaluator-critical functions that should never be replaced.
BASE_TRUSTED_FUNCTIONS = frozenset({
    "generate_input", "ref_kernel", "check_implementation",
    "clear_l2_cache", "clear_l2_cache_large", "time_kernel",
    "_clone_data", "_run_single_benchmark", "_run_single_test",
})
TRUSTED_FUNCTIONS = BASE_TRUSTED_FUNCTIONS


def _safe_ast_parse(code: str):
    """Parse Python code into AST, returning None on SyntaxError.

    Submissions may contain embedded C++ in triple-quoted strings which can
    confuse the parser when the string isn't properly closed.
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _collect_trusted_aliases(tree: ast.AST) -> dict[str, str]:
    """Walk AST and collect local aliases for trusted modules.

    Returns mapping: local_name -> module_name
    e.g. ``import reference as _ref`` → {"_ref": "reference"}
         ``import reference``        → {"reference": "reference"}
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in TRUSTED_MODULES:
                    local = alias.asname or alias.name
                    aliases[local] = mod_root
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in TRUSTED_MODULES:
                mod_root = node.module.split(".")[0]
                for alias in node.names:
                    local = alias.asname or alias.name
                    aliases[local] = mod_root
    return aliases


def detect_trusted_module_import(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 1: import of evaluator-trusted modules.

    Catches: import reference, from reference import generate_input, etc.
    Severity medium — the import alone is suspicious but not conclusive.
    """
    facts = ensure_submission_facts(code)
    if facts.ast_tree is None:
        return []

    matches = []
    for node in facts._imports:
        for alias in node.names:
            mod_root = alias.name.split(".")[0]
            if mod_root in TRUSTED_MODULES:
                matches.append({
                    "pattern": "TRUSTED_MODULE_IMPORT",
                    "severity": "medium",
                    "evidence": f"import {alias.name} (trusted evaluator module)",
                })
    for node in facts._import_froms:
        if node.module and node.module.split(".")[0] in TRUSTED_MODULES:
            names = ", ".join(a.name for a in node.names)
            matches.append({
                "pattern": "TRUSTED_MODULE_IMPORT",
                "severity": "medium",
                "evidence": f"from {node.module} import {names}",
            })
    return matches


def detect_module_mutation(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 2: attribute assignment to trusted modules.

    Catches:
      reference.generate_input = hooked_fn
      _ref.ref_kernel.__code__ = fake.__code__
      _ref_module.clear_l2_cache = noop
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    aliases = facts.trusted_aliases
    if not aliases:
        return []

    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # Direct: module.attr = value
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id in aliases:
                    mod = aliases[target.value.id]
                    attr = target.attr
                    sev = "critical" if attr in TRUSTED_FUNCTIONS else "high"
                    matches.append({
                        "pattern": "MODULE_MUTATION",
                        "severity": sev,
                        "evidence": f"{target.value.id}.{attr} = ... (mutates trusted module '{mod}')",
                    })

            # Nested: module.func.__code__ = ...
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)):
                if target.value.value.id in aliases and target.attr == "__code__":
                    mod = aliases[target.value.value.id]
                    matches.append({
                        "pattern": "CODE_REPLACEMENT",
                        "severity": "critical",
                        "evidence": (
                            f"{target.value.value.id}.{target.value.attr}.__code__ = ... "
                            f"(bytecode replacement on '{mod}')"
                        ),
                    })
    return matches


def detect_globals_mutation(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 3: __globals__ dict mutation.

    Catches:
      fn.__globals__['generate_input'] = hooked_fn
      getattr(obj, '__globals__')[key] = value
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    assigned_lines: set[int] = set()

    # Pass 1: find __globals__[key] = value assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "__globals__"):
                    key_name = None
                    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        key_name = target.slice.value
                    sev = "critical" if key_name in TRUSTED_FUNCTIONS else "high"
                    evidence = (
                        f"__globals__['{key_name}'] = ..." if key_name
                        else "__globals__[...] = ..."
                    )
                    matches.append({
                        "pattern": "GLOBALS_MUTATION",
                        "severity": sev,
                        "evidence": evidence + " (mutates function's global namespace)",
                    })
                    assigned_lines.add(getattr(node, "lineno", -1))

    # Pass 2: flag __globals__ reads (potential reconnaissance)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__globals__":
            lineno = getattr(node, "lineno", -1)
            if lineno not in assigned_lines:
                matches.append({
                    "pattern": "GLOBALS_ACCESS",
                    "severity": "high",
                    "evidence": "__globals__ attribute accessed (potential namespace manipulation)",
                })
                break  # one read finding is enough

    return matches


def detect_introspection_exploit(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 4: split frame-walk access from frame-based mutation."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    seen_patterns: set[str] = set()
    frame_namespace_aliases: set[str] = set()
    helper_frame_namespace_params: dict[str, set[int]] = {}
    saw_frame_access = False
    saw_frame_mutation = False
    function_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def expr_has_frame_namespace(expr: ast.AST | None) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ast.Name) and expr.id in frame_namespace_aliases:
            return True
        return any(
            isinstance(sub, ast.Attribute) and sub.attr in ("f_globals", "f_locals")
            for sub in ast.walk(expr)
        )

    for fn in function_defs:
        param_positions = {arg.arg: idx for idx, arg in enumerate(fn.args.args)}
        mutated_params: set[int] = set()
        for child in ast.walk(fn):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in param_positions
                ):
                    mutated_params.add(param_positions[target.value.id])
        if mutated_params:
            helper_frame_namespace_params[fn.name] = mutated_params

    for node in ast.walk(tree):
        # Track aliases of frame globals/locals dicts.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                value = node.value
                if (isinstance(value, ast.Attribute) and value.attr in ("f_globals", "f_locals")) or (
                    isinstance(value, ast.Subscript)
                    and any(
                        isinstance(sub, ast.Attribute) and sub.attr in ("f_globals", "f_locals")
                        for sub in ast.walk(value.value)
                    )
                ):
                    frame_namespace_aliases.add(target.id)

        # inspect.currentframe() / inspect.stack()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "inspect"
                    and node.func.attr in ("currentframe", "stack", "getmembers")):
                key = f"inspect.{node.func.attr}"
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    saw_frame_access = True

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            mutated_positions = helper_frame_namespace_params.get(node.func.id)
            if mutated_positions:
                for idx in mutated_positions:
                    if idx < len(node.args) and expr_has_frame_namespace(node.args[idx]):
                        saw_frame_mutation = True

        # f_back, f_globals, f_locals attribute access
        if isinstance(node, ast.Attribute) and node.attr in ("f_back", "f_globals", "f_locals"):
            key = f".{node.attr}"
            if key not in seen_patterns:
                seen_patterns.add(key)
                saw_frame_access = True

        # Writes through frame globals/locals aliases are much stronger than
        # plain access and should stand on their own.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if expr_has_frame_namespace(target.value):
                        saw_frame_mutation = True

        # sys.modules access
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys" and node.attr == "modules"):
            if "sys.modules" not in seen_patterns:
                seen_patterns.add("sys.modules")
                matches.append({
                    "pattern": "SYS_MODULES_ACCESS",
                    "severity": "high",
                    "evidence": "sys.modules accessed (potential module namespace manipulation)",
                })

    if saw_frame_mutation:
        matches.append({
            "pattern": "FRAME_WALK_MUTATION",
            "severity": "critical",
            "evidence": "frame globals/locals accessed and then mutated",
        })
    elif saw_frame_access:
        matches.append({
            "pattern": "FRAME_WALK_ACCESS",
            "severity": "medium",
            "evidence": "frame walking / frame globals access",
        })

    return matches


def detect_code_replacement(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 5: __code__ attribute assignment (bytecode replacement).

    Catches any func.__code__ = other.__code__ regardless of module context.
    The module-specific variant is handled by detect_module_mutation;
    this catches the general case.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    assigned_lines: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__code__":
                    # Module-scoped __code__ replacement is already explained by
                    # detect_module_mutation with better source attribution.
                    if (isinstance(target.value, ast.Attribute)
                            and isinstance(target.value.value, ast.Name)
                            and target.value.value.id in facts.trusted_aliases):
                        continue
                    lineno = getattr(node, "lineno", -1)
                    assigned_lines.add(lineno)
                    matches.append({
                        "pattern": "CODE_REPLACEMENT",
                        "severity": "critical",
                        "evidence": f"__code__ assignment (bytecode replacement)",
                    })

    # Flag __code__ reads if no assignment found (extraction for later use)
    if not assigned_lines:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "__code__":
                matches.append({
                    "pattern": "CODE_ACCESS",
                    "severity": "high",
                    "evidence": "__code__ attribute read (potential bytecode extraction)",
                })
                break

    return matches


BASE_TRUSTED_HARNESS_NAMES = frozenset({
    "generate_input", "check_implementation", "clear_l2_cache",
    "clear_l2_cache_large", "_clone_data", "ref_kernel",
    "calculate_stats", "run_single_benchmark", "Stats",
}) | BASE_TRUSTED_FUNCTIONS
TRUSTED_HARNESS_NAMES = BASE_TRUSTED_HARNESS_NAMES


def detect_harness_runtime_patching(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern: dynamic runtime patching of trusted evaluator functions.

    Targets the ooousay-style exploit family that scans `sys.modules` / `gc`
    for live evaluator objects, then rewrites module attrs or function globals
    through helper functions instead of direct `reference.foo = ...` syntax.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    dynamic_discovery = set()
    trusted_targets: set[str] = set()
    globals_write = False
    setattr_write = False
    builtins_aliases = {"__builtins__"}
    builtins_write = False
    host_aliases: set[str] = set()
    namespace_aliases: set[str] = set()
    string_aliases: dict[str, str] = {}
    helper_namespace_patch_params: dict[str, dict[int, set[str]]] = {}
    function_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    for fn in function_defs:
        param_positions = {arg.arg: idx for idx, arg in enumerate(fn.args.args)}
        patched_by_position: dict[int, set[str]] = defaultdict(set)
        local_string_aliases: dict[str, str] = {}
        for child in ast.walk(fn):
            if isinstance(child, ast.Assign):
                static_value = _static_string(child.value)
                if static_value is not None:
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            local_string_aliases[target.id] = static_value
                for target in child.targets:
                    if not (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in param_positions
                    ):
                        continue
                    key = _static_string(target.slice)
                    if key is None and isinstance(target.slice, ast.Name):
                        key = local_string_aliases.get(target.slice.id)
                    if key in TRUSTED_HARNESS_NAMES:
                        patched_by_position[param_positions[target.value.id]].add(key)
        if patched_by_position:
            helper_namespace_patch_params[fn.name] = patched_by_position

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    static_value = _static_string(node.value)
                    if static_value is not None:
                        string_aliases[target.id] = static_value
                    if isinstance(node.value, ast.Name) and node.value.id in builtins_aliases:
                        builtins_aliases.add(target.id)
                    if (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == "importlib"
                        and node.value.func.attr == "import_module"
                        and node.value.args
                        and _static_string(node.value.args[0]) == "__main__"
                    ):
                        host_aliases.add(target.id)
                        dynamic_discovery.add("importlib.import_module('__main__')")
                    elif (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "__import__"
                        and node.value.args
                        and _static_string(node.value.args[0]) == "__main__"
                    ):
                        host_aliases.add(target.id)
                        dynamic_discovery.add("__import__('__main__')")
                    elif (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "vars"
                        and node.value.args
                    ):
                        arg = node.value.args[0]
                        if isinstance(arg, ast.Name) and arg.id in host_aliases:
                            namespace_aliases.add(target.id)
                        elif (
                            isinstance(arg, ast.Call)
                            and isinstance(arg.func, ast.Attribute)
                            and isinstance(arg.func.value, ast.Name)
                            and arg.func.value.id == "importlib"
                            and arg.func.attr == "import_module"
                            and arg.args
                            and _static_string(arg.args[0]) == "__main__"
                        ):
                            namespace_aliases.add(target.id)
                            dynamic_discovery.add("vars(importlib.import_module('__main__'))")

        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr == "modules"):
            dynamic_discovery.add("sys.modules")
        elif isinstance(node, ast.Attribute) and node.attr in ("f_globals", "f_locals"):
            dynamic_discovery.add(f"frame {node.attr}")
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "gc"
              and node.func.attr == "get_objects"):
            dynamic_discovery.add("gc.get_objects")

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in TRUSTED_HARNESS_NAMES:
                trusted_targets.add(node.value)

        # hasattr(mod, "calculate_stats") — probing for harness functions
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "hasattr" and len(node.args) >= 2):
            attr_arg = node.args[1]
            if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                if attr_arg.value in TRUSTED_HARNESS_NAMES:
                    trusted_targets.add(attr_arg.value)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
            if len(node.args) >= 2:
                name_arg = node.args[1]
                name_value = _static_string(name_arg)
                if name_value is None and isinstance(name_arg, ast.Name):
                    name_value = string_aliases.get(name_arg.id)
                if isinstance(node.args[0], ast.Name) and node.args[0].id in builtins_aliases:
                    builtins_write = True
                if name_value in TRUSTED_HARNESS_NAMES:
                    setattr_write = True
                    trusted_targets.add(name_value)
                elif isinstance(name_arg, ast.Name):
                    setattr_write = True
                if isinstance(node.args[0], ast.Name) and node.args[0].id in host_aliases:
                    dynamic_discovery.add("main module alias")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            patched_by_position = helper_namespace_patch_params.get(node.func.id)
            if patched_by_position:
                for idx, targets in patched_by_position.items():
                    if idx >= len(node.args):
                        continue
                    if any(
                        isinstance(sub, ast.Attribute) and sub.attr in ("f_globals", "f_locals")
                        for sub in ast.walk(node.args[idx])
                    ):
                        globals_write = True
                        trusted_targets.update(targets)
                        dynamic_discovery.add("frame globals")

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    target_root = _ast_root_name(target.value)
                    key = _static_string(target.slice)
                    if target_root in builtins_aliases:
                        builtins_write = True
                    if target_root in namespace_aliases:
                        if key in TRUSTED_HARNESS_NAMES:
                            globals_write = True
                            trusted_targets.add(key)
                            dynamic_discovery.add("vars(main)")
                    if (
                        isinstance(target.value, ast.Attribute)
                        and target.value.attr == "__dict__"
                        and _ast_root_name(target.value) in host_aliases
                        and key in TRUSTED_HARNESS_NAMES
                    ):
                        globals_write = True
                        trusted_targets.add(key)
                        dynamic_discovery.add("main module __dict__")
                    if any(
                        isinstance(sub, ast.Attribute) and sub.attr == "__globals__"
                        for sub in ast.walk(target)
                    ):
                        globals_write = True
                # Direct attribute assignment: mod.calculate_stats = fake_fn
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id in builtins_aliases:
                        builtins_write = True
                    if target.attr in TRUSTED_HARNESS_NAMES:
                        setattr_write = True
                        trusted_targets.add(target.attr)

    if builtins_write:
        return [{
            "pattern": "HARNESS_RUNTIME_PATCHING",
            "severity": "critical",
            "evidence": "builtins namespace mutated at runtime",
        }]
    if dynamic_discovery and (globals_write or setattr_write) and len(trusted_targets) >= 1:
        channels = ", ".join(sorted(dynamic_discovery))
        targets = ", ".join(sorted(trusted_targets))
        return [{
            "pattern": "HARNESS_RUNTIME_PATCHING",
            "severity": "critical",
            "evidence": f"dynamic harness patching via {channels} targeting {targets}",
        }]
    return []


VALIDATION_TORCH_PATCH_TARGETS = frozenset({
    "torch.triu",
    "torch.linalg.householder_product",
})


def detect_torch_validation_monkeypatch(code: str | SubmissionFacts) -> list[dict]:
    """Detect direct monkeypatches of torch functions used by correctness checks."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                targets = [node.target]
            for target in targets:
                dotted = _ast_dotted_name(target)
                if dotted in VALIDATION_TORCH_PATCH_TARGETS:
                    hits.append(f"{dotted} = ...")

        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            base = _ast_dotted_name(node.args[0])
            dotted = f"{base}.{node.args[1].value}" if base else None
            if dotted in VALIDATION_TORCH_PATCH_TARGETS:
                hits.append(f"setattr({base}, {node.args[1].value!r}, ...)")

    if not hits:
        return []
    return [{
        "pattern": "TORCH_VALIDATION_MONKEYPATCH",
        "severity": "critical",
        "evidence": "validation torch function monkeypatch: " + ", ".join(sorted(set(hits))),
    }]


def detect_evaluator_mode_branch(code: str | SubmissionFacts) -> list[dict]:
    """Detect branching on evaluator CLI mode such as test vs benchmark."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    mode_literals = {"test", "benchmark", "leaderboard"}
    saw_argv = False
    matched_modes: set[str] = set()

    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "sys"
                and node.value.attr == "argv"):
            saw_argv = True
        elif (isinstance(node, ast.Attribute)
              and isinstance(node.value, ast.Name)
              and node.value.id == "sys"
              and node.attr == "argv"):
            saw_argv = True

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in mode_literals:
                matched_modes.add(node.value)

    if saw_argv and matched_modes:
        return [{
            "pattern": "EVALUATOR_MODE_BRANCH",
            "severity": "critical",
            "evidence": (
                "submission branches on sys.argv evaluator mode(s): "
                + ", ".join(sorted(matched_modes))
            ),
        }]
    return []


def detect_benchmark_shape_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect public benchmark-shape caches that replay cached factorization outputs."""
    facts = ensure_submission_facts(code)
    active = facts.python_active

    signals = []
    if re.search(r'def\s+_is_benchmark_shape\s*\(', active) or re.search(r'_BENCH(?:MARK)?_SHAPES\b', active):
        signals.append("explicit benchmark shape gate")
    if re.search(r'\b_CACHE\s*=\s*\{\}', active) or re.search(r'\b\w*CACHE\w*\s*=\s*\{\}', active):
        signals.append("module cache dict")
    if re.search(r'\b_READ_POS\s*=\s*\{\}', active) or re.search(r'%\s*len\s*\(\s*cached\s*\)', active):
        signals.append("cyclic cached-output cursor")
    if re.search(r'cached\.append\s*\(\s*(?:result|output|out|entry)\s*\)', active):
        signals.append("stores computed output in cache")
    if re.search(r'return\s+_(?:cached|benchmark)\w*\s*\(', active) or re.search(r'return\s+cached\s*\[', active):
        signals.append("returns cached benchmark output")
    if re.search(r'\b(?:_content_signature|_tiny_token)\s*\(', active) or re.search(
        r'\.detach\s*\(\s*\)\s*\.cpu\s*\(\s*\)\s*\.numpy\s*\(\s*\)\s*\.tobytes\s*\(',
        active,
    ):
        signals.append("content/token signature gate")
    if re.search(
        r'\b\w*CACHE\w*\s*\[[^\]]+\]\s*=\s*\([\s\S]{0,500}\b[Hh]\b[\s\S]{0,160}\btau\b',
        active,
    ) or re.search(
        r'\b\w*CACHE\w*\s*\[[^\]]+\]\s*=\s*\([\s\S]{0,500}\bout\b',
        active,
    ):
        signals.append("stores factorization output in cache")
    if re.search(
        r'def\s+_\w*lookup\s*\([^)]*\):[\s\S]{0,2600}\breturn\s+(?:[Hh]\s*,\s*tau|out|hit)\b',
        active,
    ):
        signals.append("lookup returns cached factorization")
    if re.search(
        r'hit\s*=\s*_\w*lookup\s*\(\s*A\b[\s\S]{0,220}?if\s+hit\s+is\s+not\s+None\s*:[\s\S]{0,80}?return\s+hit\b',
        active,
    ):
        signals.append("entrypoint returns lookup hit")
    if re.search(
        r'\b[Hh]\s*,\s*tau\s*=\s*_(?:compute|blocked|run)\w*\s*\([^)]*\)[\s\S]{0,260}?_\w*store\s*\(',
        active,
    ):
        signals.append("compute result stored for replay")
    if re.search(r'\b_(?:BLIND|GUARDED|BENCH)_CACHE\b', active):
        signals.append("shape-only replay cache")

    required = {
        "explicit benchmark shape gate",
        "module cache dict",
        "stores computed output in cache",
    }
    if required.issubset(signals) and (
        "cyclic cached-output cursor" in signals
        or "returns cached benchmark output" in signals
    ):
        return [{
            "pattern": "BENCHMARK_SHAPE_REPLAY",
            "severity": "critical",
            "evidence": "benchmark-shape output replay signals: " + ", ".join(signals),
        }]
    replay_cache_required = {
        "explicit benchmark shape gate",
        "module cache dict",
        "stores factorization output in cache",
        "lookup returns cached factorization",
        "entrypoint returns lookup hit",
    }
    if replay_cache_required.issubset(signals) and (
        "content/token signature gate" in signals
        or "compute result stored for replay" in signals
        or "shape-only replay cache" in signals
    ):
        return [{
            "pattern": "BENCHMARK_SHAPE_REPLAY",
            "severity": "critical",
            "evidence": "benchmark-shape output replay signals: " + ", ".join(signals),
        }]
    return []


RE_REFERENCE_GENERATE_INPUT = re.compile(r'from\s+reference\s+import\s+generate_input')
RE_PRECOMPUTE_STATE = re.compile(r'\b(?:_PRECOMPUTE_STATE|_PUBLIC_CASES|_PUBLIC_WARMUP_DUPLICATES)\b')
RE_PRECOMPUTE_APPEND = re.compile(r'outputs\.append\s*\(\s*_compute_output\s*\(')
RE_PRECOMPUTE_REPLAY = re.compile(r'state\.outputs\s*\[\s*state\.cursor\s*\]')


def detect_reference_precompute_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: import trusted reference generator, precompute public cases, replay outputs.

    This targets files that do not mutate evaluator state but still leverage
    trusted `reference.generate_input` to synthesize benchmark cases ahead of
    timing and then return precomputed outputs.
    """
    facts = ensure_submission_facts(code)
    code = facts.python_active
    public_seed_signals = []
    if re.search(r'\b_PUBLIC_SPECS\s*=\s*\{', code):
        public_seed_signals.append("_PUBLIC_SPECS")
    if re.search(r'\b_PUBLIC_RESULTS\s*=\s*\{', code):
        public_seed_signals.append("_PUBLIC_RESULTS")
    if re.search(r'torch\.Generator\s*\(', code) and re.search(r'\.manual_seed\s*\(', code):
        public_seed_signals.append("manual seeded public input generation")
    if re.search(r'torch\.randn\s*\(', code):
        public_seed_signals.append("torch.randn public input synthesis")
    if re.search(r'["\']output["\']\s*:\s*torch\.geqrf\s*\(', code):
        public_seed_signals.append("precomputed torch.geqrf output")
    if re.search(r'\bentry\s*\[\s*["\']fingerprint["\']\s*\]\s*==\s*_fingerprint\s*\(', code):
        public_seed_signals.append("fingerprint gate")
    if re.search(r'return\s+\w+\s*\[\s*["\']output["\']\s*\]', code):
        public_seed_signals.append("returns stored output field")

    public_seed_required = {
        "_PUBLIC_SPECS",
        "_PUBLIC_RESULTS",
        "manual seeded public input generation",
        "torch.randn public input synthesis",
        "precomputed torch.geqrf output",
        "fingerprint gate",
        "returns stored output field",
    }
    if public_seed_required.issubset(public_seed_signals):
        return [{
            "pattern": "REFERENCE_PRECOMPUTE_REPLAY",
            "severity": "critical",
            "evidence": (
                "public-seed precompute replay signals: "
                + ", ".join(public_seed_signals)
            ),
        }]

    import_hits = []
    imported_functions = set()
    for module_name in sorted(TRUSTED_MODULES):
        for function_name in sorted(TRUSTED_FUNCTIONS):
            if re.search(
                rf'from\s+{re.escape(module_name)}\s+import\s+{re.escape(function_name)}\b',
                code,
            ):
                import_hits.append(f"{module_name}.{function_name}")
                imported_functions.add(function_name)
    if not import_hits:
        return []
    if not RE_PRECOMPUTE_STATE.search(code):
        return []
    call_count = sum(
        len(re.findall(rf'\b{re.escape(function_name)}\s*\(', code))
        for function_name in imported_functions
    )
    if call_count < 2:
        return []
    if not RE_PRECOMPUTE_APPEND.search(code):
        return []
    if not RE_PRECOMPUTE_REPLAY.search(code):
        return []

    return [{
        "pattern": "REFERENCE_PRECOMPUTE_REPLAY",
        "severity": "critical",
        "evidence": (
            f"trusted import(s) {', '.join(sorted(import_hits))} used to precompute public cases "
            "and replay outputs from precompute state"
        ),
    }]


def detect_shape_output_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: output replay keyed only by config / shape state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    active = facts.python_active
    if re.search(r'torch\.zeros', active) and re.search(r'return\s+_Z\s*\[\s*key\s*\]', active):
        return [{
            "pattern": "SHAPE_OUTPUT_REPLAY",
            "severity": "critical",
            "evidence": f"shape-keyed zero/output cache returned directly from {entrypoint_name}",
        }]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not is_entrypoint_name(node.name):
            continue

        state_slots: dict[str, tuple[object, ...]] = {}
        for arg, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
            if isinstance(default, (ast.List, ast.Tuple)) and len(default.elts) >= 2:
                state_slots[arg.arg] = tuple(range(len(default.elts)))

        shape_state_match = False
        replay_state = None
        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            test = child.test
            if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
                continue
            left, right = test.left, test.comparators[0]
            for state_name in state_slots:
                if (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name) and left.value.id == state_name
                        and isinstance(right, ast.Name)) or (
                    isinstance(right, ast.Subscript) and isinstance(right.value, ast.Name) and right.value.id == state_name
                    and isinstance(left, ast.Name)
                ):
                    returns = [stmt for stmt in child.body if isinstance(stmt, ast.Return)]
                    if returns:
                        replay_state = state_name
                        shape_state_match = True
                        break
            if shape_state_match:
                break

        if shape_state_match and replay_state:
            helper_called = any(
                isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)
                for stmt in node.body
            )
            for sub in ast.walk(tree):
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if replay_state not in {arg.arg for arg in sub.args.args}:
                    continue
                assigns_first = any(
                    isinstance(stmt, ast.Assign)
                    and any(
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == replay_state
                        for t in stmt.targets
                    )
                    for stmt in ast.walk(sub)
                )
                if assigns_first:
                    return [{
                        "pattern": "SHAPE_OUTPUT_REPLAY",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} replays saved output based only on config/shape state",
                    }]
    return []


def detect_timed_input_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: timed-input preprocess/quant replay keyed by shape or pointer."""
    facts = ensure_submission_facts(code)
    active = facts.python_active
    indicators = []

    if re.search(r'_q_fp8_cache', active) and re.search(r'q\.data_ptr\s*\(', active):
        indicators.append("q_fp8 cached by q.data_ptr()")
    if re.search(r"c\['_q_ptr'\]\s*!=\s*q_ptr", active) and re.search(r"c\['_q_fp8'\]", active):
        indicators.append("stateful _q_ptr/_q_fp8 reuse")
    if re.search(r'_q_scale', active) and re.search(r'q_scale\s*=\s*_q_scale', active):
        indicators.append("reused q_scale buffer")
    if re.search(r'_FP8_BUF_CACHE', active) and re.search(r'dynamic_per_tensor_quant', active):
        indicators.append("shape-keyed fp8 quant buffer cache")
    if re.search(r'_s\s*=\s*\[\s*-?1\s*,\s*None\s*\]', active) and re.search(r'return\s+_s\s*\[\s*1\s*\]', active):
        indicators.append("saved output reused from mutable default state")

    # Only promote when the cached object is tied to live input quantization or
    # transformed q tensors, not just generic workspace reuse.
    if indicators and (
        re.search(r'q_fp8', active)
        or re.search(r'dynamic_per_tensor_quant', active)
        or "saved output reused from mutable default state" in indicators
    ):
        return [{
            "pattern": "TIMED_INPUT_REPLAY",
            "severity": "critical",
            "evidence": "timed-input quantization / transformed-Q replay: " + ", ".join(sorted(set(indicators))),
        }]
    return []


def detect_pointer_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: single-slot output replay keyed by input pointer equality."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    def _is_data_ptr_call(expr: ast.AST | None) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "data_ptr"
        )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not is_entrypoint_name(node.name):
            continue

        pointer_aliases: set[str] = set()
        saved_ptr = None
        saved_out = None

        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    # Catches bare data_ptr() AND tuples/containers that contain data_ptr()
                    if _expr_has_data_ptr_fast(child.value, facts._nodes_with_data_ptr):
                        pointer_aliases.add(child.targets[0].id)

        def _pointer_pair(left: ast.AST, right: ast.AST) -> Optional[tuple[str, Optional[str]]]:
            if isinstance(left, ast.Name) and left.id in pointer_aliases and isinstance(right, ast.Name):
                return right.id, left.id
            if isinstance(right, ast.Name) and right.id in pointer_aliases and isinstance(left, ast.Name):
                return left.id, right.id
            if _is_data_ptr_call(left) and isinstance(right, ast.Name):
                return right.id, None
            if _is_data_ptr_call(right) and isinstance(left, ast.Name):
                return left.id, None
            return None

        for idx, stmt in enumerate(node.body):
            if not isinstance(stmt, ast.If):
                continue
            compare = stmt.test
            if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1 and isinstance(compare.ops[0], ast.NotEq)):
                continue
            pair = _pointer_pair(compare.left, compare.comparators[0])
            if pair is None:
                continue
            saved_ptr_name, pointer_alias_name = pair
            stored_out_names: set[str] = set()
            stores_ptr = False
            for inner in ast.walk(stmt):
                if not isinstance(inner, ast.Assign):
                    continue
                for target in inner.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id == saved_ptr_name:
                        if (
                            (pointer_alias_name is not None and isinstance(inner.value, ast.Name) and inner.value.id == pointer_alias_name)
                            or _is_data_ptr_call(inner.value)
                            or _expr_has_data_ptr_fast(inner.value, facts._nodes_with_data_ptr)
                        ):
                            stores_ptr = True
                    elif _looks_output_value_name(target.id) or _looks_stateful_name(target.id):
                        stored_out_names.add(target.id)
            if not stores_ptr or not stored_out_names:
                continue
            for follow in node.body[idx + 1:]:
                if isinstance(follow, ast.Return) and follow.value is not None:
                    if _expr_names(follow.value) & stored_out_names:
                        return [{
                            "pattern": "POINTER_REPLAY",
                            "severity": "critical",
                            "evidence": f"{entrypoint_name} refreshes saved output on pointer mismatch then replays it",
                        }]
                    break

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            compare = child.test
            if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1):
                continue
            op = compare.ops[0]
            left = compare.left
            right = compare.comparators[0]

            if isinstance(op, ast.In):
                # Pattern: if cache_key in cache_dict — dict-based pointer cache
                if not (isinstance(left, ast.Name) and left.id in pointer_aliases):
                    continue
                dict_name = _ast_root_name(right)
                if dict_name is None:
                    continue
                has_return = any(
                    isinstance(n, ast.Return) and n.value is not None
                    for n in ast.walk(child)
                )
                if not has_return:
                    continue
                dict_written = any(
                    isinstance(n, ast.Assign) and any(
                        _ast_root_name(t) == dict_name for t in n.targets
                    )
                    for n in ast.walk(node)
                ) or any(
                    isinstance(n, ast.Assign) and any(
                        _ast_root_name(t) == dict_name for t in n.targets
                    )
                    for n in tree.body
                    if isinstance(n, ast.Assign)
                )
                if dict_written:
                    return [{
                        "pattern": "POINTER_REPLAY",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns saved output from dict keyed by input data_ptr",
                    }]
                continue

            if not isinstance(op, ast.Eq):
                continue
            pair = _pointer_pair(left, right)
            if pair is None:
                continue
            returns = [stmt for stmt in child.body if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name)]
            if not returns:
                continue
            saved_ptr = pair[0]
            saved_out = returns[0].value.id
            break

        if not saved_ptr or not saved_out:
            continue

        stores_ptr = False
        stores_out = False
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == saved_ptr:
                    if (isinstance(child.value, ast.Name) and child.value.id in pointer_aliases) or _is_data_ptr_call(child.value):
                        stores_ptr = True
                elif target.id == saved_out:
                    stores_out = True

        if stores_ptr and stores_out:
            return [{
                "pattern": "POINTER_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_name} returns saved output when input data_ptr matches previous pointer",
            }]

    return []


def detect_pointer_cache_hit_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect pointer/id-derived cache hits that return cached outputs."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)
    function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def cache_ok(name: Optional[str]) -> bool:
        if not name:
            return False
        lowered = name.lower()
        if any(token in lowered for token in ("compile", "compiled", "kernel", "module", "plan", "workspace", "preprocess", "scale", "meta")):
            return False
        return any(token in lowered for token in ("cache", "saved", "memo", "result", "out"))

    def identity_expr(expr: ast.AST | None, input_names: set[str], aliases: set[str]) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ast.Name) and expr.id in input_names:
            return True
        if _expr_has_data_ptr_fast(expr, facts._nodes_with_data_ptr):
            return bool(_expr_names(expr) & (input_names | aliases))
        for node in ast.walk(expr):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "id"
                and node.args
                and (_expr_names(node.args[0]) & (input_names | aliases))
            ):
                return True
        return False

    def returned_names(body: list[ast.stmt]) -> set[str]:
        names: set[str] = set()
        for stmt in body:
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                names.update(_expr_names(stmt.value))
        return names

    def thread_target_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        targets: set[str] = set()
        for node in _iter_non_nested_nodes(fn):
            if not isinstance(node, ast.Call):
                continue
            is_thread_ctor = (
                (isinstance(node.func, ast.Name) and node.func.id == "Thread")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "Thread")
            )
            if not is_thread_ctor:
                continue
            for kw in node.keywords:
                if kw.arg == "target" and isinstance(kw.value, ast.Name):
                    targets.add(kw.value.id)
        return targets

    def helper_identity_cache_stores(helper: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        helper_inputs = _function_input_names(helper)
        stores: set[str] = set()
        for node in _iter_non_nested_nodes(helper):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                cache_name = _ast_root_name(target.value)
                if not cache_name:
                    continue
                if not (_expr_names(target.slice) & helper_inputs and _expr_names(node.value) & helper_inputs):
                    continue
                lowered = cache_name.lower()
                if cache_ok(cache_name) or "mailbox" in lowered:
                    stores.add(cache_name)
        return stores

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        aliases = set(input_names)
        key_aliases: set[str] = set()
        cache_hits: dict[str, tuple[str, str]] = {}
        cache_stores: set[tuple[str, str]] = set()

        for stmt in _iter_non_nested_nodes(fn):
            if not isinstance(stmt, ast.Assign):
                continue
            value_names = _expr_names(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    if value_names & aliases:
                        aliases.add(target.id)
                    if identity_expr(stmt.value, input_names, aliases):
                        key_aliases.add(target.id)
                        aliases.add(target.id)

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    value = stmt.value
                    cache_name = None
                    key_expr = None
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                        if value.func.attr in {"get", "pop"} and value.args:
                            cache_name = _ast_root_name(value.func.value)
                            key_expr = value.args[0]
                    elif isinstance(value, ast.Subscript):
                        cache_name = _ast_root_name(value.value)
                        key_expr = value.slice
                    if cache_ok(cache_name) and (
                        (isinstance(key_expr, ast.Name) and key_expr.id in key_aliases)
                        or identity_expr(key_expr, input_names, aliases)
                    ):
                        cache_hits[target.id] = (cache_name or "", target.id)

                for target in stmt.targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    cache_name = _ast_root_name(target.value)
                    if not cache_ok(cache_name):
                        continue
                    key_expr = target.slice
                    if not (
                        (isinstance(key_expr, ast.Name) and key_expr.id in key_aliases)
                        or identity_expr(key_expr, input_names, aliases)
                    ):
                        continue
                    if _expr_names(stmt.value) & input_names or _expr_contains_input_derived_call(stmt.value, input_names):
                        cache_stores.add((cache_name or "", "identity"))
                    elif any(_looks_output_value_name(name) for name in _expr_names(stmt.value)):
                        cache_stores.add((cache_name or "", "identity"))

        for stmt in _iter_non_nested_nodes(fn):
            if not isinstance(stmt, ast.If) or _body_has_calls(stmt.body):
                continue
            names = returned_names(stmt.body)
            for hit_name, (cache_name, _) in cache_hits.items():
                if hit_name in names and (cache_name, "identity") in cache_stores:
                    return [{
                        "pattern": "POINTER_REPLAY",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns cache hit keyed by input pointer/id",
                    }]

        for stmt in _iter_non_nested_nodes(fn):
            if not isinstance(stmt, ast.For):
                continue
            iter_expr = stmt.iter
            if (
                isinstance(iter_expr, ast.Call)
                and isinstance(iter_expr.func, ast.Name)
                and iter_expr.func.id == "list"
                and iter_expr.args
            ):
                iter_expr = iter_expr.args[0]
            if not (
                isinstance(iter_expr, ast.Call)
                and isinstance(iter_expr.func, ast.Attribute)
                and iter_expr.func.attr == "items"
                and isinstance(stmt.target, ast.Tuple)
                and len(stmt.target.elts) == 2
                and isinstance(stmt.target.elts[0], ast.Name)
                and isinstance(stmt.target.elts[1], ast.Name)
            ):
                continue
            cache_name = _ast_root_name(iter_expr.func.value)
            if not cache_ok(cache_name):
                continue
            key_name = stmt.target.elts[0].id
            value_name = stmt.target.elts[1].id
            for child in stmt.body:
                if not isinstance(child, ast.If) or _body_has_calls(child.body):
                    continue
                if value_name not in returned_names(child.body):
                    continue
                if not identity_expr(child.test, input_names, aliases | {key_name}):
                    continue
                if (cache_name or "", "identity") in cache_stores:
                    return [{
                        "pattern": "POINTER_REPLAY",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns cache item whose key matches input pointer/id",
                    }]

        thread_cache_names: set[str] = set()
        for helper_name in thread_target_names(fn):
            helper = function_defs.get(helper_name)
            if helper is not None:
                thread_cache_names.update(helper_identity_cache_stores(helper))
        if thread_cache_names:
            for stmt in _iter_non_nested_nodes(fn):
                if not isinstance(stmt, ast.Return) or stmt.value is None:
                    continue
                value = stmt.value
                cache_name = None
                key_expr = None
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                    if value.func.attr in {"get", "pop"} and value.args:
                        cache_name = _ast_root_name(value.func.value)
                        key_expr = value.args[0]
                elif isinstance(value, ast.Subscript):
                    cache_name = _ast_root_name(value.value)
                    key_expr = value.slice
                if cache_name not in thread_cache_names:
                    continue
                if (
                    (isinstance(key_expr, ast.Name) and key_expr.id in key_aliases)
                    or identity_expr(key_expr, input_names, aliases)
                ):
                    return [{
                        "pattern": "POINTER_REPLAY",
                        "severity": "critical",
                        "evidence": f"{entrypoint_name} returns thread-populated cache keyed by input identity",
                    }]

    return []


def detect_partial_bound_storage_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect functools.partial entrypoints that replay from bound mutable state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    functools_aliases = {"functools"}
    partial_names = {"partial"}
    types_aliases = {"types"}
    namespace_names = {"SimpleNamespace"}
    for node in facts._imports:
        for alias in node.names:
            if alias.name == "functools":
                functools_aliases.add(alias.asname or alias.name)
            elif alias.name == "types":
                types_aliases.add(alias.asname or alias.name)
    for node in facts._import_froms:
        if node.module == "functools":
            for alias in node.names:
                if alias.name == "partial":
                    partial_names.add(alias.asname or alias.name)
        elif node.module == "types":
            for alias in node.names:
                if alias.name == "SimpleNamespace":
                    namespace_names.add(alias.asname or alias.name)

    def is_partial_call(expr: ast.AST | None) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        if isinstance(expr.func, ast.Name):
            return expr.func.id in partial_names
        return (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "partial"
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id in functools_aliases
        )

    def mutable_literal(expr: ast.AST | None) -> bool:
        if isinstance(expr, (ast.List, ast.Dict, ast.Set)):
            return True
        if not isinstance(expr, ast.Call):
            return False
        if isinstance(expr.func, ast.Name) and expr.func.id in namespace_names:
            return True
        return (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "SimpleNamespace"
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id in types_aliases
        )

    mutable_roots: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not mutable_literal(stmt.value):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                mutable_roots.add(target.id)

    def mutable_bound_arg(expr: ast.AST | None) -> bool:
        return mutable_literal(expr) or (
            isinstance(expr, ast.Name) and expr.id in mutable_roots
        )

    def return_from_bound(expr: ast.AST | None, bound_params: set[str], aliases: set[str]) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in aliases
        return isinstance(expr, (ast.Subscript, ast.Attribute)) and _ast_root_name(expr) in bound_params

    mutating_methods = {
        "add", "append", "clear", "extend", "insert", "pop", "popitem",
        "remove", "rotate", "setdefault", "update",
    }

    def partial_bindings() -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]]:
        bindings: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
                continue
            if not is_partial_call(stmt.value) or not stmt.value.args:
                continue
            target_name = stmt.value.args[0].id if isinstance(stmt.value.args[0], ast.Name) else None
            target_fn = functions.get(target_name or "")
            if target_fn is None:
                continue
            positional_params = list(target_fn.args.posonlyargs) + list(target_fn.args.args)
            bound_params: set[str] = set()
            for arg, value in zip(positional_params, stmt.value.args[1:]):
                if mutable_bound_arg(value):
                    bound_params.add(arg.arg)
            for keyword in stmt.value.keywords:
                if keyword.arg and mutable_bound_arg(keyword.value):
                    bound_params.add(keyword.arg)
            if bound_params:
                bindings.append((target_fn, bound_params))
        return bindings

    def function_replays_bound_storage(fn: ast.FunctionDef | ast.AsyncFunctionDef, bound_params: set[str]) -> bool:
        input_params = _function_input_names(fn) - bound_params
        if not input_params:
            return False
        tainted = set(input_params)
        stores_bound_output = False
        has_fast_return = False

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                if _expr_names(stmt.value) & tainted or _expr_contains_input_derived_call(stmt.value, input_params):
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                for target in stmt.targets:
                    if _ast_root_name(target) in bound_params and (
                        _expr_names(stmt.value) & tainted
                        or _expr_contains_input_derived_call(stmt.value, input_params)
                        or any(_looks_output_value_name(name) for name in _expr_names(stmt.value))
                    ):
                        stores_bound_output = True
            elif isinstance(stmt, ast.AugAssign):
                if _ast_root_name(stmt.target) in bound_params:
                    stores_bound_output = True
            elif (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and stmt.func.attr in mutating_methods
                and _ast_root_name(stmt.func.value) in bound_params
            ):
                if (
                    any(_expr_names(arg) & tainted for arg in stmt.args)
                    or any(_expr_contains_input_derived_call(arg, input_params) for arg in stmt.args)
                    or stmt.func.attr in {"clear", "pop", "popitem", "remove"}
                ):
                    stores_bound_output = True
            elif isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if _expr_names(stmt.test) & input_params:
                    continue
                if not (_expr_names(stmt.test) & bound_params):
                    continue
                aliases: set[str] = set()
                for inner in stmt.body:
                    if isinstance(inner, ast.Assign):
                        if isinstance(inner.value, (ast.Subscript, ast.Attribute)) and _ast_root_name(inner.value) in bound_params:
                            for target in inner.targets:
                                aliases.update(_target_names(target))
                    elif isinstance(inner, ast.Return) and return_from_bound(inner.value, bound_params, aliases):
                        has_fast_return = True

        return stores_bound_output and has_fast_return

    for fn, bound_params in partial_bindings():
        if function_replays_bound_storage(fn, bound_params):
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_label(facts.entrypoint_name)} partial replays mutated bound mutable state",
            }]

    return []


def detect_class_pointer_sentinel_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect class __call__ entrypoints replaying state by input data_ptr."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    none_roots = set(facts._none_inited)
    if not none_roots:
        return []

    def data_ptr_owner(expr: ast.AST | None) -> Optional[str]:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "data_ptr":
            return _ast_root_name(expr.func.value)
        return None

    def compare_state_none(expr: ast.AST | None) -> Optional[str]:
        if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
            return None
        if not isinstance(expr.ops[0], (ast.Is, ast.IsNot)):
            return None
        left_root = _ast_root_name(expr.left)
        right_root = _ast_root_name(expr.comparators[0])
        if _expr_is_none(expr.left) and right_root in none_roots:
            return right_root
        if _expr_is_none(expr.comparators[0]) and left_root in none_roots:
            return left_root
        return None

    def compare_input_state_ptr(expr: ast.AST | None, input_names: set[str]) -> Optional[str]:
        if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or not isinstance(expr.ops[0], ast.Eq):
            return None
        if len(expr.comparators) != 1:
            return None
        left_owner = data_ptr_owner(expr.left)
        right_owner = data_ptr_owner(expr.comparators[0])
        if left_owner in input_names and right_owner in none_roots:
            return right_owner
        if right_owner in input_names and left_owner in none_roots:
            return left_owner
        return None

    for cls in facts._class_defs:
        if not is_entrypoint_name(cls.name):
            continue
        method = _method_from_class(cls, ("__call__",))
        if method is None:
            continue
        input_names = _function_input_names(method)
        if not input_names:
            continue
        for stmt in _iter_non_nested_nodes(method):
            if not isinstance(stmt, ast.If) or _body_has_calls(stmt.body):
                continue
            tests = stmt.test.values if isinstance(stmt.test, ast.BoolOp) else [stmt.test]
            none_states = {state for test in tests if (state := compare_state_none(test))}
            ptr_states = {state for test in tests if (state := compare_input_state_ptr(test, input_names))}
            replay_states = none_states & ptr_states
            if not replay_states:
                continue
            returns_replay_state = any(
                isinstance(inner, ast.Return)
                and inner.value is not None
                and _ast_root_name(inner.value) in replay_states
                for inner in stmt.body
            )
            if returns_replay_state:
                return [{
                    "pattern": "POINTER_REPLAY",
                    "severity": "critical",
                    "evidence": f"class {cls.name}.__call__ replays state guarded by input data_ptr",
                }]

    return []


def detect_mutable_default_state_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect persistent default-argument containers used as replay state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    mutating_methods = {
        "add", "append", "clear", "extend", "insert", "pop", "popitem",
        "remove", "rotate", "setdefault", "update",
    }

    def is_mutable_default(expr: ast.AST | None) -> bool:
        return isinstance(expr, (ast.List, ast.Dict, ast.Set)) or (
            isinstance(expr, ast.Tuple) and bool(expr.elts)
        )

    def return_from_default(expr: ast.AST | None, default_names: set[str], aliases: set[str]) -> bool:
        if isinstance(expr, ast.Name) and expr.id in aliases:
            return True
        return isinstance(expr, (ast.Subscript, ast.Attribute, ast.Name)) and _ast_root_name(expr) in default_names

    for fn in _entrypoint_function_nodes(facts):
        positional = list(fn.args.posonlyargs) + list(fn.args.args)
        default_args = positional[-len(fn.args.defaults):] if fn.args.defaults else []
        default_names = {
            arg.arg for arg, default in zip(default_args, fn.args.defaults) if is_mutable_default(default)
        }
        if not default_names:
            continue
        input_names = _function_input_names(fn) - default_names
        if not input_names:
            continue
        tainted = set(input_names)
        aliases: set[str] = set()
        stores_default = False
        has_fast_return = False

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                value_tainted = bool(_expr_names(stmt.value) & tainted) or _expr_contains_input_derived_call(stmt.value, input_names)
                if value_tainted:
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and _ast_root_name(stmt.value) in default_names:
                        aliases.add(target.id)
                    if _ast_root_name(target) in default_names and (
                        value_tainted
                        or any(_looks_output_value_name(name) for name in _expr_names(stmt.value))
                    ):
                        stores_default = True
            elif (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and stmt.func.attr in mutating_methods
                and _ast_root_name(stmt.func.value) in default_names
            ):
                if any((_expr_names(arg) & tainted) or _expr_contains_input_derived_call(arg, input_names) for arg in stmt.args):
                    stores_default = True
            elif isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and return_from_default(inner.value, default_names, aliases)
                    for inner in stmt.body
                ):
                    has_fast_return = True

        if stores_default and has_fast_return:
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_name} replays output from persistent mutable default state",
            }]

    return []


def detect_function_attribute_state_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect replay state stored on the exported function object."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    def function_attr(expr: ast.AST | None, fn_name: str) -> Optional[str]:
        if (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == fn_name
        ):
            return expr.attr
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "getattr"
            and len(expr.args) >= 2
            and isinstance(expr.args[0], ast.Name)
            and expr.args[0].id == fn_name
            and isinstance(expr.args[1], ast.Constant)
            and isinstance(expr.args[1].value, str)
        ):
            return expr.args[1].value
        return None

    def uses_attr(expr: ast.AST | None, fn_name: str, aliases: set[str]) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ast.Name) and expr.id in aliases:
            return True
        return any(function_attr(node, fn_name) is not None for node in ast.walk(expr))

    for fn in _entrypoint_function_nodes(facts):
        if not is_entrypoint_name(fn.name):
            continue
        input_names = _function_input_names(fn)
        if not input_names:
            continue
        aliases: set[str] = set()
        stores_attr = False
        has_fast_return = False
        tainted = set(input_names)

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                if function_attr(stmt.value, fn.name):
                    for target in stmt.targets:
                        aliases.update(_target_names(target))
                value_tainted = bool(_expr_names(stmt.value) & tainted) or _expr_contains_input_derived_call(stmt.value, input_names)
                if value_tainted:
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                for target in stmt.targets:
                    if function_attr(target, fn.name) and value_tainted:
                        stores_attr = True
            elif (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Name)
                and stmt.func.id == "setattr"
                and len(stmt.args) >= 3
                and isinstance(stmt.args[0], ast.Name)
                and stmt.args[0].id == fn.name
                and (bool(_expr_names(stmt.args[2]) & tainted) or _expr_contains_input_derived_call(stmt.args[2], input_names))
            ):
                stores_attr = True
            elif isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if any(
                    isinstance(inner, ast.Return) and uses_attr(inner.value, fn.name, aliases)
                    for inner in stmt.body
                ):
                    has_fast_return = True

        if stores_attr and has_fast_return:
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_name} replays output stored on its function object",
            }]

    return []


def detect_nonlocal_factory_state_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect factory-returned closures that replay nonlocal state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    factories: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        stmt.name: stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    exported_factories: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
            continue
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
            if stmt.value.func.id in factories:
                exported_factories.add(stmt.value.func.id)

    def inner_replays_state(inner: ast.FunctionDef | ast.AsyncFunctionDef, state_names: set[str]) -> bool:
        nonlocal_names = {
            name
            for stmt in inner.body
            if isinstance(stmt, ast.Nonlocal)
            for name in stmt.names
        } & state_names
        if not nonlocal_names:
            return False
        input_names = _function_input_names(inner)
        tainted = set(input_names)
        has_fast_return = False
        stores_state = False
        for stmt in inner.body:
            if isinstance(stmt, ast.Nonlocal):
                continue
            if isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if any(
                    isinstance(ret, ast.Return)
                    and isinstance(ret.value, ast.Name)
                    and ret.value.id in nonlocal_names
                    for ret in stmt.body
                ):
                    has_fast_return = True
                    continue
            if isinstance(stmt, ast.Assign):
                value_tainted = bool(_expr_names(stmt.value) & tainted) or _expr_contains_input_derived_call(stmt.value, input_names)
                if value_tainted:
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                if any(isinstance(target, ast.Name) and target.id in nonlocal_names for target in stmt.targets) and value_tainted:
                    stores_state = True
                    continue
            if isinstance(stmt, ast.Return):
                continue
        return has_fast_return and stores_state

    for factory_name in exported_factories:
        factory = factories[factory_name]
        local_none = {
            target.id
            for stmt in factory.body
            if isinstance(stmt, ast.Assign) and _expr_is_none(stmt.value)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        if not local_none:
            continue
        nested = {
            stmt.name: stmt
            for stmt in factory.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        returned = _factory_returned_function(factory)
        candidates = [returned] if returned is not None else list(nested.values())
        for inner in candidates:
            if inner is not None and inner_replays_state(inner, local_none):
                return [{
                    "pattern": "LAST_CALL_REPLAY",
                    "severity": "critical",
                    "evidence": f"{entrypoint_label(facts.entrypoint_name)} replays nonlocal factory state",
                }]

    return []


def detect_contextvar_state_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect contextvars used as inter-call output replay storage."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    contextvars_aliases = {"contextvars"}
    contextvar_names = {"ContextVar"}
    for node in facts._imports:
        for alias in node.names:
            if alias.name == "contextvars":
                contextvars_aliases.add(alias.asname or alias.name)
    for node in facts._import_froms:
        if node.module == "contextvars":
            for alias in node.names:
                if alias.name == "ContextVar":
                    contextvar_names.add(alias.asname or alias.name)

    context_slots: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        is_contextvar = (
            isinstance(stmt.value, ast.Call)
            and (
                (isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in contextvar_names)
                or (
                    isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "ContextVar"
                    and isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.value.id in contextvars_aliases
                )
            )
        )
        if is_contextvar:
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    context_slots.add(target.id)
    if not context_slots:
        return []

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        tainted = set(input_names)
        get_aliases: set[str] = set()
        has_fast_return = False
        stores_output = False
        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                if (
                    isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "get"
                    and _ast_root_name(stmt.value.func.value) in context_slots
                ):
                    for target in stmt.targets:
                        get_aliases.update(_target_names(target))
                if bool(_expr_names(stmt.value) & tainted) or _expr_contains_input_derived_call(stmt.value, input_names):
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
            elif isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id in get_aliases
                    for inner in stmt.body
                ):
                    has_fast_return = True
            elif (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and stmt.func.attr == "set"
                and _ast_root_name(stmt.func.value) in context_slots
                and stmt.args
                and (bool(_expr_names(stmt.args[0]) & tainted) or _expr_contains_input_derived_call(stmt.args[0], input_names))
            ):
                stores_output = True
        if has_fast_return and stores_output:
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_label(facts.entrypoint_name)} replays output through ContextVar state",
            }]

    return []


def detect_alias_state_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect local aliases to captured state used for output replay."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    mutating_methods = {"append", "extend", "insert", "update", "setdefault", "add", "__setitem__"}
    top_level_defs = {
        stmt.name
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        if not input_names:
            continue
        local_names = {
            name
            for stmt in _iter_non_nested_nodes(fn)
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            for name in _target_names(target)
        }
        captured_roots = {
            name
            for name in _expr_names(fn)
            if name not in input_names and name not in local_names and name not in top_level_defs
        }
        aliases: set[str] = set()
        tainted = set(input_names)
        stores_alias = False
        has_fast_return = False

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    target_name = stmt.targets[0].id
                    value = stmt.value
                    value_root = _ast_root_name(value)
                    if (
                        (
                            isinstance(value, (ast.Attribute, ast.Subscript))
                            and value_root not in input_names
                            and value_root is not None
                        )
                        or (
                            isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name)
                            and value.func.id in {"vars", "list", "tuple"}
                            and value.args
                            and _ast_root_name(value.args[0]) not in input_names
                        )
                    ):
                        aliases.add(target_name)
                value_tainted = bool(_expr_names(stmt.value) & tainted) or _expr_contains_input_derived_call(stmt.value, input_names)
                if value_tainted:
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                for target in stmt.targets:
                    if _ast_root_name(target) in aliases and value_tainted:
                        stores_alias = True
            elif (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and stmt.func.attr in mutating_methods
                and _ast_root_name(stmt.func.value) in aliases
                and any((bool(_expr_names(arg) & tainted) or _expr_contains_input_derived_call(arg, input_names)) for arg in stmt.args)
            ):
                stores_alias = True
            elif isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and inner.value is not None
                    and _ast_root_name(inner.value) in aliases
                    for inner in stmt.body
                ):
                    has_fast_return = True

        if stores_alias and has_fast_return:
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_name} replays output through local alias to captured state",
            }]

    return []


def detect_partial_bound_method_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect functools.partial around a bound method that replays self state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    classes = {stmt.name: stmt for stmt in tree.body if isinstance(stmt, ast.ClassDef)}
    instances: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if (
            isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id in classes
        ):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    instances[target.id] = stmt.value.func.id

    functools_aliases = {"functools"}
    partial_names = {"partial"}
    for node in facts._imports:
        for alias in node.names:
            if alias.name == "functools":
                functools_aliases.add(alias.asname or alias.name)
    for node in facts._import_froms:
        if node.module == "functools":
            for alias in node.names:
                if alias.name == "partial":
                    partial_names.add(alias.asname or alias.name)

    def is_partial_call(expr: ast.AST | None) -> bool:
        return isinstance(expr, ast.Call) and (
            (isinstance(expr.func, ast.Name) and expr.func.id in partial_names)
            or (
                isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "partial"
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id in functools_aliases
            )
        )

    def resolve_bound_method(expr: ast.AST | None) -> tuple[ast.ClassDef, str] | None:
        if not isinstance(expr, ast.Attribute):
            return None
        owner = expr.value
        class_name = None
        if isinstance(owner, ast.Name):
            class_name = instances.get(owner.id)
        elif isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id in classes:
            class_name = owner.func.id
        if class_name in classes:
            return classes[class_name], expr.attr
        return None

    def method_replays_self(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        if not method.args.args:
            return False
        self_name = method.args.args[0].arg
        input_names = _function_input_names(method)
        tainted = set(input_names)
        stores_attr = False
        has_fast_return = False
        for stmt in _iter_non_nested_nodes(method):
            if isinstance(stmt, ast.Assign):
                value_tainted = bool(_expr_names(stmt.value) & tainted) or _expr_contains_input_derived_call(stmt.value, input_names)
                if value_tainted:
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == self_name
                        and value_tainted
                    ):
                        stores_attr = True
            elif isinstance(stmt, ast.If):
                if _body_has_calls(stmt.body):
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and isinstance(inner.value, (ast.Attribute, ast.Subscript))
                    and _ast_root_name(inner.value) == self_name
                    for inner in stmt.body
                ):
                    has_fast_return = True
        return stores_attr and has_fast_return

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
            continue
        if not is_partial_call(stmt.value) or not stmt.value.args:
            continue
        bound = resolve_bound_method(stmt.value.args[0])
        if bound is None:
            continue
        cls, method_name = bound
        method = _method_from_class(cls, (method_name,))
        if method is not None and method_replays_self(method):
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_label(facts.entrypoint_name)} is partial-bound to a method replaying self state",
            }]

    return []


def detect_generator_send_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect persistent generator send() state machines used for replay."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        stmt.name: stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def generator_replays(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        send_names = {
            target.id
            for stmt in ast.walk(fn)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, (ast.Yield, ast.YieldFrom))
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        if not send_names:
            return False
        tainted = set(send_names)
        state_names: set[str] = set()
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.Assign):
                value_tainted = bool(_expr_names(stmt.value) & tainted)
                if value_tainted:
                    for target in stmt.targets:
                        names = _target_names(target)
                        tainted.update(names)
                        state_names.update(names)
        yielded_state = any(
            isinstance(stmt, (ast.Yield, ast.YieldFrom))
            and bool(_expr_names(stmt.value) & state_names)
            for stmt in ast.walk(fn)
        )
        return bool(state_names and yielded_state)

    replay_generators = {name for name, fn in functions.items() if generator_replays(fn)}
    if not replay_generators:
        return []

    generator_instances: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in replay_generators:
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    generator_instances.add(target.id)

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        for stmt in _iter_non_nested_nodes(fn):
            if (
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "send"
                and _ast_root_name(stmt.value.func.value) in generator_instances
                and stmt.value.args
                and _expr_names(stmt.value.args[0]) & input_names
            ):
                return [{
                    "pattern": "LAST_CALL_REPLAY",
                    "severity": "critical",
                    "evidence": f"{entrypoint_label(facts.entrypoint_name)} sends input into persistent replay generator",
                }]

    return []


def detect_class_self_pointer_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect class callables replaying self.state under input data_ptr equality."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    exported_classes: set[str] = {cls.name for cls in facts._class_defs if is_entrypoint_name(cls.name)}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
            continue
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
            exported_classes.add(stmt.value.func.id)

    def data_ptr_owner(expr: ast.AST | None) -> Optional[str]:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "data_ptr":
            return _ast_root_name(expr.func.value)
        return None

    for cls in facts._class_defs:
        if cls.name not in exported_classes:
            continue
        method = _method_from_class(cls, ("__call__",))
        if method is None or not method.args.args:
            continue
        self_name = method.args.args[0].arg
        input_names = _function_input_names(method)
        if not input_names:
            continue
        has_pointer_return = False
        for stmt in _iter_non_nested_nodes(method):
            if not isinstance(stmt, ast.If) or _body_has_calls(stmt.body):
                continue
            ptr_match = False
            for cmp in ast.walk(stmt.test):
                if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq):
                    continue
                left_owner = data_ptr_owner(cmp.left)
                right_owner = data_ptr_owner(cmp.comparators[0])
                if (left_owner == self_name and right_owner in input_names) or (right_owner == self_name and left_owner in input_names):
                    ptr_match = True
                    break
            if not ptr_match:
                continue
            if any(
                isinstance(inner, ast.Return)
                and inner.value is not None
                and _ast_root_name(inner.value) == self_name
                for inner in stmt.body
            ):
                has_pointer_return = True
                break
        if not has_pointer_return:
            continue
        stores_self_state = False
        for stmt in _iter_non_nested_nodes(method):
            if not isinstance(stmt, ast.Assign):
                continue
            value_names = _expr_names(stmt.value)
            for target in stmt.targets:
                if _ast_root_name(target) == self_name and (value_names & input_names or any(_looks_output_value_name(name) for name in value_names)):
                    stores_self_state = True
                    break
            if stores_self_state:
                break
        if stores_self_state:
            return [{
                "pattern": "POINTER_REPLAY",
                "severity": "critical",
                "evidence": f"class {cls.name}.__call__ replays self state guarded by input data_ptr",
            }]

    return []


def detect_lambda_pointer_dispatch_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect lambda entrypoints using operator/setitem pointer-keyed caches."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    def identity_expr(expr: ast.AST | None, input_names: set[str]) -> bool:
        if expr is None:
            return False
        if _expr_has_data_ptr_fast(expr, facts._nodes_with_data_ptr) and bool(_expr_names(expr) & input_names):
            return True
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "id"
            and node.args
            and bool(_expr_names(node.args[0]) & input_names)
            for node in ast.walk(expr)
        )

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and is_entrypoint_name(target.id) for target in stmt.targets):
            continue
        if not isinstance(stmt.value, ast.Lambda):
            continue
        lambda_node = stmt.value
        input_names = _lambda_input_names(lambda_node)
        if not input_names:
            continue
        stores: set[str] = set()
        returns: set[str] = set()
        for node in ast.walk(lambda_node.body):
            if isinstance(node, ast.Call):
                cache_name = None
                key_expr = None
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__setitem__"
                    and len(node.args) >= 2
                ):
                    cache_name = _ast_root_name(node.func.value)
                    key_expr = node.args[0]
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setitem"
                    and len(node.args) >= 3
                ):
                    cache_name = _ast_root_name(node.args[0])
                    key_expr = node.args[1]
                elif (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "setitem"
                    and len(node.args) >= 3
                ):
                    cache_name = _ast_root_name(node.args[0])
                    key_expr = node.args[1]
                if cache_name and identity_expr(key_expr, input_names):
                    stores.add(cache_name)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
                    cache_name = _ast_root_name(node.func.value)
                    if cache_name and identity_expr(node.args[0], input_names):
                        returns.add(cache_name)
            elif isinstance(node, ast.Subscript):
                cache_name = _ast_root_name(node.value)
                if cache_name and identity_expr(node.slice, input_names):
                    returns.add(cache_name)
        if stores & returns:
            return [{
                "pattern": "POINTER_REPLAY",
                "severity": "critical",
                "evidence": f"{entrypoint_label(facts.entrypoint_name)} lambda caches outputs by input pointer/id",
            }]

    return []


def detect_object_output_cache_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect object/pointer-keyed caches that return stored outputs.

    This covers helper-mediated forms where the replay state is a tuple/dict
    record rather than a simple ``_last_output`` variable:
    cached = cache.get(key)
    cached_id, cached_ref, cached_output = cached
    if cached_id == id(data) and cached_ref() is data:
        return cached_output
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None or facts.entrypoint_name is None:
        return []

    function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if facts.entrypoint_name not in function_defs:
        return []

    def _reachable_functions() -> set[str]:
        reachable = {facts.entrypoint_name}
        pending = [facts.entrypoint_name]
        while pending:
            current = pending.pop()
            fn = function_defs.get(current)
            if fn is None:
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                    continue
                callee = call.func.id
                if callee in function_defs and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        return reachable

    def _assign_parts(stmt: ast.AST) -> tuple[list[ast.AST], ast.AST | None]:
        if isinstance(stmt, ast.Assign):
            return list(stmt.targets), stmt.value
        if isinstance(stmt, ast.AnnAssign):
            return [stmt.target], stmt.value
        return [], None

    def _assignment_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
        return [
            node
            for node in ast.walk(fn)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]

    def _looks_cache_name(name: Optional[str]) -> bool:
        if not name:
            return False
        lowered = name.lower()
        return any(token in lowered for token in ("cache", "saved", "save"))

    def _is_cache_lookup(value: ast.AST | None) -> bool:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            return value.func.attr == "get" and _looks_cache_name(_ast_root_name(value.func.value))
        if isinstance(value, ast.Subscript):
            return _looks_cache_name(_ast_root_name(value))
        return False

    def _is_cache_store_target(target: ast.AST | None) -> bool:
        if isinstance(target, ast.Subscript):
            return _looks_cache_name(_ast_root_name(target))
        return any(_looks_cache_name(name) for name in _target_names(target))

    def _is_output_like_name(name: str) -> bool:
        lowered = name.lower()
        return (
            lowered in {"h", "tau", "q", "r"}
            or any(token in lowered for token in ("output", "result", "out"))
        )

    def _expr_has_pointer_identity(
        expr: ast.AST | None,
        input_names: set[str],
        pointer_aliases: set[str],
    ) -> bool:
        if expr is None:
            return False
        if _expr_names(expr) & pointer_aliases:
            return True
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name)
                        and node.func.id == "id"
                        and node.args
                        and (_expr_names(node.args[0]) & input_names)):
                    return True
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "data_ptr"
                        and (_expr_names(node.func.value) & input_names)):
                    return True
                if (isinstance(node.func, ast.Name)
                        and any(token in node.func.id.lower() for token in ("storage", "fingerprint", "ver"))
                        and any(_expr_names(arg) & input_names for arg in node.args)):
                    return True
            if isinstance(node, ast.Compare):
                compare_names = _expr_names(node)
                has_identity_op = any(isinstance(op, (ast.Is, ast.Eq)) for op in node.ops)
                has_ref_name = any("ref" in name.lower() for name in compare_names)
                if has_identity_op and has_ref_name and (compare_names & input_names):
                    return True
        return False

    def _pointer_aliases(fn: ast.FunctionDef | ast.AsyncFunctionDef, input_names: set[str]) -> set[str]:
        aliases: set[str] = set()
        for stmt in _assignment_nodes(fn):
            targets, value = _assign_parts(stmt)
            if _expr_has_pointer_identity(value, input_names, set()):
                for target in targets:
                    aliases.update(_target_names(target))
        return aliases

    def _cached_output_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        cache_hit_names: set[str] = set()
        cached_output_names: set[str] = set()

        for stmt in _assignment_nodes(fn):
            targets, value = _assign_parts(stmt)
            if value is None:
                continue
            assigned_names: set[str] = set()
            for target in targets:
                assigned_names.update(_target_names(target))

            if _is_cache_lookup(value):
                cache_hit_names.update(assigned_names)

            value_root = _ast_root_name(value)
            if value_root in cache_hit_names:
                for target in targets:
                    if isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            names = _target_names(elt)
                            if any(_is_output_like_name(name) for name in names):
                                cached_output_names.update(names)
                    else:
                        cache_hit_names.update(_target_names(target))
            elif value_root in cached_output_names:
                cached_output_names.update(assigned_names)

        return cached_output_names

    def _has_pointer_cached_return(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        input_names = {arg.arg for arg in fn.args.args}
        if not input_names:
            return False
        pointer_aliases = _pointer_aliases(fn, input_names)
        cached_output_names = _cached_output_names(fn)

        for if_node in ast.walk(fn):
            if not isinstance(if_node, ast.If):
                continue
            if not _expr_has_pointer_identity(if_node.test, input_names, pointer_aliases):
                continue
            if _body_has_calls(if_node.body):
                continue
            for stmt in if_node.body:
                for nested in ast.walk(stmt):
                    if not isinstance(nested, ast.Return) or nested.value is None:
                        continue
                    returned_names = _expr_names(nested.value)
                    if returned_names & cached_output_names:
                        return True
                    if any(
                        _is_output_like_name(name)
                        and any(token in name.lower() for token in ("cache", "cached", "saved"))
                        for name in returned_names
                    ):
                        return True
        return False

    def _has_output_store(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for stmt in _assignment_nodes(fn):
            targets, value = _assign_parts(stmt)
            if value is None or not any(_is_cache_store_target(target) for target in targets):
                continue
            value_names = _expr_names(value)
            target_names: set[str] = set()
            for target in targets:
                target_names.update(_target_names(target))
            if any(_is_output_like_name(name) for name in value_names):
                return True
            if (any(_is_output_like_name(name) for name in target_names)
                    and any(_is_output_like_name(name) for name in value_names)):
                return True
        return False

    reachable = sorted(_reachable_functions())
    has_replay_return = any(
        _has_pointer_cached_return(function_defs[name])
        for name in reachable
        if name in function_defs
    )
    if not has_replay_return:
        return []

    has_store = any(
        _has_output_store(function_defs[name])
        for name in reachable
        if name in function_defs
    )
    if not has_store:
        return []

    return [{
        "pattern": "POINTER_REPLAY",
        "severity": "critical",
        "evidence": (
            "Reachable object/pointer-keyed output cache returns stored outputs "
            "under id/data_ptr/weakref guard"
        ),
    }]


def detect_last_call_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: saved-output replay when prior call signature matches current state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    # Use pre-computed indices from build_submission_facts
    none_inited = facts._none_inited
    data_ptr_helpers = facts._data_ptr_helpers
    version_helpers = facts._version_helpers
    dp_index = facts._nodes_with_data_ptr
    ver_index = facts._nodes_with_version

    def _has_ptr(expr: ast.AST | None) -> bool:
        if _expr_has_data_ptr_fast(expr, dp_index):
            return True
        if expr is None:
            return False
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in data_ptr_helpers
            for n in ast.walk(expr)
        )

    def _has_ver(expr: ast.AST | None) -> bool:
        if _expr_has_version_fast(expr, ver_index):
            return True
        if expr is None:
            return False
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in version_helpers
            for n in ast.walk(expr)
        )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not is_entrypoint_name(node.name):
            continue

        signature_features: dict[str, set[str]] = defaultdict(set)
        saved_state_features: dict[str, set[str]] = defaultdict(set)

        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue

            direct_features = set()
            if _has_ptr(child.value):
                direct_features.add("pointer")
            if _has_ver(child.value):
                direct_features.add("version")

            for target in child.targets:
                target_root = _ast_root_name(target)
                if target_root is None:
                    continue

                if direct_features:
                    if isinstance(target, ast.Name) and not _looks_stateful_name(target.id):
                        signature_features[target.id].update(direct_features)
                    else:
                        saved_state_features[target_root].update(direct_features)

                indirect_features = set()
                if isinstance(child.value, ast.Name):
                    indirect_features.update(signature_features.get(child.value.id, set()))
                    indirect_features.update(saved_state_features.get(child.value.id, set()))
                if indirect_features and target_root != _ast_root_name(child.value):
                    saved_state_features[target_root].update(indirect_features)

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            if _body_has_calls(child.body):
                continue

            return_roots = {
                _ast_root_name(stmt.value)
                for stmt in child.body
                if isinstance(stmt, ast.Return) and stmt.value is not None
            } - {None}
            if len(return_roots) != 1:
                continue
            returned_root = next(iter(return_roots))

            guard_features: set[str] = set()
            for compare in ast.walk(child.test):
                if not isinstance(compare, ast.Compare):
                    continue
                if not any(isinstance(op, (ast.Eq, ast.Is)) for op in compare.ops):
                    continue
                rights = compare.comparators
                if len(rights) != 1:
                    continue

                left = compare.left
                right = rights[0]

                left_saved = set().union(*(saved_state_features.get(name, set()) for name in _expr_names(left)))
                right_saved = set().union(*(saved_state_features.get(name, set()) for name in _expr_names(right)))

                left_current = set().union(*(signature_features.get(name, set()) for name in _expr_names(left)))
                right_current = set().union(*(signature_features.get(name, set()) for name in _expr_names(right)))

                if _has_ptr(left):
                    left_current.add("pointer")
                if _has_ptr(right):
                    right_current.add("pointer")
                if _has_ver(left):
                    left_current.add("version")
                if _has_ver(right):
                    right_current.add("version")

                guard_features.update(left_saved & right_current)
                guard_features.update(right_saved & left_current)

            if not guard_features:
                continue

            stores_output = False
            for assign in ast.walk(node):
                if not isinstance(assign, ast.Assign):
                    continue
                for target in assign.targets:
                    if _ast_root_name(target) != returned_root:
                        continue
                    stores_output = True
                    break
                if stores_output:
                    break

            if not stores_output:
                continue

            evidence_bits = []
            if "pointer" in guard_features:
                evidence_bits.append("pointer signature")
            if "version" in guard_features:
                evidence_bits.append("tensor version")
            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": (
                    f"{entrypoint_name} returns saved output when prior "
                    f"{' + '.join(evidence_bits)} matches the current call"
                ),
            }]

        # Secondary pass: direct object-identity replay.
        # Pattern: if <input> is <stateful_var>: return <stateful_var2>
        # where both stateful vars are stored elsewhere in the function body.
        # Catches: if data is _LAST_DATA: return _LAST_OUTPUT
        # Also catches abbreviated names like _cd/_co when initialized to None
        # at module level (detected via none_inited).

        # Names assigned only inside nested functions are not in the outer
        # entrypoint's scope, so exclude them from the stateful-stored set.
        _nested_assigned: set[str] = set()
        for _fn in ast.walk(node):
            if _fn is node or not isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for _ch in ast.walk(_fn):
                if isinstance(_ch, ast.Assign):
                    for _t in _ch.targets:
                        _n = _ast_root_name(_t)
                        if _n:
                            _nested_assigned.add(_n)

        stateful_stored: set[str] = {
            _ast_root_name(t)
            for child in ast.walk(node)
            if isinstance(child, ast.Assign)
            for t in child.targets
            if _looks_stateful_name(_ast_root_name(t) or "")
        } - _nested_assigned - {None}

        # All names assigned anywhere inside the entrypoint — used to confirm
        # that a none_inited var is actually used as a storage slot (not just
        # a sentinel that's never written inside the function).
        assigned_in_node: set[str] = {
            _ast_root_name(t)
            for child in ast.walk(node)
            if isinstance(child, ast.Assign)
            for t in child.targets
        } - {None}

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            if _body_has_calls(child.body):
                continue
            # Find `is` comparisons in the test (excluding `is None`)
            is_pairs: list[tuple[str, str]] = []
            for subnode in ast.walk(child.test):
                if not isinstance(subnode, ast.Compare):
                    continue
                operands = [subnode.left] + list(subnode.comparators)
                for i, op in enumerate(subnode.ops):
                    if not isinstance(op, ast.Is):
                        continue
                    lname = _ast_root_name(operands[i])
                    rname = _ast_root_name(operands[i + 1])
                    if lname and rname:
                        is_pairs.append((lname, rname))

            if not any(
                _looks_stateful_name(l) or _looks_stateful_name(r)
                or l in none_inited or r in none_inited
                for l, r in is_pairs
            ):
                continue

            # Fix: use child.body only — not ast.walk(child) which would
            # also walk elif/else branches and produce spurious return roots.
            returned = {
                _ast_root_name(stmt.value)
                for stmt in child.body
                if isinstance(stmt, ast.Return) and stmt.value is not None
            } - {None}

            if not any(
                (_looks_stateful_name(r) and r in stateful_stored)
                or (r in none_inited and r in assigned_in_node)
                for r in returned
            ):
                continue

            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": (
                    f"{entrypoint_name} returns saved output on object-identity "
                    f"match with previous call input"
                ),
            }]

        # Tertiary pass: identity+version replay via dict cache.
        # Pattern: cached = _cache.get(key)
        #          if cached is not None:
        #              ref, ver, result = cached   (or ref = cached[0])
        #              if ref is <input> and <input>._version == ver:
        #                  return result
        # This catches dict-based replay where identity and ._version guard
        # a stored result, but variables are local (not module-level stateful).
        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            # Look for both `is` and `._version` / `== ` in the test or
            # nested if-tests within the body
            all_tests = [child.test]
            for inner_stmt in child.body:
                if isinstance(inner_stmt, ast.If):
                    all_tests.append(inner_stmt.test)

            has_is = False
            has_version = False
            for test_node in all_tests:
                for cmp in ast.walk(test_node):
                    if not isinstance(cmp, ast.Compare):
                        continue
                    for op in cmp.ops:
                        if isinstance(op, ast.Is):
                            has_is = True
                    if _has_ver(cmp):
                        has_version = True

            if not (has_is and has_version):
                continue

            # Check the deepest if-body for a return without real compute
            target_body = child.body
            for inner_stmt in child.body:
                if isinstance(inner_stmt, ast.If):
                    target_body = inner_stmt.body
                    break

            if _body_has_calls(target_body):
                continue

            has_return = any(
                isinstance(s, ast.Return) and s.value is not None
                for s in target_body
            )
            if not has_return:
                continue

            return [{
                "pattern": "LAST_CALL_REPLAY",
                "severity": "critical",
                "evidence": (
                    f"{entrypoint_name} returns cached output guarded by "
                    f"object identity + tensor version check"
                ),
            }]

    return []


def detect_first_call_state_replay(code: str | SubmissionFacts) -> list[dict]:
    """Detect first-call/sentinel and captured-state output replay."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    class_none_attrs: set[tuple[str, str]] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        inherited = {
            (stmt.name, attr)
            for base in stmt.bases
            if isinstance(base, ast.Name)
            for cls, attr in class_none_attrs
            if cls == base.id
        }
        class_none_attrs.update(inherited)
        for child in stmt.body:
            if not isinstance(child, ast.Assign) or not _expr_is_none(child.value):
                continue
            for target in child.targets:
                if isinstance(target, ast.Name):
                    class_none_attrs.add((stmt.name, target.id))

    def slot_key(expr: ast.AST | None) -> Optional[str]:
        if expr is None:
            return None
        try:
            return ast.unparse(expr)
        except Exception:
            return ast.dump(expr)

    def is_slot_expr(expr: ast.AST | None, captured_roots: set[str], none_slots: set[str]) -> bool:
        root = _ast_root_name(expr)
        if root in none_slots:
            return isinstance(expr, ast.Name)
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name) and (expr.value.id, expr.attr) in class_none_attrs:
                return True
            return root in captured_roots
        if isinstance(expr, ast.Subscript):
            return root in captured_roots or root in none_slots
        return False

    def input_derived(expr: ast.AST | None, tainted: set[str], input_names: set[str]) -> bool:
        names = _expr_names(expr)
        return bool(names & tainted) or _expr_contains_input_derived_call(expr, input_names)

    def local_target_names(target: ast.AST | None) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for elt in target.elts:
                names.update(local_target_names(elt))
            return names
        if isinstance(target, ast.Starred):
            return local_target_names(target.value)
        return set()

    for fn in _entrypoint_function_nodes(facts):
        input_names = _function_input_names(fn)
        if not input_names:
            continue

        global_names = {
            name
            for stmt in fn.body
            if isinstance(stmt, ast.Global)
            for name in stmt.names
        }
        local_names = {
            name
            for stmt in _iter_non_nested_nodes(fn)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With))
            for name in (
                local_target_names(stmt.target) if isinstance(stmt, (ast.AnnAssign, ast.AugAssign, ast.For))
                else set().union(*(local_target_names(t) for t in stmt.targets)) if isinstance(stmt, ast.Assign)
                else set()
            )
        } - global_names
        captured_roots = {
            name
            for name in _expr_names(fn)
            if name not in input_names and name not in local_names
        }
        none_slots = set(facts._none_inited) | {
            name for name in captured_roots if name in facts._none_inited
        }

        tainted = set(input_names)
        aliases: dict[str, str] = {}
        stored_slots: set[str] = set()
        returned_slots: set[str] = set()

        for stmt in _iter_non_nested_nodes(fn):
            if isinstance(stmt, ast.Assign):
                if input_derived(stmt.value, tainted, input_names):
                    for target in stmt.targets:
                        tainted.update(_target_names(target))
                        key = slot_key(target) if is_slot_expr(target, captured_roots, none_slots) else None
                        if key:
                            stored_slots.add(key)
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and is_slot_expr(stmt.value, captured_roots, none_slots):
                        key = slot_key(stmt.value)
                        if key:
                            aliases[target.id] = key
                for target in stmt.targets:
                    if is_slot_expr(target, captured_roots, none_slots) and input_derived(stmt.value, tainted, input_names):
                        key = slot_key(target)
                        if key:
                            stored_slots.add(key)
            elif isinstance(stmt, ast.AnnAssign):
                if input_derived(stmt.value, tainted, input_names):
                    tainted.update(_target_names(stmt.target))
                    key = slot_key(stmt.target) if is_slot_expr(stmt.target, captured_roots, none_slots) else None
                    if key:
                        stored_slots.add(key)
            elif (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and stmt.func.attr in {"append", "extend", "insert", "update", "setdefault", "add"}
                and _ast_root_name(stmt.func.value) in captured_roots
                and any(input_derived(arg, tainted, input_names) for arg in stmt.args)
            ):
                key = slot_key(stmt.func.value)
                if key:
                    stored_slots.add(key)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                key = aliases.get(stmt.value.id) if isinstance(stmt.value, ast.Name) else None
                if key is None and is_slot_expr(stmt.value, captured_roots, none_slots):
                    key = slot_key(stmt.value)
                if key:
                    returned_slots.add(key)

        for stmt in _iter_non_nested_nodes(fn):
            if not isinstance(stmt, ast.If):
                continue

            # First-call branch: if slot is None: slot = input_derived; return slot
            body_return_slots: set[str] = set()
            body_store_slots: set[str] = set()
            for inner in stmt.body:
                if isinstance(inner, ast.Assign):
                    if not input_derived(inner.value, tainted, input_names):
                        continue
                    for target in inner.targets:
                        if is_slot_expr(target, captured_roots, none_slots):
                            key = slot_key(target)
                            if key:
                                body_store_slots.add(key)
                elif isinstance(inner, ast.Return) and inner.value is not None:
                    key = aliases.get(inner.value.id) if isinstance(inner.value, ast.Name) else None
                    if key is None and is_slot_expr(inner.value, captured_roots, none_slots):
                        key = slot_key(inner.value)
                    if key:
                        body_return_slots.add(key)
            if body_store_slots & body_return_slots:
                return [{
                    "pattern": "LAST_CALL_REPLAY",
                    "severity": "critical",
                    "evidence": f"{entrypoint_name} lazily stores and returns input-derived output state",
                }]
            if body_store_slots:
                for idx, top_stmt in enumerate(fn.body):
                    if top_stmt is not stmt:
                        continue
                    for follow in fn.body[idx + 1:]:
                        if isinstance(follow, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            continue
                        if any(isinstance(node, ast.Call) for node in ast.walk(follow)):
                            break
                        if isinstance(follow, ast.Return) and follow.value is not None:
                            key = aliases.get(follow.value.id) if isinstance(follow.value, ast.Name) else None
                            if key is None and is_slot_expr(follow.value, captured_roots, none_slots):
                                key = slot_key(follow.value)
                            if key in body_store_slots:
                                return [{
                                    "pattern": "LAST_CALL_REPLAY",
                                    "severity": "critical",
                                    "evidence": f"{entrypoint_name} returns first-call input-derived output state",
                                }]
                            break
                    break

            # Replay branch: if state/flag says ready: return slot
            if _expr_names(stmt.test) & input_names:
                continue
            if _body_has_calls(stmt.body):
                continue
            fast_return_slots: set[str] = set()
            for inner in stmt.body:
                if not isinstance(inner, ast.Return) or inner.value is None:
                    continue
                key = aliases.get(inner.value.id) if isinstance(inner.value, ast.Name) else None
                if key is None and is_slot_expr(inner.value, captured_roots, none_slots):
                    key = slot_key(inner.value)
                if key:
                    fast_return_slots.add(key)
            if fast_return_slots & (stored_slots | returned_slots):
                return [{
                    "pattern": "LAST_CALL_REPLAY",
                    "severity": "critical",
                    "evidence": f"{entrypoint_name} returns captured state filled from prior input-derived output",
                }]

    return []


RE_OBJECT_ID_DATA = re.compile(r"\bdata_id\s*=\s*id\s*\(\s*data\s*\)")
RE_RESULT_BANK_SET = re.compile(r"_superbatch_results\s*\[\s*(?:did|data_id)\s*\]\s*=")
RE_RESULT_BANK_RETURN = re.compile(r"return\s+_superbatch_results\s*\[\s*data_id\s*\]")
RE_FAST_RESULT_CACHE_GET = re.compile(r"_fast_cache\.get\s*\(\s*data_id\s*\)")
RE_FAST_RESULT_CACHE_SET = re.compile(r"_fast_cache\s*\[\s*data_id\s*\]\s*=")
RE_RETURN_RESULT = re.compile(r"return\s+result\b")
RE_RESULT_LIST = re.compile(r"\bresult_list\b")
RE_LEARNING_DATA = re.compile(r"\b_learning_data\b")


def detect_result_bank_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: object-identity replay through learned result banks / super-batches.

    Targets kernels that key caches by `id(data)`, accumulate per-object result
    lists, and later return stored results or launch handles for those same
    objects instead of recomputing normally.
    """
    facts = ensure_submission_facts(code)
    active = facts.python_active
    entrypoint_name = entrypoint_label(facts.entrypoint_name)
    if not RE_OBJECT_ID_DATA.search(active):
        return []

    superbatch_replay = (
        RE_RESULT_BANK_SET.search(active)
        and RE_RESULT_BANK_RETURN.search(active)
        and RE_RESULT_LIST.search(active)
    )
    fast_cache_replay = (
        RE_FAST_RESULT_CACHE_GET.search(active)
        and RE_FAST_RESULT_CACHE_SET.search(active)
        and RE_RETURN_RESULT.search(active)
        and RE_LEARNING_DATA.search(active)
    )

    if superbatch_replay or fast_cache_replay:
        evidence_parts = [f"{entrypoint_name} keys replay state by id(data)"]
        if superbatch_replay:
            evidence_parts.append("returns _superbatch_results[data_id]")
        if fast_cache_replay:
            evidence_parts.append("returns cached result from _fast_cache[data_id]")
        return [{
            "pattern": "RESULT_BANK_REPLAY",
            "severity": "critical",
            "evidence": "; ".join(evidence_parts),
        }]
    return []


def detect_config_cache_exploit(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 7: config-keyed result caching inside the configured entrypoint.

    Detects: an entrypoint that looks up a cache on entry, returns the cached
    value WITHOUT calling any GPU kernel, and stores output into the cache
    before the final return.

    Distinguishes from legitimate workspace caching by requiring that the
    early-return path does NOT contain any function calls (a real exploit
    returns the cached tensor directly without computation).
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []
    entrypoint_name = entrypoint_label(facts.entrypoint_name)

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not is_entrypoint_name(node.name):
            continue

        def _lookup_source(expr: ast.AST | None, sources: dict[str, str]) -> Optional[str]:
            root = _ast_root_name(expr)
            if root is None:
                return None
            return sources.get(root, root)

        cache_reads: dict[str, str] = {}
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if len(child.targets) != 1 or not isinstance(child.targets[0], ast.Name):
                continue
            source = None
            if (isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Attribute)
                    and child.value.func.attr == "get"):
                source = _ast_root_name(child.value.func.value)
            elif isinstance(child.value, ast.Subscript):
                source = _ast_root_name(child.value.value)
            if source:
                cache_reads[child.targets[0].id] = source

        returned_roots = {
            _lookup_source(sub.value, cache_reads)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Return)
        } - {None}
        shortcircuit_containers: set[str] = set()
        store_containers: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.If):
                if not any(isinstance(stmt, ast.Return) for stmt in child.body):
                    continue
                if _body_has_calls(child.body):
                    continue
                return_roots = {
                    _lookup_source(stmt.value, cache_reads)
                    for stmt in child.body
                    if isinstance(stmt, ast.Return)
                } - {None}
                if not return_roots:
                    continue
                test_roots = {
                    _lookup_source(sub, cache_reads)
                    for sub in ast.walk(child.test)
                    if isinstance(sub, (ast.Name, ast.Attribute, ast.Subscript))
                } - {None}
                shortcircuit_containers.update(return_roots & test_roots)

            if isinstance(child, ast.Try):
                if _body_has_calls(child.body):
                    continue
                has_keyerror = any(
                    isinstance(handler.type, ast.Name) and handler.type.id == "KeyError"
                    for handler in child.handlers
                    if handler.type is not None
                )
                if not has_keyerror:
                    continue
                local_sources = dict(cache_reads)
                for stmt in child.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        continue
                    source = None
                    if isinstance(stmt.value, ast.Subscript):
                        source = _ast_root_name(stmt.value.value)
                    elif (isinstance(stmt.value, ast.Call)
                          and isinstance(stmt.value.func, ast.Attribute)
                          and stmt.value.func.attr == "get"):
                        source = _ast_root_name(stmt.value.func.value)
                    if source:
                        local_sources[stmt.targets[0].id] = source
                for stmt in child.body:
                    if isinstance(stmt, ast.Return):
                        resolved = _lookup_source(stmt.value, local_sources)
                        if resolved:
                            shortcircuit_containers.add(resolved)

            if isinstance(child, ast.Assign):
                value_root = _lookup_source(child.value, cache_reads)
                if value_root not in returned_roots:
                    continue
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        target_root = _ast_root_name(target.value)
                        if target_root:
                            store_containers.add(target_root)

        if shortcircuit_containers & store_containers:
                matches.append({
                    "pattern": "CONFIG_CACHE_EXPLOIT",
                    "severity": "high",
                    "evidence": (
                        f"{entrypoint_name} short-circuits on cache hit + stores output "
                        "(config-keyed result caching)"
                    ),
                })

    return matches


def _is_obfuscated_exec(node: ast.Call) -> bool:
    """Return True when an exec/eval call wraps an encoded/compressed payload.

    Detects patterns like:
      exec(lzma.decompress(base64.b64decode(...)))
      exec(zlib.decompress(base64.b64decode(...)))
      eval(base64.b64decode(...))
    These hide exploit code that static analysis cannot see.
    """
    if not node.args:
        return False
    arg = node.args[0]
    # Walk the nested call chain looking for encoding/compression functions
    obfuscation_names = {
        "decompress", "b64decode", "b32decode", "b16decode",
        "urlsafe_b64decode", "decodebytes", "decodestring",
        "unhexlify", "a2b_hex", "a2b_base64",
    }
    obfuscation_modules = {"base64", "lzma", "zlib", "gzip", "bz2", "codecs", "binascii"}
    for inner in ast.walk(arg):
        if not isinstance(inner, ast.Call):
            continue
        if isinstance(inner.func, ast.Attribute):
            if inner.func.attr in obfuscation_names:
                return True
            if isinstance(inner.func.value, ast.Name) and inner.func.value.id in obfuscation_modules:
                return True
        elif isinstance(inner.func, ast.Name) and inner.func.id in obfuscation_names:
            return True
    return False


def _scope_has_obfuscation(scope: ast.AST) -> bool:
    """Return True if the scope contains decode/decompress calls (even not nested in exec)."""
    obfuscation_names = {
        "decompress", "b64decode", "b32decode", "b16decode",
        "urlsafe_b64decode", "decodebytes", "decodestring",
        "unhexlify", "a2b_hex", "a2b_base64",
    }
    obfuscation_modules = {"base64", "lzma", "zlib", "gzip", "bz2", "codecs", "binascii"}
    for n in ast.walk(scope):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute):
            if n.func.attr in obfuscation_names:
                return True
            if isinstance(n.func.value, ast.Name) and n.func.value.id in obfuscation_modules:
                return True
        if isinstance(n.func, ast.Name) and n.func.id in obfuscation_names:
            return True
    return False


def _decode_like_helpers(tree: ast.AST) -> set[str]:
    helpers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if _expr_has_decode_like_call(child):
                helpers.add(node.name)
                break
    return helpers


def detect_dynamic_execution(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 6: exec/eval/compile and dynamic import.

    Splits into two sub-patterns:
      OBFUSCATED_EXEC — exec/eval wrapping base64/lzma/zlib payloads (auto_filter)
      DYNAMIC_EXECUTION — plain exec/eval/compile/__import__ (telemetry)
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        active = facts.python_active
        aliases = set(re.findall(r'^\s*(\w+)\s*=\s*(?:exec|eval)\s*$', active, re.MULTILINE))
        if aliases and any(re.search(rf'\b{re.escape(alias)}\s*\(', active) for alias in aliases):
            if re.search(r'\bcustom_kernel\b|def\s+\w+\s*\(', active):
                return [{
                    "pattern": "OBFUSCATED_EXEC",
                    "severity": "critical",
                    "evidence": "aliased exec/eval in syntactically invalid source can hide submitted entrypoint code",
                }]
        return []

    matches = []
    seen: set[str] = set()
    exec_aliases = {"exec": "exec", "eval": "eval"}
    decode_helpers = _decode_like_helpers(tree)
    string_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        static_value = _static_string(node.value)
        if static_value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    string_aliases[target.id] = static_value
        if isinstance(node.value, ast.Name) and node.value.id in {"exec", "eval"}:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    exec_aliases[target.id] = node.value.id

    entrypoint_fns = _entrypoint_function_nodes(facts)

    def _call_inside_fn(fn: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call) -> bool:
        return any(inner is call for inner in _iter_non_nested_nodes(fn))

    def _expr_uses_namespace(expr: ast.AST | None, namespace_names: set[str]) -> bool:
        if expr is None:
            return False
        for node in ast.walk(expr):
            if isinstance(node, ast.Subscript) and _ast_root_name(node.value) in namespace_names:
                return True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Subscript) and _ast_root_name(node.func.value) in namespace_names:
                    return True
        return False

    def _entrypoint_exec_builds_return(call: ast.Call) -> bool:
        namespace_arg = call.args[1] if len(call.args) >= 2 else None
        namespace_names = {namespace_arg.id} if isinstance(namespace_arg, ast.Name) else set()
        payload = _static_string(call.args[0]) if call.args else None
        if payload is None and call.args and isinstance(call.args[0], ast.Name):
            payload = string_aliases.get(call.args[0].id)
        payload_builds_code = bool(payload and re.search(r'\bdef\s+\w+\s*\(|\blambda\b|\bimport\s+\w+', payload))

        for fn in entrypoint_fns:
            if not _call_inside_fn(fn, call):
                continue
            local_namespaces = set(namespace_names)
            for stmt in _iter_non_nested_nodes(fn):
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict):
                    for target in stmt.targets:
                        local_namespaces.update(_target_names(target))
            if local_namespaces and payload_builds_code:
                for stmt in _iter_non_nested_nodes(fn):
                    if isinstance(stmt, ast.Return) and _expr_uses_namespace(stmt.value, local_namespaces):
                        return True
        return False

    # Pre-scan: check if any function scope contains both exec/eval AND
    # decode/decompress calls (catches split-variable patterns like:
    #   b = unhexlify(hex_str); s = decompress(b); exec(compile(s, ...)))
    obfuscated_scopes: set[int] = set()
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        has_exec = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in exec_aliases
            for n in ast.walk(scope)
        )
        if has_exec and (_scope_has_obfuscation(scope) or _expr_has_decode_like_call(scope, decode_helpers)):
            obfuscated_scopes.add(id(scope))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = f"{node.func.value.id}.{node.func.attr}"

        sink_name = exec_aliases.get(func_name, func_name)
        if sink_name in ("exec", "eval") and func_name not in seen:
            is_obfuscated = _is_obfuscated_exec(node) or (
                bool(node.args) and _expr_has_decode_like_call(node.args[0], decode_helpers)
            )
            builds_entrypoint_return = _entrypoint_exec_builds_return(node)
            # Also check if this exec lives in a scope that has decode/decompress
            if not is_obfuscated:
                for scope in ast.walk(tree):
                    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                        continue
                    if id(scope) in obfuscated_scopes:
                        # Verify this exec node is inside this scope
                        for inner in ast.walk(scope):
                            if inner is node:
                                is_obfuscated = True
                                break
                    if is_obfuscated:
                        break
            if is_obfuscated or builds_entrypoint_return:
                key = f"obfuscated_{func_name}"
                if key not in seen:
                    seen.add(key)
                    evidence = (
                        f"{func_name}() builds and returns generated callable inside {entrypoint_label(facts.entrypoint_name)}"
                        if builds_entrypoint_return and not is_obfuscated
                        else f"{func_name}() with encoded/decoded payload (hides exploit code)"
                    )
                    matches.append({
                        "pattern": "OBFUSCATED_EXEC",
                        "severity": "critical",
                        "evidence": evidence,
                    })
            else:
                seen.add(func_name)
                matches.append({
                    "pattern": "DYNAMIC_EXECUTION",
                    "severity": "high",
                    "evidence": f"{func_name}() call (could hide exploit code)",
                })
        elif func_name in ("compile", "__import__") and func_name not in seen:
            seen.add(func_name)
            matches.append({
                "pattern": "DYNAMIC_EXECUTION",
                "severity": "high",
                "evidence": f"{func_name}() call (could hide exploit code)",
            })
        elif func_name == "importlib.import_module" and func_name not in seen:
            seen.add(func_name)
            matches.append({
                "pattern": "DYNAMIC_EXECUTION",
                "severity": "high",
                "evidence": "importlib.import_module() (dynamic module loading)",
            })
        elif func_name == "importlib.reload" and func_name not in seen:
            seen.add(func_name)
            matches.append({
                "pattern": "MODULE_RELOAD",
                "severity": "high",
                "evidence": "importlib.reload() (module state reset/manipulation)",
            })

    return matches


def detect_thread_injection(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 8: background thread/process to offload computation.

    The kernel spawns a background thread that performs the real work while
    the main thread returns an empty/placeholder tensor immediately.  By the
    time the correctness check runs the thread has finished.

    Ref: reference exploit note: thread injection
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    # Collect all imports to resolve aliases (e.g. `import threading as t`)
    threading_aliases: set[str] = {"threading"}
    mp_aliases: set[str] = {"multiprocessing"}
    futures_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name == "threading":
                    threading_aliases.add(name)
                elif alias.name == "multiprocessing":
                    mp_aliases.add(name)
                elif alias.name == "concurrent.futures":
                    futures_aliases.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "threading":
                for alias in node.names:
                    name = alias.asname or alias.name
                    if alias.name == "Thread":
                        threading_aliases.add(name)
            elif node.module == "multiprocessing":
                for alias in node.names:
                    name = alias.asname or alias.name
                    if alias.name == "Process":
                        mp_aliases.add(name)
            elif node.module and node.module.startswith("concurrent"):
                for alias in node.names:
                    futures_aliases.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # threading.Thread(...) or Thread(...)
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in threading_aliases
                    and node.func.attr == "Thread"):
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": f"{node.func.value.id}.Thread() — background thread offloads compute outside timing",
                })
            elif (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in mp_aliases
                    and node.func.attr == "Process"):
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": f"{node.func.value.id}.Process() — background process offloads compute",
                })
            elif (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in futures_aliases
                    and node.func.attr in ("ThreadPoolExecutor", "ProcessPoolExecutor")):
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": f"concurrent.futures.{node.func.attr}() — executor pool offloads compute",
                })
        elif isinstance(node.func, ast.Name):
            if node.func.id == "Thread" and "Thread" in threading_aliases:
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": "Thread() — background thread offloads compute outside timing",
                })
            elif node.func.id == "Process" and "Process" in mp_aliases:
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": "Process() — background process offloads compute",
                })

    # Deduplicate
    seen = set()
    deduped = []
    for m in matches:
        key = m["evidence"]
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def detect_lazy_tensor(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 9: lazy tensor subclass that defers computation.

    The kernel returns a torch.Tensor subclass whose data is not materialized.
    Computation is deferred to __eq__, __torch_function__, or similar hooks
    that run during the correctness check rather than during timing.

    Ref: reference exploit note: lazy evaluation
    """
    facts = ensure_submission_facts(code)
    if facts.ast_tree is None:
        return []

    matches = []
    tensor_method_overrides: set[tuple[str, str]] = set()
    materialization_methods = {
        "double", "detach", "to", "cpu", "cuda", "float", "half", "bfloat16",
        "long", "int", "item", "numpy", "clone", "contiguous",
    }

    for node in facts._class_defs:

        # Check if class inherits from torch.Tensor
        is_tensor_subclass = False
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                if (isinstance(base.value, ast.Name)
                        and base.value.id == "torch" and base.attr == "Tensor"):
                    is_tensor_subclass = True
            elif isinstance(base, ast.Name) and base.id == "Tensor":
                is_tensor_subclass = True

        if not is_tensor_subclass:
            continue

        # Found a torch.Tensor subclass — check for suspicious methods
        suspicious_methods = []
        has_make_subclass = False

        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                if child.name in ("__eq__", "__torch_function__", "__torch_dispatch__"):
                    suspicious_methods.append(child.name)
            # _make_subclass call
            if isinstance(child, ast.Attribute) and child.attr == "_make_subclass":
                has_make_subclass = True

        if suspicious_methods or has_make_subclass:
            evidence_parts = [f"torch.Tensor subclass '{node.name}'"]
            if has_make_subclass:
                evidence_parts.append("_make_subclass()")
            if suspicious_methods:
                evidence_parts.append(f"deferred hooks: {', '.join(suspicious_methods)}")
            matches.append({
                "pattern": "LAZY_TENSOR",
                "severity": "critical",
                "evidence": " + ".join(evidence_parts) + " — computation deferred past timing",
            })
        else:
            # Tensor subclass without known hooks — still suspicious
            matches.append({
                "pattern": "LAZY_TENSOR",
                "severity": "high",
                "evidence": f"torch.Tensor subclass '{node.name}' — may defer computation",
            })

    for node in ast.walk(facts.ast_tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and target.attr in materialization_methods
                and isinstance(target.value, ast.Name)
            ):
                continue
            # Per-instance overrides such as `tensor.double = f` or
            # `tensor.detach = f` can make check-time materialization execute
            # work that was skipped inside the timed region. Normal Tensor
            # subclasses are handled above; this catches the plain-tensor case.
            if target.attr in {"double", "detach"}:
                tensor_method_overrides.add((target.value.id, target.attr))

    if tensor_method_overrides:
        evidence = ", ".join(
            f"{name}.{method}" for name, method in sorted(tensor_method_overrides)
        )
        matches.append({
            "pattern": "TENSOR_INSTANCE_METHOD_OVERRIDE",
            "severity": "critical",
            "evidence": f"plain Tensor instance materialization methods overridden: {evidence}",
        })

    return matches


def detect_precision_downgrade(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 10: silent precision downgrade for speed.

    The kernel casts inputs to lower precision (fp16/bf16), computes faster,
    then casts back to the original dtype.  This cheats the benchmark while
    potentially passing allclose with loose tolerance.

    Ref: reference exploit note: precision downgrading
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []

    # Look inside the configured entrypoint (or the whole file if none is found)
    target_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_entrypoint_name(node.name)
    ]
    if not target_funcs:
        target_funcs = [tree]

    for func_node in target_funcs:
        has_downcast = False
        has_upcast = False
        downcast_evidence = []

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            # .half(), .bfloat16(), .float16() method calls
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("half", "bfloat16", "float16"):
                    has_downcast = True
                    downcast_evidence.append(f".{attr}()")
                elif attr in ("float", "double", "float32"):
                    has_upcast = True
                # .to(torch.float16), .to(dtype=torch.bfloat16)
                elif attr == "to":
                    for arg in node.args:
                        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                            if arg.value.id == "torch" and arg.attr in ("float16", "bfloat16", "half"):
                                has_downcast = True
                                downcast_evidence.append(f".to(torch.{arg.attr})")
                            elif arg.value.id == "torch" and arg.attr in ("float32", "float64"):
                                has_upcast = True
                    for kw in node.keywords:
                        if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute):
                            if (isinstance(kw.value.value, ast.Name)
                                    and kw.value.value.id == "torch"
                                    and kw.value.attr in ("float16", "bfloat16", "half")):
                                has_downcast = True
                                downcast_evidence.append(f".to(dtype=torch.{kw.value.attr})")
                            elif (isinstance(kw.value.value, ast.Name)
                                    and kw.value.value.id == "torch"
                                    and kw.value.attr in ("float32", "float64")):
                                has_upcast = True

        if has_downcast and has_upcast:
            evidence = ", ".join(dict.fromkeys(downcast_evidence))
            matches.append({
                "pattern": "PRECISION_DOWNGRADE",
                "severity": "medium",
                "evidence": f"Downcast ({evidence}) then upcast back — may cheat benchmark via lower precision",
            })

    return matches


# ---------------------------------------------------------------------------
# Score anomaly detection
# ---------------------------------------------------------------------------

def _collect_scores(metadata: Optional[dict]) -> tuple[list[float], Optional[float], Optional[float]]:
    """Extract all available scores from metadata into a unified list."""
    if not metadata:
        return [], None, None
    all_scores: list[float] = []
    score = metadata.get("score")
    if score is not None and isinstance(score, (int, float)):
        all_scores.append(score)
    for s in metadata.get("scores", []):
        if isinstance(s, (int, float)):
            all_scores.append(s)
    improved_score = metadata.get("improved_score")
    baseline_score = metadata.get("baseline_score")
    if improved_score is not None and isinstance(improved_score, (int, float)):
        all_scores.append(improved_score)
    return all_scores, improved_score, baseline_score


# ---------------------------------------------------------------------------
# Roofline physics floor
# ---------------------------------------------------------------------------
# The theoretical minimum execution time for a GEMM on a given GPU is bounded
# by: t_min = max(FLOPs / Peak_FLOPS, Bytes / Peak_BW, kernel_launch_overhead)
#
# Any benchmark score (geometric mean of execution times) below this floor is
# physically impossible regardless of how optimized the kernel is.
#
# Score in KernelBot = geometric_mean(benchmark_times_in_seconds).

# GPU specifications — conservative peak values per hardware platform.
GPU_PROFILES: dict[str, dict[str, Any]] = {
    "B200": {
        "name": "B200",
        "fp4_tflops": 4500,        # FP4 dense tensor core TFLOPS
        "fp16_tflops": 2250,       # FP16 tensor core TFLOPS
        "hbm_bw_tb_s": 8.0,        # HBM3e bandwidth in TB/s
        "launch_overhead_ns": 3000, # ~3µs CUDA launch
        "scale_block_size": 16,     # NVFP4: 16 elements per scale factor
    },
    "MI355X": {
        "name": "MI355X",
        "fp4_tflops": 20100,       # MXFP4 dense TFLOPS (CDNA4)
        "fp16_tflops": 5000,       # FP16 TFLOPS
        "hbm_bw_tb_s": 8.0,        # HBM3E bandwidth in TB/s
        "launch_overhead_ns": 4000, # ~4µs HIP launch
        "scale_block_size": 32,     # OCP MX: 32 elements per E8M0 scale
    },
}
BASE_GPU_SPECS: dict[str, Any] = copy.deepcopy(GPU_PROFILES["B200"])

# Per-problem benchmark specifications.
# Each entry maps a problem name to a list of benchmark shapes + the GPU profile.
# Shapes are dicts with keys: m, k, n (optional, default 1), l (optional, default 1),
# g (optional, for group gemm — value is a list of per-group shapes).
# multiplier: how many GEMMs per benchmark case (e.g. 2 for dual_gemm).
BASE_BENCHMARK_SPECS: dict[str, list[dict]] = {
    # ---- NVIDIA B200 competitions ----
    "nvfp4_gemv": [
        {"m": 7168, "k": 16384, "n": 1, "l": 1},
        {"m": 4096, "k": 7168,  "n": 1, "l": 8},
        {"m": 7168, "k": 2048,  "n": 1, "l": 4},
    ],
    "nvfp4_gemm": [
        {"m": 128, "k": 16384, "n": 7168, "l": 1},
        {"m": 128, "k": 7168,  "n": 4096, "l": 1},
        {"m": 128, "k": 2048,  "n": 7168, "l": 1},
    ],
    "nvfp4_dual_gemm": [
        {"m": 256, "k": 7168, "n": 4096, "l": 1, "multiplier": 2},
        {"m": 512, "k": 7168, "n": 4096, "l": 1, "multiplier": 2},
        {"m": 256, "k": 4096, "n": 3072, "l": 1, "multiplier": 2},
        {"m": 512, "k": 7168, "n": 3072, "l": 1, "multiplier": 2},
    ],
    "modal_nvfp4_dual_gemm": [
        {"m": 256, "k": 7168, "n": 4096, "l": 1, "multiplier": 2},
        {"m": 512, "k": 7168, "n": 4096, "l": 1, "multiplier": 2},
        {"m": 256, "k": 4096, "n": 3072, "l": 1, "multiplier": 2},
        {"m": 512, "k": 7168, "n": 3072, "l": 1, "multiplier": 2},
    ],
    "nvfp4_group_gemm": [
        {"groups": [
            {"m": 80, "k": 7168, "n": 4096}, {"m": 176, "k": 7168, "n": 4096},
            {"m": 128, "k": 7168, "n": 4096}, {"m": 72, "k": 7168, "n": 4096},
            {"m": 64, "k": 7168, "n": 4096}, {"m": 248, "k": 7168, "n": 4096},
            {"m": 96, "k": 7168, "n": 4096}, {"m": 160, "k": 7168, "n": 4096},
        ]},
        {"groups": [
            {"m": 40, "k": 2048, "n": 7168}, {"m": 76, "k": 2048, "n": 7168},
            {"m": 168, "k": 2048, "n": 7168}, {"m": 72, "k": 2048, "n": 7168},
            {"m": 164, "k": 2048, "n": 7168}, {"m": 148, "k": 2048, "n": 7168},
            {"m": 196, "k": 2048, "n": 7168}, {"m": 160, "k": 2048, "n": 7168},
        ]},
        {"groups": [
            {"m": 192, "k": 4096, "n": 3072}, {"m": 320, "k": 4096, "n": 3072},
        ]},
        {"groups": [
            {"m": 128, "k": 1536, "n": 4096}, {"m": 384, "k": 1536, "n": 4096},
        ]},
    ],
    # ---- AMD MI355X competitions (DeepSeek-R1 dimensions) ----
    # Shapes from gpu-mode/reference-kernels problems/amd_202602/*/task.yml
    "amd-mxfp4-mm": [
        # bf16 A [M,K] x MXFP4 B [N,K] -> MXFP4 GEMM -> bf16 C [M,N]
        {"m": 4,   "k": 512,  "n": 2880, "gpu": "MI355X"},
        {"m": 16,  "k": 7168, "n": 2112, "gpu": "MI355X"},
        {"m": 32,  "k": 512,  "n": 4096, "gpu": "MI355X"},
        {"m": 32,  "k": 512,  "n": 2880, "gpu": "MI355X"},
        {"m": 64,  "k": 2048, "n": 7168, "gpu": "MI355X"},
        {"m": 256, "k": 1536, "n": 3072, "gpu": "MI355X"},
    ],
    # MoE: fused gate_up + SwiGLU + down GEMM with MXFP4.
    # Each benchmark case runs top-9 experts (8 routed + 1 shared).
    # Floor approximation: 2 GEMMs per expert × top_k active experts.
    # Stage 1: [bs, d_hidden] × [2*d_expert, d_hidden] (gate+up)
    # Stage 2: [bs, d_expert] × [d_hidden, d_expert] (down)
    "amd-moe-mxfp4": [
        # TP=8: 257 experts, d_expert=256
        {"m": 16,  "k": 7168, "n": 512,  "gpu": "MI355X", "multiplier": 9},  # bs=16, 9 active
        {"m": 128, "k": 7168, "n": 512,  "gpu": "MI355X", "multiplier": 9},
        {"m": 512, "k": 7168, "n": 512,  "gpu": "MI355X", "multiplier": 9},
        # TP=4: 33 experts, d_expert=512
        {"m": 16,  "k": 7168, "n": 1024, "gpu": "MI355X", "multiplier": 9},
        {"m": 128, "k": 7168, "n": 1024, "gpu": "MI355X", "multiplier": 9},
        {"m": 512, "k": 7168, "n": 1024, "gpu": "MI355X", "multiplier": 9},
        # EP: 33 experts, d_expert=2048
        {"m": 512, "k": 7168, "n": 4096, "gpu": "MI355X", "multiplier": 9},
    ],
    # MLA decode: attention with compressed KV cache.
    # Memory-bound: loads KV cache (total_kv × 1 × 576 × element_size).
    # Compute: 2 × batch × kv_len × qk_head_dim × num_heads (Q×K^T + softmax×V).
    # Using FP16 compute for attention (not FP4).
    "amd-mixed-mla": [
        # attention floor ≈ max(kv_cache_load_time, attention_compute_time)
        # kv_cache in mxfp4: 0.5 bytes/element, 576 dims → 288 bytes/row
        {"batch": 4,   "kv_len": 1024, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 4,   "kv_len": 8192, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 32,  "kv_len": 1024, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 32,  "kv_len": 8192, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 64,  "kv_len": 1024, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 64,  "kv_len": 8192, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 256, "kv_len": 1024, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
        {"batch": 256, "kv_len": 8192, "heads": 16, "head_dim": 576, "v_dim": 512, "gpu": "MI355X"},
    ],
}

# Problem → GPU profile mapping (for problems that don't embed "gpu" in shapes)
PROBLEM_GPU_MAP: dict[str, str] = {
    "nvfp4_gemv": "B200",
    "nvfp4_gemm": "B200",
    "nvfp4_dual_gemm": "B200",
    "modal_nvfp4_dual_gemm": "B200",
    "nvfp4_group_gemm": "B200",
    "amd-mxfp4-mm": "MI355X",
    "amd-moe-mxfp4": "MI355X",
    "amd-mixed-mla": "MI355X",
}

GPU_SPECS: dict[str, Any] = copy.deepcopy(BASE_GPU_SPECS)
BENCHMARK_SPECS: dict[str, list[dict]] = copy.deepcopy(BASE_BENCHMARK_SPECS)


def _gemm_floor_ns(m: int, k: int, n: int, l: int = 1, multiplier: int = 1,
                    gpu: Optional[dict] = None) -> float:
    """Roofline lower bound in nanoseconds for a single GEMM(m, k, n) x l batches."""
    specs = gpu or GPU_SPECS
    peak_flops = specs["fp4_tflops"] * 1e12
    peak_bw = specs["hbm_bw_tb_s"] * 1e12
    launch_ns = float(specs["launch_overhead_ns"])
    scale_block = specs.get("scale_block_size", 16)

    flops = 2.0 * m * k * n * l * multiplier
    compute_ns = flops / peak_flops * 1e9

    # FP4 inputs (0.5B), FP16/BF16 output (2B), block scales
    input_bytes = (m * k + k * n) * l * 0.5
    output_bytes = m * n * l * 2.0
    scale_bytes = (m * k // scale_block + k * n // scale_block) * l * 1.0
    total_bytes = (input_bytes + output_bytes + scale_bytes) * multiplier
    memory_ns = total_bytes / peak_bw * 1e9

    return max(compute_ns, memory_ns, launch_ns)


def _resolve_gpu_specs(problem_name: str, spec: Optional[dict] = None) -> dict:
    """Resolve GPU specs for a problem, checking spec-level override first."""
    if spec and "gpu" in spec:
        profile_name = spec["gpu"]
        if profile_name in GPU_PROFILES:
            return GPU_PROFILES[profile_name]
    profile_name = PROBLEM_GPU_MAP.get(problem_name, "")
    if profile_name in GPU_PROFILES:
        return GPU_PROFILES[profile_name]
    return GPU_SPECS  # fallback to active global specs


def compute_physics_floor(problem_name: str) -> Optional[float]:
    """Compute the roofline physics floor score for a benchmark problem.

    Returns the geometric mean of per-benchmark-case minimum times in seconds,
    which is the absolute minimum achievable score.  Returns None if the problem
    has no registered benchmark specs.
    """
    specs = BENCHMARK_SPECS.get(problem_name)
    if not specs:
        return None

    import math as _math
    floors_ns: list[float] = []
    for spec in specs:
        gpu = _resolve_gpu_specs(problem_name, spec)
        if "groups" in spec:
            group_total = sum(
                _gemm_floor_ns(g["m"], g["k"], g["n"], gpu=gpu)
                for g in spec["groups"]
            )
            floors_ns.append(group_total)
        elif "batch" in spec and "kv_len" in spec:
            # MLA attention: memory-bound KV cache load + attention compute
            batch = spec["batch"]
            kv_len = spec["kv_len"]
            heads = spec.get("heads", 16)
            head_dim = spec.get("head_dim", 576)
            v_dim = spec.get("v_dim", 512)
            peak_bw = gpu["hbm_bw_tb_s"] * 1e12
            peak_flops = gpu["fp16_tflops"] * 1e12  # attention in FP16
            launch_ns = float(gpu["launch_overhead_ns"])
            # KV cache load: total_kv × 1 × head_dim × 0.5 bytes (mxfp4)
            total_kv = batch * kv_len
            kv_bytes = total_kv * head_dim * 0.5  # mxfp4 compressed
            # Q load + output write
            q_bytes = batch * heads * head_dim * 2  # bf16
            out_bytes = batch * heads * v_dim * 2   # bf16
            total_bytes = kv_bytes + q_bytes + out_bytes
            memory_ns = total_bytes / peak_bw * 1e9
            # Attention FLOPs: batch × heads × (2 × kv_len × head_dim + 2 × kv_len × v_dim)
            attn_flops = batch * heads * kv_len * (2.0 * head_dim + 2.0 * v_dim)
            compute_ns = attn_flops / peak_flops * 1e9
            floors_ns.append(max(compute_ns, memory_ns, launch_ns))
        else:
            m = spec.get("m", 1)
            k = spec.get("k", 1)
            n = spec.get("n", 1)
            l = spec.get("l", 1)
            mult = spec.get("multiplier", 1)
            floors_ns.append(_gemm_floor_ns(m, k, n, l, mult, gpu=gpu))

    if not floors_ns:
        return None

    log_sum = sum(_math.log(f * 1e-9) for f in floors_ns)
    return _math.exp(log_sum / len(floors_ns))


def detect_score_anomaly(metadata: Optional[dict]) -> list[dict]:
    """Score-based anomaly detection.

    Works across all input modes:
    - parquet: metadata has "score" (float)
    - nvidia_archive: metadata has "scores" (list of floats)
    - jsonl: metadata has "improved_score" / "baseline_score"

    Includes roofline physics floor check when problem_name and benchmark
    specs are available.
    """
    if not metadata or not SCORE_CONFIG.get("enabled", True):
        return []
    matches = []

    all_scores, improved_score, baseline_score = _collect_scores(metadata)

    # --- Physics floor check (roofline model) ---
    # This uses per-problem benchmark shapes + GPU specs to compute the
    # absolute minimum achievable score.  Anything below is mathematically
    # impossible regardless of kernel quality.
    problem_name = metadata.get("problem_name") or metadata.get("problem")
    if problem_name and all_scores:
        physics_floor = compute_physics_floor(problem_name)
        if physics_floor is not None:
            for s in all_scores:
                if not isinstance(s, (int, float)) or s <= 0:
                    continue
                if s < physics_floor:
                    matches.append({
                        "pattern": "SCORE_PHYSICS_FLOOR",
                        "severity": "critical",
                        "evidence": (
                            f"Score {s:.2e}s is below the roofline physics floor "
                            f"({physics_floor:.2e}s) for {problem_name} — "
                            f"mathematically impossible on {GPU_SPECS.get('name', 'target GPU')}"
                        ),
                        "field": "metadata",
                    })
                    break

    # --- Fixed threshold checks ---
    impossible_below = float(SCORE_CONFIG["impossible_below"])
    suspect_floor_below = float(SCORE_CONFIG["suspect_floor_below"])
    broken_above = float(SCORE_CONFIG["broken_above"])

    # When a per-problem physics floor exists, use it instead of the static
    # threshold.  SCORE_PHYSICS_FLOOR (critical) catches scores below the
    # floor.  SCORE_SUSPECT_FLOOR only adds value for problems without a
    # physics floor — otherwise the floor itself is the boundary and scores
    # above it are achievable by definition.
    effective_suspect = suspect_floor_below
    pf = None
    if problem_name:
        pf = compute_physics_floor(problem_name)
        if pf is not None:
            # Physics floor is authoritative — disable the blunt threshold.
            # Anything above the floor is achievable; anything below is
            # already caught by SCORE_PHYSICS_FLOOR.
            effective_suspect = 0.0

    for s in all_scores:
        if not isinstance(s, (int, float)) or s <= 0:
            continue
        if s < impossible_below:
            matches.append({
                "pattern": "SCORE_IMPOSSIBLE",
                "severity": "critical",
                "evidence": f"Score {s:.2e} is physically impossible (<{impossible_below:.2e}s)",
            })
            break
        if s < effective_suspect:
            # Skip if already caught by SCORE_PHYSICS_FLOOR above
            if any(m["pattern"] == "SCORE_PHYSICS_FLOOR" for m in matches):
                break
            matches.append({
                "pattern": "SCORE_SUSPECT_FLOOR",
                "severity": "high",
                "evidence": (
                    f"Score {s:.2e}s is at/near timing floor (<{effective_suspect:.2e}s"
                    + (f", physics floor={pf:.2e}s" if problem_name and pf is not None else "")
                    + ")"
                ),
            })
            break
        if s > broken_above:
            matches.append({
                "pattern": "SCORE_BROKEN",
                "severity": "critical",
                "evidence": f"Score {s:.2e} exceeds configured upper bound ({broken_above:.2e}s)",
            })
            break

    # Extreme speedup check (JSONL mode)
    if (improved_score and baseline_score
            and isinstance(improved_score, (int, float))
            and isinstance(baseline_score, (int, float))
            and improved_score > 0):
        speedup = baseline_score / improved_score
        if speedup > float(SCORE_CONFIG["extreme_speedup_above"]):
            matches.append({
                "pattern": "SCORE_EXTREME_SPEEDUP",
                "severity": "high",
                "evidence": (
                    f"Speedup {speedup:.1f}x over baseline exceeds configured bound "
                    f"({float(SCORE_CONFIG['extreme_speedup_above']):.1f}x)"
                ),
            })

    return matches


# ---------------------------------------------------------------------------
# Rule policies and detector registry
# ---------------------------------------------------------------------------

AUTO_FILTER = "auto_filter"
SUSPICIOUS_ONLY = "suspicious"
TELEMETRY_ONLY = "low_confidence"


@dataclass(frozen=True)
class RulePolicy:
    rule_name: str
    technique_family: str
    evidence_tier: str
    max_outcome: str
    requires_companion_patterns: tuple[str, ...]
    mapped_positive_fixtures: tuple[str, ...]
    default_verdict: str


AMD_CODE_REPLACEMENT_FIXTURES = ("amd:514036",)
AMD_MODULE_MUTATION_FIXTURES = (
    "amd:514274", "amd:514288", "amd:514336", "amd:514350", "amd:514361",
    "amd:514378", "amd:514394", "amd:515019", "amd:515037", "amd:515048",
    "amd:515064", "amd:515082", "amd:515102", "amd:515123", "amd:519613",
    "amd:519659", "amd:519693", "amd:521294", "amd:521414",
)
AMD_CONFIG_CACHE_FIXTURES = ("amd:520921", "amd:521087", "amd:521134", "amd:521160")
AMD_MANUAL_HARNESS_FIXTURES = ("manual_review:515117",)
AMD_MANUAL_PRECOMPUTE_FIXTURES = ("manual_review:514553",)
AMD_MANUAL_POINTER_REPLAY_FIXTURES = ("manual_review:515367",)
AMD_MANUAL_LAST_CALL_REPLAY_FIXTURES = ()
AMD_MANUAL_FRAME_MUTATION_FIXTURES = ("manual_review:516235",)
TOP5_LOWCONF_RESULT_BANK_FIXTURES = ("top5low:506336",)
AMD_PACKAGE_TIMED_INPUT_REPLAY_FIXTURES = ()
AMD_PACKAGE_SHAPE_REPLAY_FIXTURES = ()
NVIDIA_ARCHIVE_TIMER_FIXTURES = (
    "nvidia_archive:108465", "nvidia_archive:108477", "nvidia_archive:108609", "nvidia_archive:108878",
    "nvidia_archive:121398", "nvidia_archive:121401",
)
NVIDIA_ARCHIVE_PRINT_FIXTURES = (
    "nvidia_archive:121398", "nvidia_archive:121401",
)
NVIDIA_ARCHIVE_EVALUATOR_FIXTURES = (
    "nvidia_archive:108465", "nvidia_archive:108477", "nvidia_archive:108609", "nvidia_archive:108878",
)


BASE_RULE_REGISTRY: dict[str, RulePolicy] = {
    "TIMER_MONKEYPATCH": RulePolicy(
        "TIMER_MONKEYPATCH", "timing_monkeypatch", "hard", AUTO_FILTER, (),
        NVIDIA_ARCHIVE_TIMER_FIXTURES, "keep",
    ),
    "FAKE_BENCHMARK_EMIT": RulePolicy(
        "FAKE_BENCHMARK_EMIT", "benchmark_output_injection", "hard", AUTO_FILTER, (),
        NVIDIA_ARCHIVE_PRINT_FIXTURES, "keep",
    ),
    "STDIO_REDIRECT": RulePolicy(
        "STDIO_REDIRECT", "stdio_compat", "telemetry", TELEMETRY_ONLY, (),
        (), "split",
    ),
    "EVALUATOR_EXPLOIT": RulePolicy(
        "EVALUATOR_EXPLOIT", "evaluator_state_mutation", "support", SUSPICIOUS_ONLY, (),
        NVIDIA_ARCHIVE_EVALUATOR_FIXTURES, "keep",
    ),
    "HARNESS_RUNTIME_PATCHING": RulePolicy(
        "HARNESS_RUNTIME_PATCHING", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_MANUAL_HARNESS_FIXTURES, "keep",
    ),
    "TORCH_VALIDATION_MONKEYPATCH": RulePolicy(
        "TORCH_VALIDATION_MONKEYPATCH", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "EVALUATOR_MODE_BRANCH": RulePolicy(
        "EVALUATOR_MODE_BRANCH", "evaluator_mode_branch", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "BENCHMARK_SHAPE_REPLAY": RulePolicy(
        "BENCHMARK_SHAPE_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SCALED_MM_REF": RulePolicy(
        "SCALED_MM_REF", "reference_path_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "split",
    ),
    "DECODE_MM_REF": RulePolicy(
        "DECODE_MM_REF", "reference_path_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "remove",
    ),
    "OUTPUT_REPLAY_CACHE": RulePolicy(
        "OUTPUT_REPLAY_CACHE", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_CONFIG_CACHE_FIXTURES, "rewrite",
    ),
    "LAST_CALL_REPLAY": RulePolicy(
        "LAST_CALL_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_MANUAL_LAST_CALL_REPLAY_FIXTURES, "keep",
    ),
    "SHAPE_OUTPUT_REPLAY": RulePolicy(
        "SHAPE_OUTPUT_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_PACKAGE_SHAPE_REPLAY_FIXTURES, "keep",
    ),
    "TIMED_INPUT_REPLAY": RulePolicy(
        "TIMED_INPUT_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_PACKAGE_TIMED_INPUT_REPLAY_FIXTURES, "keep",
    ),
    "PREPROCESS_CACHE": RulePolicy(
        "PREPROCESS_CACHE", "preprocess_cache", "telemetry", TELEMETRY_ONLY, (),
        (), "keep",
    ),
    "WORKSPACE_CACHE": RulePolicy(
        "WORKSPACE_CACHE", "workspace_cache", "telemetry", TELEMETRY_ONLY, (),
        (), "keep",
    ),
    "RUNNER_PLAN_CACHE": RulePolicy(
        "RUNNER_PLAN_CACHE", "runner_plan_cache", "telemetry", TELEMETRY_ONLY, (),
        (), "keep",
    ),
    "CUDA_GRAPH_REPLAY": RulePolicy(
        "CUDA_GRAPH_REPLAY", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "SILENT_FALLBACK": RulePolicy(
        "SILENT_FALLBACK", "reference_path_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "remove",
    ),
    "TRIVIAL_PROBE": RulePolicy(
        "TRIVIAL_PROBE", "low_signal", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "INPUT_PASSTHROUGH_OUTPUT": RulePolicy(
        "INPUT_PASSTHROUGH_OUTPUT", "fake_output", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "INPUT_REDUCTION_OUTPUT": RulePolicy(
        "INPUT_REDUCTION_OUTPUT", "fake_output", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SELF_MATMUL_OUTPUT": RulePolicy(
        "SELF_MATMUL_OUTPUT", "fake_output", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "TORCH_COMPILE_CACHE": RulePolicy(
        "TORCH_COMPILE_CACHE", "performance_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "CUDA_GRAPH_PYTHON": RulePolicy(
        "CUDA_GRAPH_PYTHON", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "HARDCODED_SHAPES": RulePolicy(
        "HARDCODED_SHAPES", "performance_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "BENCHMARK_UNSAFE_ALGO_DISPATCH": RulePolicy(
        "BENCHMARK_UNSAFE_ALGO_DISPATCH", "fixture_specialization", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "INPUT_STRUCTURE_TRUNCATION_DISPATCH": RulePolicy(
        "INPUT_STRUCTURE_TRUNCATION_DISPATCH", "fixture_specialization", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "UNSYNC_MULTISTREAM": RulePolicy(
        "UNSYNC_MULTISTREAM", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "CUDA_EVENT_DISABLE_TIMING": RulePolicy(
        "CUDA_EVENT_DISABLE_TIMING", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "TRUSTED_MODULE_IMPORT": RulePolicy(
        "TRUSTED_MODULE_IMPORT", "evaluator_state_support", "support", SUSPICIOUS_ONLY,
        ("MODULE_MUTATION", "GLOBALS_MUTATION", "CODE_REPLACEMENT", "EVALUATOR_EXPLOIT"),
        AMD_MODULE_MUTATION_FIXTURES + AMD_CODE_REPLACEMENT_FIXTURES, "downgrade",
    ),
    "MODULE_MUTATION": RulePolicy(
        "MODULE_MUTATION", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_MODULE_MUTATION_FIXTURES + AMD_CODE_REPLACEMENT_FIXTURES, "keep",
    ),
    "GLOBALS_MUTATION": RulePolicy(
        "GLOBALS_MUTATION", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_MODULE_MUTATION_FIXTURES, "keep",
    ),
    "GLOBALS_ACCESS": RulePolicy(
        "GLOBALS_ACCESS", "evaluator_state_support", "support", SUSPICIOUS_ONLY,
        ("GLOBALS_MUTATION", "MODULE_MUTATION", "EVALUATOR_EXPLOIT"),
        AMD_MODULE_MUTATION_FIXTURES, "downgrade",
    ),
    "FRAME_WALK_ACCESS": RulePolicy(
        "FRAME_WALK_ACCESS", "evaluator_state_support", "telemetry", TELEMETRY_ONLY,
        (), (), "split",
    ),
    "FRAME_WALK_MUTATION": RulePolicy(
        "FRAME_WALK_MUTATION", "evaluator_state_mutation", "hard", AUTO_FILTER,
        (), AMD_MANUAL_FRAME_MUTATION_FIXTURES, "keep",
    ),
    "SYS_MODULES_ACCESS": RulePolicy(
        "SYS_MODULES_ACCESS", "evaluator_state_support", "telemetry", TELEMETRY_ONLY,
        (), AMD_MODULE_MUTATION_FIXTURES, "downgrade",
    ),
    "CODE_REPLACEMENT": RulePolicy(
        "CODE_REPLACEMENT", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_CODE_REPLACEMENT_FIXTURES, "keep",
    ),
    "CODE_ACCESS": RulePolicy(
        "CODE_ACCESS", "evaluator_state_support", "telemetry", TELEMETRY_ONLY,
        (), AMD_CODE_REPLACEMENT_FIXTURES, "downgrade",
    ),
    "CONFIG_CACHE_EXPLOIT": RulePolicy(
        "CONFIG_CACHE_EXPLOIT", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_CONFIG_CACHE_FIXTURES, "rewrite",
    ),
    "REFERENCE_PRECOMPUTE_REPLAY": RulePolicy(
        "REFERENCE_PRECOMPUTE_REPLAY", "reference_replay", "hard", AUTO_FILTER, (),
        AMD_MANUAL_PRECOMPUTE_FIXTURES, "keep",
    ),
    "POINTER_REPLAY": RulePolicy(
        "POINTER_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_MANUAL_POINTER_REPLAY_FIXTURES, "keep",
    ),
    "RESULT_BANK_REPLAY": RulePolicy(
        "RESULT_BANK_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        TOP5_LOWCONF_RESULT_BANK_FIXTURES, "keep",
    ),
    "OBFUSCATED_EXEC": RulePolicy(
        "OBFUSCATED_EXEC", "dynamic_execution", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "DYNAMIC_EXECUTION": RulePolicy(
        "DYNAMIC_EXECUTION", "dynamic_execution", "telemetry", TELEMETRY_ONLY, (),
        (), "split",
    ),
    "MODULE_RELOAD": RulePolicy(
        "MODULE_RELOAD", "dynamic_execution", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "THREAD_INJECTION": RulePolicy(
        "THREAD_INJECTION", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "LAZY_TENSOR": RulePolicy(
        "LAZY_TENSOR", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "TENSOR_INSTANCE_METHOD_OVERRIDE": RulePolicy(
        "TENSOR_INSTANCE_METHOD_OVERRIDE", "timing_manipulation", "hard", AUTO_FILTER,
        (), (), "keep",
    ),
    "PRECISION_DOWNGRADE": RulePolicy(
        "PRECISION_DOWNGRADE", "approximation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "TOKEN_PASTE_CUDA_API": RulePolicy(
        "TOKEN_PASTE_CUDA_API", "filter_evasion", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SEQUENCE_BATCH_GRAPH": RulePolicy(
        "SEQUENCE_BATCH_GRAPH", "result_reuse", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "PARTIAL_GRAPH_KEY": RulePolicy(
        "PARTIAL_GRAPH_KEY", "result_reuse", "support", SUSPICIOUS_ONLY, (),
        (), "keep",
    ),
    "RUNTIME_PACKAGE_INSTALL": RulePolicy(
        "RUNTIME_PACKAGE_INSTALL", "sandbox_violation", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SCORE_PHYSICS_FLOOR": RulePolicy(
        "SCORE_PHYSICS_FLOOR", "score_anomaly", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SCORE_IMPOSSIBLE": RulePolicy(
        "SCORE_IMPOSSIBLE", "score_anomaly", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SCORE_SUSPECT_FLOOR": RulePolicy(
        "SCORE_SUSPECT_FLOOR", "score_anomaly", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SCORE_BROKEN": RulePolicy(
        "SCORE_BROKEN", "score_anomaly", "support", SUSPICIOUS_ONLY, (),
        (), "keep",
    ),
    "SCORE_EXTREME_SPEEDUP": RulePolicy(
        "SCORE_EXTREME_SPEEDUP", "score_anomaly", "support", SUSPICIOUS_ONLY, (),
        (), "downgrade",
    ),
    "DUPLICATE_SPAM": RulePolicy(
        "DUPLICATE_SPAM", "administrative", "support", SUSPICIOUS_ONLY, (),
        (), "downgrade",
    ),
    "NEAR_CLONE_SPAM": RulePolicy(
        "NEAR_CLONE_SPAM", "administrative", "support", SUSPICIOUS_ONLY, (),
        (), "downgrade",
    ),
}

BASE_SCORE_CONFIG = {
    "enabled": True,
    "impossible_below": 1e-8,       # Global absolute floor (~10ns, below any real GPU op)
    "suspect_floor_below": 5e-6,    # Near HIP/CUDA event timing floor (~4-5µs)
    "broken_above": 0.5,
    "extreme_speedup_above": 100.0,
    # Per-problem detection uses SCORE_PHYSICS_FLOOR (roofline model) instead
    # of these global thresholds.  These are last-resort fallbacks for problems
    # without registered benchmark specs.
}

BASE_DUPLICATE_CONFIG = {
    "exact": True,
    "near_clone": True,
}

BASE_CLASSIFICATION_CONFIG = {
    "default_reference_classification": "low_confidence",
    "default_reference_allowed_medium": ["DUPLICATE_SPAM"],
    "support_only_code_classification": "low_confidence",
    "support_code_classification": "suspicious",
    "telemetry_code_classification": "low_confidence",
    "metadata_support_classification": "suspicious",
    "admin_support_classification": "low_confidence",
    "code_auto_filter_reason": "high_critical",
    "metadata_auto_filter_reason": "score_anomaly",
    "admin_reason": "admin_review",
    "none_reason": "none",
}

BASE_AUDIT_CONFIG = {
    "archive_dir": "",
    "ground_truth_dir": "",
    "manual_review_files": [],
    "filtered_results_path": "",
    "result_files": {},
}

RULE_REGISTRY: dict[str, RulePolicy] = dict(BASE_RULE_REGISTRY)
SCORE_CONFIG: dict[str, Any] = copy.deepcopy(BASE_SCORE_CONFIG)
DUPLICATE_CONFIG: dict[str, Any] = copy.deepcopy(BASE_DUPLICATE_CONFIG)
CLASSIFICATION_CONFIG: dict[str, Any] = copy.deepcopy(BASE_CLASSIFICATION_CONFIG)
ACTIVE_RUNTIME_CONFIG: dict[str, Any] = {}


OUTCOME_ORDER = {
    TELEMETRY_ONLY: 1,
    SUSPICIOUS_ONLY: 2,
    AUTO_FILTER: 3,
}


def get_rule_policy(pattern: str) -> RulePolicy:
    return RULE_REGISTRY.get(
        pattern,
        RulePolicy(pattern, "unclassified", "telemetry", TELEMETRY_ONLY, (), (), "keep"),
    )


def strongest_rule_outcome(matched_patterns: list[dict]) -> str:
    if not matched_patterns:
        return TELEMETRY_ONLY
    return max(
        (get_rule_policy(p["pattern"]).max_outcome for p in matched_patterns),
        key=lambda outcome: OUTCOME_ORDER[outcome],
    )


ADMIN_PATTERNS = {"DUPLICATE_SPAM", "NEAR_CLONE_SPAM"}


def split_match_domains(matched_patterns: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split matches into code, metadata, and administrative domains."""
    code_patterns = []
    metadata_patterns = []
    admin_patterns = []
    for pattern in matched_patterns:
        name = pattern["pattern"]
        if name in ADMIN_PATTERNS:
            admin_patterns.append(pattern)
        elif name.startswith("SCORE_") or pattern.get("field") == "metadata":
            metadata_patterns.append(pattern)
        else:
            code_patterns.append(pattern)
    return code_patterns, metadata_patterns, admin_patterns


def support_only_patterns(matched_patterns: list[dict]) -> bool:
    """Return True when every pattern is only support/telemetry evidence."""
    if not matched_patterns:
        return False
    return all(get_rule_policy(p["pattern"]).evidence_tier != "hard" for p in matched_patterns)


# ---------------------------------------------------------------------------
# All detectors list
# ---------------------------------------------------------------------------

CODE_DETECTORS = [
    # Regex-based detectors (Layer 1)
    detect_timer_monkeypatch,
    detect_print_injection,
    detect_evaluator_exploit,
    detect_scaled_mm_ref,
    detect_decode_mm_ref,
    detect_result_caching,
    detect_helper_output_replay_cache,
    detect_value_keyed_output_replay,
    detect_object_output_cache_replay,
    detect_last_call_replay,
    detect_first_call_state_replay,
    detect_shape_output_replay,
    detect_timed_input_replay,
    detect_pointer_cache_hit_replay,
    detect_partial_bound_storage_replay,
    detect_class_pointer_sentinel_replay,
    detect_mutable_default_state_replay,
    detect_function_attribute_state_replay,
    detect_nonlocal_factory_state_replay,
    detect_contextvar_state_replay,
    detect_alias_state_replay,
    detect_partial_bound_method_replay,
    detect_generator_send_replay,
    detect_class_self_pointer_replay,
    detect_lambda_pointer_dispatch_replay,
    detect_cuda_graph_replay,
    detect_silent_fallback,
    detect_trivial_probe,
    detect_input_passthrough_output,
    detect_indirect_input_passthrough_output,
    detect_input_reduction_output,
    detect_self_matmul_output,
    detect_torch_compile_cache,
    detect_cuda_graph_python,
    detect_hardcoded_shapes,
    detect_benchmark_unsafe_algo_dispatch,
    detect_input_structure_truncation_dispatch,
    detect_unsync_multistream,
    detect_cuda_event_disable_timing,
    detect_token_paste_cuda_api,
    detect_sequence_batch_graph,
    detect_runtime_package_install,
    # AST-based detectors (Layer 2)
    detect_trusted_module_import,
    detect_module_mutation,
    detect_globals_mutation,
    detect_introspection_exploit,
    detect_code_replacement,
    detect_harness_runtime_patching,
    detect_torch_validation_monkeypatch,
    detect_evaluator_mode_branch,
    detect_benchmark_shape_replay,
    detect_config_cache_exploit,
    detect_reference_precompute_replay,
    detect_pointer_replay,
    detect_result_bank_replay,
    detect_dynamic_execution,
    detect_thread_injection,
    detect_lazy_tensor,
    detect_precision_downgrade,
]

BASE_DETECTOR_SPECS = [
    ("timer_monkeypatch", detect_timer_monkeypatch),
    ("print_injection", detect_print_injection),
    ("evaluator_exploit", detect_evaluator_exploit),
    ("scaled_mm_ref", detect_scaled_mm_ref),
    ("decode_mm_ref", detect_decode_mm_ref),
    ("result_caching", detect_result_caching),
    ("helper_output_replay_cache", detect_helper_output_replay_cache),
    ("value_keyed_output_replay", detect_value_keyed_output_replay),
    ("object_output_cache_replay", detect_object_output_cache_replay),
    ("last_call_replay", detect_last_call_replay),
    ("first_call_state_replay", detect_first_call_state_replay),
    ("shape_output_replay", detect_shape_output_replay),
    ("timed_input_replay", detect_timed_input_replay),
    ("pointer_cache_hit_replay", detect_pointer_cache_hit_replay),
    ("partial_bound_storage_replay", detect_partial_bound_storage_replay),
    ("class_pointer_sentinel_replay", detect_class_pointer_sentinel_replay),
    ("mutable_default_state_replay", detect_mutable_default_state_replay),
    ("function_attribute_state_replay", detect_function_attribute_state_replay),
    ("nonlocal_factory_state_replay", detect_nonlocal_factory_state_replay),
    ("contextvar_state_replay", detect_contextvar_state_replay),
    ("alias_state_replay", detect_alias_state_replay),
    ("partial_bound_method_replay", detect_partial_bound_method_replay),
    ("generator_send_replay", detect_generator_send_replay),
    ("class_self_pointer_replay", detect_class_self_pointer_replay),
    ("lambda_pointer_dispatch_replay", detect_lambda_pointer_dispatch_replay),
    ("cuda_graph_replay", detect_cuda_graph_replay),
    ("silent_fallback", detect_silent_fallback),
    ("trivial_probe", detect_trivial_probe),
    ("input_passthrough_output", detect_input_passthrough_output),
    ("indirect_input_passthrough_output", detect_indirect_input_passthrough_output),
    ("input_reduction_output", detect_input_reduction_output),
    ("self_matmul_output", detect_self_matmul_output),
    ("torch_compile_cache", detect_torch_compile_cache),
    ("cuda_graph_python", detect_cuda_graph_python),
    ("hardcoded_shapes", detect_hardcoded_shapes),
    ("benchmark_unsafe_algo_dispatch", detect_benchmark_unsafe_algo_dispatch),
    ("input_structure_truncation_dispatch", detect_input_structure_truncation_dispatch),
    ("unsync_multistream", detect_unsync_multistream),
    ("cuda_event_disable_timing", detect_cuda_event_disable_timing),
    ("token_paste_cuda_api", detect_token_paste_cuda_api),
    ("sequence_batch_graph", detect_sequence_batch_graph),
    ("runtime_package_install", detect_runtime_package_install),
    ("trusted_module_import", detect_trusted_module_import),
    ("module_mutation", detect_module_mutation),
    ("globals_mutation", detect_globals_mutation),
    ("introspection_exploit", detect_introspection_exploit),
    ("code_replacement", detect_code_replacement),
    ("harness_runtime_patching", detect_harness_runtime_patching),
    ("torch_validation_monkeypatch", detect_torch_validation_monkeypatch),
    ("evaluator_mode_branch", detect_evaluator_mode_branch),
    ("benchmark_shape_replay", detect_benchmark_shape_replay),
    ("config_cache_exploit", detect_config_cache_exploit),
    ("reference_precompute_replay", detect_reference_precompute_replay),
    ("pointer_replay", detect_pointer_replay),
    ("result_bank_replay", detect_result_bank_replay),
    ("dynamic_execution", detect_dynamic_execution),
    ("thread_injection", detect_thread_injection),
    ("lazy_tensor", detect_lazy_tensor),
    ("precision_downgrade", detect_precision_downgrade),
]

VALID_RULE_OUTCOMES = {AUTO_FILTER, SUSPICIOUS_ONLY, TELEMETRY_ONLY}
VALID_RULE_TIERS = {"hard", "support", "telemetry"}
VALID_NONFILTER_CLASSES = {"valid", "low_confidence", "suspicious"}
BUILTIN_PROFILES = ("default", "strict", "generic", "sol-execbench")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def is_default_reference(matched_patterns: list[dict]) -> bool:
    """Return True when the submission is effectively a default/reference path.

    `_scaled_mm` reference paths are common in correctness fallbacks and
    default submissions. Keep a separate class for these rows, but only allow
    them to filter when the rule policy explicitly permits auto-filtering.
    """
    relevant_patterns = {
        p["pattern"] for p in matched_patterns
        if p["pattern"] not in {"DUPLICATE_SPAM", "NEAR_CLONE_SPAM"}
    }
    if not relevant_patterns:
        return False
    if relevant_patterns != {"SCALED_MM_REF"}:
        return False
    medium_families = {p["pattern"] for p in matched_patterns if p["severity"] == "medium"}
    allowed_medium = set(CLASSIFICATION_CONFIG["default_reference_allowed_medium"])
    disqualifying_medium = medium_families - allowed_medium - {"SCALED_MM_REF"}
    return not disqualifying_medium


def filter_reason(matched_patterns: list[dict]) -> str:
    """Return a short label explaining what drove the filter decision."""
    if not matched_patterns:
        return str(CLASSIFICATION_CONFIG["none_reason"])
    code_patterns, metadata_patterns, admin_patterns = split_match_domains(matched_patterns)

    code_auto_filter = [
        p for p in code_patterns if get_rule_policy(p["pattern"]).max_outcome == AUTO_FILTER
    ]
    metadata_auto_filter = [
        p for p in metadata_patterns if get_rule_policy(p["pattern"]).max_outcome == AUTO_FILTER
    ]
    if code_auto_filter:
        return str(CLASSIFICATION_CONFIG["code_auto_filter_reason"])
    if metadata_auto_filter:
        return str(CLASSIFICATION_CONFIG["metadata_auto_filter_reason"])
    if admin_patterns:
        return str(CLASSIFICATION_CONFIG["admin_reason"])
    return str(CLASSIFICATION_CONFIG["none_reason"])


def classify(matched_patterns: list[dict]) -> tuple[str, bool]:
    """Return (classification, should_filter) from matched patterns.

    Precision-first policy:
      hacked            — at least one rule whose policy allows auto-filtering.
      default_reference — `_scaled_mm` reference behavior; not filterable in this pass.
      suspicious        — support signals or rewrite-before-trust rules.
      low_confidence    — telemetry-only signals.
      valid             — no signals detected.
    """
    if not matched_patterns:
        return "valid", False

    code_patterns, metadata_patterns, admin_patterns = split_match_domains(matched_patterns)

    code_strongest = strongest_rule_outcome(code_patterns) if code_patterns else TELEMETRY_ONLY
    metadata_strongest = strongest_rule_outcome(metadata_patterns) if metadata_patterns else TELEMETRY_ONLY
    admin_strongest = strongest_rule_outcome(admin_patterns) if admin_patterns else TELEMETRY_ONLY

    # Score/metadata auto-filter always takes priority — even default reference
    # implementations should be filtered if their score is physically impossible.
    if metadata_strongest == AUTO_FILTER:
        return "hacked", True

    if is_default_reference(code_patterns):
        return str(CLASSIFICATION_CONFIG["default_reference_classification"]), False

    if code_strongest == AUTO_FILTER:
        return "hacked", True

    if code_patterns:
        if code_strongest == SUSPICIOUS_ONLY:
            if support_only_patterns(code_patterns):
                return str(CLASSIFICATION_CONFIG["support_only_code_classification"]), False
            return str(CLASSIFICATION_CONFIG["support_code_classification"]), False
        return str(CLASSIFICATION_CONFIG["telemetry_code_classification"]), False

    if metadata_strongest == SUSPICIOUS_ONLY:
        return str(CLASSIFICATION_CONFIG["metadata_support_classification"]), False
    if admin_strongest == SUSPICIOUS_ONLY:
        return str(CLASSIFICATION_CONFIG["admin_support_classification"]), False
    return "valid", False


def build_default_runtime_config() -> dict[str, Any]:
    """Return the default detector configuration as a plain dict."""
    return {
        "version": CONFIG_VERSION,
        "profile": DEFAULT_PROFILE_NAME,
        "entrypoints": {
            "names": list(BASE_ENTRYPOINT_NAMES),
        },
        "trusted": {
            "modules": sorted(BASE_TRUSTED_MODULES),
            "functions": sorted(BASE_TRUSTED_FUNCTIONS),
        },
        "thresholds": {
            "score": copy.deepcopy(BASE_SCORE_CONFIG),
        },
        "duplicates": copy.deepcopy(BASE_DUPLICATE_CONFIG),
        "classification": copy.deepcopy(BASE_CLASSIFICATION_CONFIG),
        "audit": copy.deepcopy(BASE_AUDIT_CONFIG),
        "detectors": {
            "enabled": [name for name, _ in BASE_DETECTOR_SPECS],
            "disabled": [],
        },
        "rules": {
            rule_name: {
                "technique_family": policy.technique_family,
                "evidence_tier": policy.evidence_tier,
                "max_outcome": policy.max_outcome,
                "requires_companion_patterns": list(policy.requires_companion_patterns),
                "default_verdict": policy.default_verdict,
            }
            for rule_name, policy in BASE_RULE_REGISTRY.items()
        },
    }


def _builtin_profile_overrides(profile_name: str) -> dict[str, Any]:
    if profile_name == "default":
        return {}
    if profile_name == "strict":
        return {
            "classification": {
                "support_only_code_classification": "suspicious",
            },
            "thresholds": {
                "score": {
                    "suspect_floor_below": 2e-5,
                    "extreme_speedup_above": 50.0,
                },
            },
            "rules": {
                "CUDA_GRAPH_REPLAY": {"max_outcome": SUSPICIOUS_ONLY},
                "CUDA_GRAPH_PYTHON": {"max_outcome": SUSPICIOUS_ONLY},
                "UNSYNC_MULTISTREAM": {"max_outcome": SUSPICIOUS_ONLY},
                "CUDA_EVENT_DISABLE_TIMING": {"max_outcome": SUSPICIOUS_ONLY},
                "THREAD_INJECTION": {"max_outcome": SUSPICIOUS_ONLY},
                "LAZY_TENSOR": {"max_outcome": SUSPICIOUS_ONLY},
                "DYNAMIC_EXECUTION": {"max_outcome": SUSPICIOUS_ONLY},
                "MODULE_RELOAD": {"max_outcome": SUSPICIOUS_ONLY},
            },
        }
    if profile_name == "generic":
        return {
            "entrypoints": {
                "names": ["custom_kernel", "kernel", "forward", "run"],
            },
            "trusted": {
                "modules": [],
                "functions": [],
            },
            "thresholds": {
                "score": {
                    "enabled": False,
                },
            },
            "duplicates": {
                "exact": False,
                "near_clone": False,
            },
            "detectors": {
                "disabled": [
                    "evaluator_exploit",
                    "trusted_module_import",
                    "module_mutation",
                    "globals_mutation",
                    "introspection_exploit",
                    "code_replacement",
                    "harness_runtime_patching",
                    "reference_precompute_replay",
                ],
            },
        }
    if profile_name == "sol-execbench":
        return {
            "entrypoints": {
                "names": ["custom_kernel", "kernel", "forward", "run"],
            },
        }
    raise ValueError(
        f"Unknown profile {profile_name!r}. Available profiles: {', '.join(BUILTIN_PROFILES)}"
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override_value(raw_value: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw_value}")["value"]
    except tomllib.TOMLDecodeError:
        return raw_value


def _apply_dotted_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if child is None or not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _parse_override_argument(raw_override: str) -> tuple[str, Any]:
    if "=" not in raw_override:
        raise ValueError(f"Override must use key=value syntax, got: {raw_override}")
    key, raw_value = raw_override.split("=", 1)
    return key.strip(), _parse_override_value(raw_value.strip())


def _validate_runtime_config(config: dict[str, Any]) -> None:
    if int(config.get("version", CONFIG_VERSION)) != CONFIG_VERSION:
        raise ValueError(
            f"Unsupported config version {config.get('version')!r}; expected {CONFIG_VERSION}"
        )

    entrypoints = config.get("entrypoints", {}).get("names", [])
    if not entrypoints or not all(isinstance(name, str) and name for name in entrypoints):
        raise ValueError("entrypoints.names must contain at least one non-empty function name")

    trusted = config.get("trusted", {})
    for key in ("modules", "functions"):
        values = trusted.get(key, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"trusted.{key} must be a list of strings")

    score_cfg = config.get("thresholds", {}).get("score", {})
    required_score_keys = {
        "enabled",
        "impossible_below",
        "suspect_floor_below",
        "broken_above",
        "extreme_speedup_above",
    }
    if set(score_cfg) < required_score_keys:
        missing = sorted(required_score_keys - set(score_cfg))
        raise ValueError(f"thresholds.score missing keys: {', '.join(missing)}")

    duplicate_cfg = config.get("duplicates", {})
    for key in ("exact", "near_clone"):
        if not isinstance(duplicate_cfg.get(key), bool):
            raise ValueError(f"duplicates.{key} must be a boolean")

    classification_cfg = config.get("classification", {})
    list_fields = {"default_reference_allowed_medium"}
    required_classification_keys = set(BASE_CLASSIFICATION_CONFIG)
    if set(classification_cfg) < required_classification_keys:
        missing = sorted(required_classification_keys - set(classification_cfg))
        raise ValueError(f"classification missing keys: {', '.join(missing)}")
    for key, value in classification_cfg.items():
        if key in list_fields:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"classification.{key} must be a list of strings")
            continue
        if key.endswith("_classification") and value not in VALID_NONFILTER_CLASSES:
            raise ValueError(
                f"classification.{key} must be one of {sorted(VALID_NONFILTER_CLASSES)}"
            )
        if key.endswith("_reason") and not isinstance(value, str):
            raise ValueError(f"classification.{key} must be a string")

    audit_cfg = config.get("audit", {})
    required_audit_keys = set(BASE_AUDIT_CONFIG)
    if set(audit_cfg) < required_audit_keys:
        missing = sorted(required_audit_keys - set(audit_cfg))
        raise ValueError(f"audit missing keys: {', '.join(missing)}")
    for key in ("archive_dir", "ground_truth_dir", "filtered_results_path"):
        value = audit_cfg.get(key)
        if not isinstance(value, str):
            raise ValueError(f"audit.{key} must be a string")
    manual_review_files = audit_cfg.get("manual_review_files", [])
    if not isinstance(manual_review_files, list) or not all(isinstance(item, str) for item in manual_review_files):
        raise ValueError("audit.manual_review_files must be a list of strings")
    result_files = audit_cfg.get("result_files", {})
    if not isinstance(result_files, dict):
        raise ValueError("audit.result_files must be a table/object")
    for label, path in result_files.items():
        if not isinstance(label, str) or not isinstance(path, str):
            raise ValueError("audit.result_files entries must map string labels to string paths")

    detector_cfg = config.get("detectors", {})
    valid_detector_ids = {name for name, _ in BASE_DETECTOR_SPECS}
    for key in ("enabled", "disabled"):
        values = detector_cfg.get(key, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"detectors.{key} must be a list of detector ids")
        unknown = sorted(set(values) - valid_detector_ids)
        if unknown:
            raise ValueError(f"Unknown detector ids in detectors.{key}: {', '.join(unknown)}")

    rule_cfg = config.get("rules", {})
    unknown_rules = sorted(set(rule_cfg) - set(BASE_RULE_REGISTRY))
    if unknown_rules:
        raise ValueError(f"Unknown rule overrides: {', '.join(unknown_rules)}")
    for rule_name, overrides in rule_cfg.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"rules.{rule_name} must be a table/object")
        if "evidence_tier" in overrides and overrides["evidence_tier"] not in VALID_RULE_TIERS:
            raise ValueError(
                f"rules.{rule_name}.evidence_tier must be one of {sorted(VALID_RULE_TIERS)}"
            )
        if "max_outcome" in overrides and overrides["max_outcome"] not in VALID_RULE_OUTCOMES:
            raise ValueError(
                f"rules.{rule_name}.max_outcome must be one of {sorted(VALID_RULE_OUTCOMES)}"
            )
        if "requires_companion_patterns" in overrides:
            values = overrides["requires_companion_patterns"]
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError(
                    f"rules.{rule_name}.requires_companion_patterns must be a list of strings"
                )


def resolve_runtime_config(
    *,
    profile: str = DEFAULT_PROFILE_NAME,
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
    config_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve built-in profile, TOML file, and dotted overrides into one config."""
    resolved = build_default_runtime_config()
    resolved = _deep_merge(resolved, _builtin_profile_overrides(profile))

    if config_path:
        with open(config_path, "rb") as config_file:
            file_config = tomllib.load(config_file)
        resolved = _deep_merge(resolved, file_config)

    if config_data:
        resolved = _deep_merge(resolved, config_data)

    for raw_override in overrides or []:
        key, value = _parse_override_argument(raw_override)
        _apply_dotted_override(resolved, key, value)

    if profile != DEFAULT_PROFILE_NAME:
        resolved["profile"] = profile
    else:
        resolved["profile"] = str(resolved.get("profile") or profile)
    resolved.setdefault("version", CONFIG_VERSION)
    _validate_runtime_config(resolved)
    return resolved


def _resolve_enabled_detector_ids(config: dict[str, Any]) -> set[str]:
    enabled = set(config["detectors"]["enabled"])
    disabled = set(config["detectors"].get("disabled", []))
    effective = enabled - disabled
    if not effective:
        raise ValueError("Configuration disables all detectors")
    return effective


def apply_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply a resolved configuration to the module-level runtime state."""
    global ACTIVE_RUNTIME_CONFIG
    global ENTRYPOINT_NAMES
    global TRUSTED_MODULES
    global TRUSTED_FUNCTIONS
    global TRUSTED_HARNESS_NAMES
    global SCORE_CONFIG
    global DUPLICATE_CONFIG
    global CLASSIFICATION_CONFIG
    global RULE_REGISTRY
    global CODE_DETECTORS
    global STRUCTURAL_HASH_PRESERVE_NAMES

    _validate_runtime_config(config)

    ENTRYPOINT_NAMES = tuple(config["entrypoints"]["names"])
    TRUSTED_MODULES = frozenset(config["trusted"]["modules"])
    TRUSTED_FUNCTIONS = frozenset(config["trusted"]["functions"])
    TRUSTED_HARNESS_NAMES = frozenset(TRUSTED_FUNCTIONS | BASE_TRUSTED_HARNESS_NAMES)
    SCORE_CONFIG = copy.deepcopy(config["thresholds"]["score"])
    DUPLICATE_CONFIG = copy.deepcopy(config["duplicates"])
    CLASSIFICATION_CONFIG = copy.deepcopy(config["classification"])

    RULE_REGISTRY = {}
    for rule_name, base_policy in BASE_RULE_REGISTRY.items():
        overrides = config["rules"].get(rule_name, {})
        RULE_REGISTRY[rule_name] = RulePolicy(
            rule_name=rule_name,
            technique_family=str(overrides.get("technique_family", base_policy.technique_family)),
            evidence_tier=str(overrides.get("evidence_tier", base_policy.evidence_tier)),
            max_outcome=str(overrides.get("max_outcome", base_policy.max_outcome)),
            requires_companion_patterns=tuple(
                overrides.get("requires_companion_patterns", list(base_policy.requires_companion_patterns))
            ),
            mapped_positive_fixtures=base_policy.mapped_positive_fixtures,
            default_verdict=str(overrides.get("default_verdict", base_policy.default_verdict)),
        )

    enabled_detector_ids = _resolve_enabled_detector_ids(config)
    CODE_DETECTORS = [
        detector
        for detector_name, detector in BASE_DETECTOR_SPECS
        if detector_name in enabled_detector_ids
    ]

    STRUCTURAL_HASH_PRESERVE_NAMES = frozenset(
        _BASE_STRUCTURAL_HASH_PRESERVE_NAMES | set(ENTRYPOINT_NAMES)
    )
    ACTIVE_RUNTIME_CONFIG = copy.deepcopy(config)
    return copy.deepcopy(ACTIVE_RUNTIME_CONFIG)


def configure_runtime(
    *,
    profile: str = DEFAULT_PROFILE_NAME,
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
    config_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve and apply runtime configuration for CLI or library usage."""
    resolved = resolve_runtime_config(
        profile=profile,
        config_path=config_path,
        overrides=overrides,
        config_data=config_data,
    )
    return apply_runtime_config(resolved)


def _toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_quote(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _append_toml_table(lines: list[str], prefix: str, table: dict[str, Any]) -> None:
    scalar_items = []
    child_tables = []
    for key, value in table.items():
        if isinstance(value, dict):
            child_tables.append((key, value))
        else:
            scalar_items.append((key, value))

    if prefix:
        lines.append(f"[{prefix}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_value(value)}")
    if scalar_items or prefix:
        lines.append("")

    for key, child in child_tables:
        child_prefix = f"{prefix}.{key}" if prefix else key
        _append_toml_table(lines, child_prefix, child)


def runtime_config_to_toml(config: Optional[dict[str, Any]] = None) -> str:
    """Serialize a runtime config dict to TOML."""
    data = copy.deepcopy(config or ACTIVE_RUNTIME_CONFIG or build_default_runtime_config())
    lines = [
        "# kernelguard runtime configuration",
        "# Generated from the current built-in defaults/profile resolution.",
        "",
    ]
    root_scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    root_tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    for key, value in root_scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if root_scalars:
        lines.append("")
    for key, value in root_tables.items():
        _append_toml_table(lines, key, value)
    return "\n".join(lines).rstrip() + "\n"


def _worker_pool_init(config: dict[str, Any]) -> None:
    apply_runtime_config(config)


# ---------------------------------------------------------------------------
# Code hashing for dedup
# ---------------------------------------------------------------------------

def normalize_code(code: str) -> str:
    """Normalize code for dedup: strip comments and collapse whitespace."""
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w*)\b')
_BASE_STRUCTURAL_HASH_PRESERVE_NAMES = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
    # common builtins / torch names to preserve structure
    "torch", "self", "data", "int", "float", "bool", "str", "list",
    "dict", "set", "tuple", "len", "range", "print", "type", "super",
    "input_t", "output_t",
})
STRUCTURAL_HASH_PRESERVE_NAMES = frozenset(_BASE_STRUCTURAL_HASH_PRESERVE_NAMES | set(ENTRYPOINT_NAMES))

apply_runtime_config(build_default_runtime_config())


def structural_hash(code: str) -> str:
    """SHA-256 after stripping comments, collapsing whitespace, and renaming
    non-keyword identifiers.  Catches trivial renames between near-clones."""
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Map identifiers to positional placeholders (preserves structure)
    ident_map: dict[str, str] = {}
    counter = 0

    def replace_ident(m: re.Match) -> str:
        nonlocal counter
        name = m.group(1)
        if name in STRUCTURAL_HASH_PRESERVE_NAMES or name.startswith('__'):
            return name
        if name not in ident_map:
            ident_map[name] = f"v{counter}"
            counter += 1
        return ident_map[name]

    normalized = _IDENT_RE.sub(replace_ident, code)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def code_hash(code: str) -> str:
    """SHA-256 of normalized code."""
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Analyze a single code sample
# ---------------------------------------------------------------------------

_RE_POPCORN_LEADERBOARD = re.compile(
    r'^#!POPCORN\s+leaderboard\s+(\S+)', re.MULTILINE,
)
_RE_PEP723_LEADERBOARD = re.compile(
    r'#\s*///\s*script\b.*?#\s*leaderboard\s*=\s*["\']([^"\']+)["\']',
    re.DOTALL,
)


def extract_problem_name(code: str) -> Optional[str]:
    """Extract competition/leaderboard name from submission code headers.

    Supports two formats:
      #!POPCORN leaderboard nvfp4_dual_gemm     (older NVIDIA subs)
      # /// script                               (PEP 723 inline metadata)
      # leaderboard = "amd-mixed-mla"
      # ///
    """
    m = _RE_POPCORN_LEADERBOARD.search(code[:500])
    if m:
        return m.group(1).strip()
    m = _RE_PEP723_LEADERBOARD.search(code[:500])
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Default / reference submission detection (for dataset cleaning, not hack detection)
# ---------------------------------------------------------------------------

def _normalize_func(func_src: str) -> str:
    """Strip comments, docstrings, blank lines from a function body."""
    lines = []
    in_doc = False
    for line in func_src.splitlines():
        s = line.strip()
        if '"""' in s or "'''" in s:
            if s.count('"""') + s.count("'''") == 1:
                in_doc = not in_doc
            continue
        if in_doc or not s or s.startswith('#'):
            continue
        lines.append(s)
    return "\n".join(lines)


def _extract_func_body(code: str, name: str) -> str:
    """Extract a function body source using AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(code, node) or ""
    return ""


def _hash_func(func_src: str) -> str:
    """Hash a normalized function body."""
    import hashlib
    return hashlib.sha256(_normalize_func(func_src).encode()).hexdigest()[:16]


# Reference kernel hashes — generated from gpu-mode/reference-kernels repo.
# Key: ref_kernel body hash. Value: list of problem directory names.
# To regenerate: hash ref_kernel from each problems/*/reference.py
_REFERENCE_HASHES: dict[str, list[str]] = {}


def _load_reference_hashes() -> None:
    """Load reference hashes from the bundled reference-kernels directory."""
    global _REFERENCE_HASHES
    if _REFERENCE_HASHES:
        return
    import hashlib as _hl
    ref_base = os.path.join(os.path.dirname(__file__), "reference-kernels", "problems")
    if not os.path.isdir(ref_base):
        return
    for root, _dirs, files in os.walk(ref_base):
        if "reference.py" not in files:
            continue
        ref_code = open(os.path.join(root, "reference.py")).read()
        body = _extract_func_body(ref_code, "ref_kernel")
        if not body:
            continue
        h = _hl.sha256(_normalize_func(body).encode()).hexdigest()[:16]
        rel = os.path.relpath(root, ref_base)
        _REFERENCE_HASHES.setdefault(h, []).append(rel)


def is_default_submission(code: str) -> dict:
    """Check if a submission is a default/reference copy.

    Returns dict with:
        is_default: bool
        reason: str  ("ref_passthrough", "ref_clone", or None)
        detail: str  (human-readable explanation)

    This is for dataset cleaning — not used in hack detection or filtering.
    """
    _load_reference_hashes()

    # 1. Literal passthrough: from reference import ref_kernel; return ref_kernel(data)
    if re.search(r'from\s+reference\s+import\s+ref_kernel', code):
        body = _extract_func_body(code, "custom_kernel")
        norm = _normalize_func(body)
        # Check the function does nothing but call ref_kernel
        active = [l for l in norm.splitlines()
                  if l and not l.startswith("def ") and not l.startswith("from ") and not l.startswith("import ")]
        if len(active) <= 2:
            return {"is_default": True, "reason": "ref_passthrough",
                    "detail": "Directly calls ref_kernel with no custom logic"}

    # 2. Code clone: custom_kernel body hashes to same as a known ref_kernel
    body = _extract_func_body(code, "custom_kernel")
    if body:
        h = _hash_func(body)
        if h in _REFERENCE_HASHES:
            problems = _REFERENCE_HASHES[h]
            return {"is_default": True, "reason": "ref_clone",
                    "detail": f"custom_kernel body matches ref_kernel from {problems}"}

    return {"is_default": False, "reason": None, "detail": None}


def analyze_code(code: str, metadata: Optional[dict] = None, field: str = "code",
                 compute_structural_hash: bool = True) -> dict:
    """Run all detectors on a code sample, return result dict.

    compute_structural_hash: set False for JSONL bulk mode (expensive identifier
    normalization is impractical at 44K × 2 entries; exact dedup suffices there).
    """
    facts = build_submission_facts(code)
    all_matches = []
    for detector in CODE_DETECTORS:
        hits = detector(facts)
        for h in hits:
            h["field"] = field
        all_matches.extend(hits)

    # Auto-detect problem_name from code if not in metadata
    if metadata is None:
        metadata = {}
    if not metadata.get("problem_name") and not metadata.get("problem"):
        extracted = extract_problem_name(code)
        if extracted:
            metadata = {**metadata, "problem_name": extracted}

    # Score anomaly (uses metadata, not code)
    score_hits = detect_score_anomaly(metadata)
    for h in score_hits:
        h["field"] = "metadata"
    all_matches.extend(score_hits)

    classification, should_filter = classify(all_matches)

    reason = filter_reason(all_matches) if should_filter else None
    sh = structural_hash(code) if compute_structural_hash else ""
    default_info = is_default_submission(code)
    return {
        "matched_patterns": all_matches,
        "classification": classification,
        "should_filter": should_filter,
        "filter_reason": reason,
        "code_hash": code_hash(code),
        "structural_hash": sh,
        "is_default": default_info["is_default"],
        "default_reason": default_info["reason"],
    }


def _extract_nvidia_archive_scores(runs_payload: Any) -> list[float]:
    """Extract leaderboard / run scores from legacy and current archive layouts."""
    scores: list[float] = []

    def add_score(value: Any) -> None:
        if isinstance(value, (int, float)) and value > 0:
            scores.append(float(value))

    def visit_row(row: Any) -> None:
        if not isinstance(row, dict):
            return
        add_score(row.get("score"))
        leaderboard = row.get("leaderboard")
        if isinstance(leaderboard, dict):
            add_score(leaderboard.get("score"))
        result = row.get("result")
        if isinstance(result, dict):
            add_score(result.get("score"))

    if isinstance(runs_payload, list):
        for row in runs_payload:
            visit_row(row)
    elif isinstance(runs_payload, dict):
        visit_row(runs_payload)
        nested_runs = runs_payload.get("runs")
        if isinstance(nested_runs, list):
            for row in nested_runs:
                visit_row(row)

    return list(dict.fromkeys(scores))


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker_jsonl(args: tuple) -> Optional[dict]:
    """Analyze one JSONL pair entry.  Returns result dict or None on parse error."""
    line_num, line = args
    line = line.strip()
    if not line:
        return None
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None

    entry_id = entry.get("id", f"line_{line_num}")
    user = entry.get("user", "unknown")
    problem = entry.get("problem_name", "unknown")
    metadata = {
        "improved_score": entry.get("improved_score"),
        "baseline_score": entry.get("baseline_score"),
    }

    # In pair mode, code-side detectors should stay attached to their source side.
    # Score anomalies are entry-level metadata signals and must be emitted once.
    r_imp = analyze_code(entry.get("improved_code", ""), None,
                         field="improved_code", compute_structural_hash=False)
    r_base = analyze_code(entry.get("baseline_code", ""), None,
                          field="baseline_code", compute_structural_hash=False)

    all_patterns = []
    for p in r_imp["matched_patterns"]:
        all_patterns.append(dict(p, field="improved_code"))
    for p in r_base["matched_patterns"]:
        all_patterns.append(dict(p, field="baseline_code"))
    for p in detect_score_anomaly(metadata):
        all_patterns.append(dict(p, field="metadata"))

    decision_patterns = [
        p for p in all_patterns if p["field"] in ("improved_code", "metadata")
    ]
    classification, should_filter = classify(decision_patterns)
    reason = filter_reason(decision_patterns) if should_filter else None

    return {
        "id": entry_id,
        "user": user,
        "problem_name": problem,
        "classification": classification,
        "should_filter": should_filter,
        "filter_reason": reason,
        "matched_patterns": all_patterns,
        "improved_score": entry.get("improved_score"),
        "baseline_score": entry.get("baseline_score"),
        "code_hash_improved": r_imp["code_hash"],
        "code_hash_baseline": r_base["code_hash"],
        "_line_num": line_num,
    }


def _worker_parquet(args: tuple) -> dict:
    """Analyze one parquet submission row."""
    sub_id, leaderboard_id, user_id, user_name, problem_name, score, passed, code = args
    metadata = {"score": score, "user": user_name, "problem": problem_name}
    r = analyze_code(code or "", metadata, field="code", compute_structural_hash=False)
    return {
        "submission_id": int(sub_id),
        "leaderboard_id": int(leaderboard_id),
        "user_id": str(user_id),
        "user": str(user_name),
        "problem_name": str(problem_name),
        "score": float(score) if score is not None else None,
        "passed": bool(passed),
        "classification": r["classification"],
        "should_filter": r["should_filter"],
        "filter_reason": r.get("filter_reason"),
        "matched_patterns": r["matched_patterns"],
        "code_hash": r["code_hash"],
    }


# ---------------------------------------------------------------------------
# Precision audit
# ---------------------------------------------------------------------------

DEFAULT_AUDIT_RESULT_FILES = (
    ("nvidia_archive", ("detection_results_nvidia_archive.jsonl", "detection_results_allnvidia.jsonl")),
    ("amd", ("detection_results_amd_submissions.jsonl", "detection_results_amd-latest-submissions-20260330.jsonl")),
    ("nvidia", ("detection_results_nvidia_submissions.jsonl", "detection_results_nvidia_nvfp4_submissions.jsonl")),
)

AUDIT_RULE_ORDER = [
    "EVALUATOR_EXPLOIT", "HARNESS_RUNTIME_PATCHING", "MODULE_MUTATION", "GLOBALS_MUTATION", "CODE_REPLACEMENT",
    "FRAME_WALK_ACCESS", "FRAME_WALK_MUTATION", "SYS_MODULES_ACCESS", "GLOBALS_ACCESS", "CODE_ACCESS",
    "TRUSTED_MODULE_IMPORT",
    "OUTPUT_REPLAY_CACHE", "LAST_CALL_REPLAY", "SHAPE_OUTPUT_REPLAY", "TIMED_INPUT_REPLAY", "CONFIG_CACHE_EXPLOIT", "POINTER_REPLAY", "RESULT_BANK_REPLAY",
    "INPUT_PASSTHROUGH_OUTPUT", "INPUT_REDUCTION_OUTPUT", "SELF_MATMUL_OUTPUT", "PREPROCESS_CACHE", "WORKSPACE_CACHE",
    "RUNNER_PLAN_CACHE", "CUDA_GRAPH_PYTHON", "CUDA_GRAPH_REPLAY",
    "TIMER_MONKEYPATCH", "FAKE_BENCHMARK_EMIT", "STDIO_REDIRECT", "UNSYNC_MULTISTREAM", "CUDA_EVENT_DISABLE_TIMING",
    "SCALED_MM_REF", "DECODE_MM_REF", "SILENT_FALLBACK", "REFERENCE_PRECOMPUTE_REPLAY", "TORCH_COMPILE_CACHE",
    "HARDCODED_SHAPES", "BENCHMARK_UNSAFE_ALGO_DISPATCH", "INPUT_STRUCTURE_TRUNCATION_DISPATCH", "TRIVIAL_PROBE",
    "OBFUSCATED_EXEC", "DYNAMIC_EXECUTION", "MODULE_RELOAD", "THREAD_INJECTION", "LAZY_TENSOR",
    "TENSOR_INSTANCE_METHOD_OVERRIDE",
    "TOKEN_PASTE_CUDA_API", "SEQUENCE_BATCH_GRAPH", "PARTIAL_GRAPH_KEY", "RUNTIME_PACKAGE_INSTALL",
    "PRECISION_DOWNGRADE", "SCORE_PHYSICS_FLOOR", "SCORE_IMPOSSIBLE", "SCORE_SUSPECT_FLOOR",
    "SCORE_BROKEN", "SCORE_EXTREME_SPEEDUP", "DUPLICATE_SPAM", "NEAR_CLONE_SPAM",
]


def _parse_nvidia_archive_submission_id(path: str) -> str:
    basename = os.path.basename(path)
    parts = basename.split("_", 3)
    return parts[2] if len(parts) >= 3 else "unknown"


def _find_nvidia_archive_submission_path(directory: str, submission_id: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(directory, f"nv_sub_{submission_id}_*.py")))
    return candidates[0] if candidates else None


def _find_amd_submission_path(directory: str, submission_id: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(directory, f"amd_mla_sub_{submission_id}_*.py")))
    return candidates[0] if candidates else None


def _safe_read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _resolve_existing_dir(preferred: str, fallbacks: tuple[str, ...]) -> str:
    for candidate in (preferred, *fallbacks):
        if os.path.isdir(candidate):
            return candidate
    return preferred


def _resolve_existing_file(preferred: str, fallbacks: tuple[str, ...]) -> str:
    for candidate in (preferred, *fallbacks):
        if os.path.exists(candidate):
            return candidate
    return preferred


def _resolve_existing_file_candidates(candidates: tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _existing_file_candidates(candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(candidate for candidate in candidates if os.path.exists(candidate))


def _fixture_id_aliases(fixture_id: str) -> tuple[str, ...]:
    aliases = {fixture_id}
    if ":" not in fixture_id:
        return tuple(sorted(aliases))

    prefix, suffix = fixture_id.split(":", 1)
    if prefix == "nvidia_archive":
        aliases.add(f"allnvidia:{suffix}")
    elif prefix == "allnvidia":
        aliases.add(f"nvidia_archive:{suffix}")
    elif prefix == "nvidia_archive_soft":
        aliases.add(f"allnvidia_soft:{suffix}")
    elif prefix == "allnvidia_soft":
        aliases.add(f"nvidia_archive_soft:{suffix}")
    elif prefix == "manual_review":
        aliases.add(f"amd_top10:{suffix}")
    elif prefix == "amd_top10":
        aliases.add(f"manual_review:{suffix}")
    return tuple(sorted(aliases))


LEGACY_NVIDIA_ARCHIVE_DIRS = ("AllNvidia",)
LEGACY_AMD_FIXTURE_DIRS = ("amd_hacked",)
LEGACY_FILTERED_NVIDIA_ARCHIVE_PATHS = ("filtered_allnvidia.jsonl",)
LEGACY_MANUAL_JUDGMENTS_PATHS = ("amd_top10_review/manual_judgments.json",)
LEGACY_EXTRA_MANUAL_REVIEW_PATHS = (
    "agentic_top5_competitions_eval_20260313/low_confidence_review/manual_judgments.json",
)
NVIDIA_HACKING_MANIFEST_FILENAMES = ("nvidia_hacking_manifest.json", "nvidia_hacking_submissions_all.json")


def _require_existing_dir(path: str, label: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_existing_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_classifier_fixture_manifest(
    nvidia_archive_dir: Optional[str] = None,
    amd_dir: Optional[str] = None,
    filtered_nvidia_archive_path: Optional[str] = None,
    manual_judgments_path: Optional[str] = None,
    extra_manual_review_paths: Optional[tuple[str, ...]] = None,
) -> dict:
    """Build the precision-audit fixture manifest.

    Hard positives:
      - amd_fixture_archive/ground_truth.json exploit=true
      - source-backed entries in NvidiaArchive/nvidia_hacking_manifest.json
    Hard negatives:
      - amd_fixture_archive/ground_truth.json exploit=false
    Soft review negatives:
      - deduplicated NvidiaArchive sources not in the hard-positive manifest and not
        already present in filtered_nvidia_archive.jsonl
    """
    if nvidia_archive_dir is None:
        nvidia_archive_dir = _resolve_existing_dir("NvidiaArchive", LEGACY_NVIDIA_ARCHIVE_DIRS)
    else:
        nvidia_archive_dir = _require_existing_dir(nvidia_archive_dir, "Audit NVIDIA archive directory")

    if amd_dir is None:
        amd_dir = _resolve_existing_dir("amd_fixture_archive", LEGACY_AMD_FIXTURE_DIRS)
    else:
        amd_dir = _require_existing_dir(amd_dir, "Audit AMD fixture directory")

    if filtered_nvidia_archive_path is None:
        filtered_nvidia_archive_path = _resolve_existing_file(
            "filtered_nvidia_archive.jsonl", LEGACY_FILTERED_NVIDIA_ARCHIVE_PATHS
        )
    else:
        filtered_nvidia_archive_path = _require_existing_file(
            filtered_nvidia_archive_path, "Filtered NVIDIA archive results"
        )

    if manual_judgments_path is None:
        manual_judgments_path = _resolve_existing_file(
            "manual_review_archive/manual_judgments.json", LEGACY_MANUAL_JUDGMENTS_PATHS
        )
    else:
        manual_judgments_path = _require_existing_file(
            manual_judgments_path, "Manual judgments file"
        )

    if extra_manual_review_paths is None:
        resolved_extra_manual_review_paths = []
        default_extra_paths = ("top_competitions_review/low_confidence_review/manual_judgments.json",)
        for idx, review_path in enumerate(default_extra_paths):
            legacy_fallback = (
                LEGACY_EXTRA_MANUAL_REVIEW_PATHS[idx]
                if idx < len(LEGACY_EXTRA_MANUAL_REVIEW_PATHS)
                else ()
            )
            if legacy_fallback:
                resolved_extra_manual_review_paths.append(
                    _resolve_existing_file(review_path, (legacy_fallback,))
                )
            else:
                resolved_extra_manual_review_paths.append(review_path)
        extra_manual_review_paths = tuple(resolved_extra_manual_review_paths)
    else:
        extra_manual_review_paths = tuple(
            _require_existing_file(path, "Extra manual review file")
            for path in extra_manual_review_paths
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hard_positives": [],
        "hard_negatives": [],
        "soft_review_negatives": [],
    }
    known_hard_positive_amd_ids: set[str] = set()

    gt_path = os.path.join(amd_dir, "ground_truth.json")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
        for row in ground_truth:
            sid = str(row["submission_id"])
            path = _find_amd_submission_path(amd_dir, sid)
            if not path:
                continue
            code = _safe_read_text(path)
            entry = {
                "fixture_id": f"amd:{sid}",
                "submission_id": sid,
                "label": "hard_positive" if row.get("is_exploit") else "hard_negative",
                "source": "amd_fixture_archive",
                "technique": row.get("technique", "unknown"),
                "note": row.get("note"),
                "file": path,
                "code_hash": code_hash(code),
            }
            bucket = "hard_positives" if row.get("is_exploit") else "hard_negatives"
            manifest[bucket].append(entry)
            if row.get("is_exploit"):
                known_hard_positive_amd_ids.add(sid)

    hacking_manifest_path = _resolve_existing_file_candidates(
        tuple(os.path.join(nvidia_archive_dir, name) for name in NVIDIA_HACKING_MANIFEST_FILENAMES)
    ) or os.path.join(nvidia_archive_dir, NVIDIA_HACKING_MANIFEST_FILENAMES[0])
    hard_positive_ids: set[str] = set()
    if os.path.exists(hacking_manifest_path):
        with open(hacking_manifest_path, encoding="utf-8") as f:
            hacking_manifest = json.load(f)
        for row in hacking_manifest:
            sid = str(row["submissionId"])
            path = _find_nvidia_archive_submission_path(nvidia_archive_dir, sid)
            if not path:
                continue
            code = _safe_read_text(path)
            manifest["hard_positives"].append({
                "fixture_id": f"nvidia_archive:{sid}",
                "submission_id": sid,
                "label": "hard_positive",
                "source": "nvidia_archive_hacking_manifest",
                "technique": "source_backed_hacking_manifest",
                "leaderboard": row.get("leaderboard"),
                "submitted_at": row.get("submittedAt"),
                "file": path,
                "code_hash": code_hash(code),
            })
            hard_positive_ids.add(sid)

    manual_review_sources = [
        (manual_judgments_path, "manual_review", "manual_archive_review"),
        *[(path, "top5low", "top5_lowconfidence_manual_review") for path in extra_manual_review_paths],
    ]
    for review_path, fixture_prefix, source_name in manual_review_sources:
        if not os.path.exists(review_path):
            continue
        with open(review_path, encoding="utf-8") as f:
            manual_rows = json.load(f)
        for row in manual_rows:
            if str(row.get("manual_filter", "")).lower() != "yes":
                continue
            sid = str(row["submission_id"])
            if fixture_prefix == "manual_review" and sid in known_hard_positive_amd_ids:
                continue
            code_path = row.get("code_path") or row.get("path")
            if not code_path or not os.path.exists(code_path):
                continue
            code = _safe_read_text(code_path)
            manifest["hard_positives"].append({
                "fixture_id": f"{fixture_prefix}:{sid}",
                "submission_id": sid,
                "label": "hard_positive",
                "source": source_name,
                "technique": row.get("primary_technique", "manual_review"),
                "problem_name": row.get("problem_name"),
                "manual_judgment": row.get("manual_judgment"),
                "file": code_path,
                "code_hash": code_hash(code),
            })
            if fixture_prefix == "manual_review":
                known_hard_positive_amd_ids.add(sid)

    filtered_ids: set[str] = set()
    if os.path.exists(filtered_nvidia_archive_path):
        with open(filtered_nvidia_archive_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                row = json.loads(line)
                filtered_ids.add(str(row.get("submission_id")))

    seen_hashes: set[str] = set()
    for path in sorted(glob.glob(os.path.join(nvidia_archive_dir, "nv_sub_*.py"))):
        sid = _parse_nvidia_archive_submission_id(path)
        if sid in hard_positive_ids or sid in filtered_ids:
            continue
        code = _safe_read_text(path)
        ch = code_hash(code)
        if ch in seen_hashes:
            continue
        seen_hashes.add(ch)
        manifest["soft_review_negatives"].append({
            "fixture_id": f"nvidia_archive_soft:{sid}",
            "submission_id": sid,
            "label": "soft_review_negative",
            "source": "nvidia_archive",
            "technique": "unknown",
            "file": path,
            "code_hash": ch,
        })

    return manifest


def _manifest_fixture_counts(manifest: dict) -> dict[str, int]:
    return {
        "hard_positives": len(manifest.get("hard_positives", [])),
        "hard_negatives": len(manifest.get("hard_negatives", [])),
        "soft_review_negatives": len(manifest.get("soft_review_negatives", [])),
    }


def _fixture_pattern_hits(fixtures: list[dict]) -> dict[str, set[str]]:
    hits_by_fixture: dict[str, set[str]] = {}
    for fixture in fixtures:
        code = _safe_read_text(fixture["file"])
        result = analyze_code(code, metadata=None, field="code", compute_structural_hash=False)
        patterns = {p["pattern"] for p in result["matched_patterns"]}
        for alias in _fixture_id_aliases(fixture["fixture_id"]):
            hits_by_fixture[alias] = patterns
    return hits_by_fixture


def _load_rule_examples_from_results(
    result_paths: tuple[str, ...], source_name: str
) -> tuple[Counter, dict[str, list[dict]]]:
    sole_counts: Counter = Counter()
    sole_examples: dict[str, list[dict]] = defaultdict(list)
    existing_paths = _existing_file_candidates(result_paths)
    if not existing_paths:
        return sole_counts, sole_examples
    seen_rows: set[tuple[Any, ...]] = set()

    for result_path in existing_paths:
        with open(result_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                row = json.loads(line)
                if not row.get("should_filter"):
                    continue
                patterns = sorted({p["pattern"] for p in row.get("matched_patterns", [])})
                if len(patterns) != 1:
                    continue
                pattern = patterns[0]
                dedupe_key = (
                    pattern,
                    row.get("submission_id"),
                    row.get("id"),
                    row.get("user"),
                    row.get("problem_name"),
                    row.get("filename"),
                )
                if dedupe_key in seen_rows:
                    continue
                seen_rows.add(dedupe_key)
                sole_counts[pattern] += 1
                if len(sole_examples[pattern]) < 20:
                    sole_examples[pattern].append({
                        "source": source_name,
                        "submission_id": row.get("submission_id"),
                        "id": row.get("id"),
                        "user": row.get("user"),
                        "problem_name": row.get("problem_name"),
                        "filename": row.get("filename"),
                    })
    return sole_counts, sole_examples


def _parse_audit_result_spec(spec: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid --audit-result value {spec!r}; expected label=/path/to/results.jsonl"
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(
            f"Invalid --audit-result value {spec!r}; expected label=/path/to/results.jsonl"
        )
    return label, (path,)


def resolve_audit_result_files(specs: list[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not specs:
        return DEFAULT_AUDIT_RESULT_FILES
    return tuple(_parse_audit_result_spec(spec) for spec in specs)


def _audit_result_specs_from_config(config: dict[str, Any]) -> list[str]:
    audit_cfg = config.get("audit", {})
    result_files = audit_cfg.get("result_files", {})
    return [f"{label}={path}" for label, path in result_files.items() if path]


def _nonempty_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def generate_rule_audit_report(
    manifest: dict,
    audit_result_files: Optional[tuple[tuple[str, tuple[str, ...]], ...]] = None,
) -> dict:
    """Generate a precision-audit report for every registered rule."""
    audit_result_files = audit_result_files or DEFAULT_AUDIT_RESULT_FILES
    hard_positive_hits = _fixture_pattern_hits(manifest["hard_positives"])
    hard_negative_hits = _fixture_pattern_hits(manifest["hard_negatives"])
    soft_negative_hits = _fixture_pattern_hits(manifest["soft_review_negatives"])

    sole_counts_by_source: dict[str, Counter] = {}
    sole_examples_by_source: dict[str, dict[str, list[dict]]] = {}
    for source_name, result_paths in audit_result_files:
        counts, examples = _load_rule_examples_from_results(result_paths, source_name)
        sole_counts_by_source[source_name] = counts
        sole_examples_by_source[source_name] = examples

    rules = {}
    hard_negative_total = len(manifest["hard_negatives"])
    for rule_name in AUDIT_RULE_ORDER:
        policy = get_rule_policy(rule_name)
        expected_positive_fixtures = list(policy.mapped_positive_fixtures)
        positive_hits = sorted(
            fixture_id
            for fixture_id in expected_positive_fixtures
            if rule_name in hard_positive_hits.get(fixture_id, set())
        )
        positive_misses = sorted(
            fixture_id
            for fixture_id in expected_positive_fixtures
            if rule_name not in hard_positive_hits.get(fixture_id, set())
        )
        negative_hits = sorted(
            fixture_id for fixture_id, patterns in hard_negative_hits.items() if rule_name in patterns
        )
        soft_hits = sorted(
            fixture_id for fixture_id, patterns in soft_negative_hits.items() if rule_name in patterns
        )
        rules[rule_name] = {
            "rule_name": rule_name,
            "technique_family": policy.technique_family,
            "evidence_tier": policy.evidence_tier,
            "max_outcome": policy.max_outcome,
            "requires_companion_patterns": list(policy.requires_companion_patterns),
            "mapped_positive_fixtures": list(policy.mapped_positive_fixtures),
            "observed_hard_positive_hits": sorted(
                fixture_id for fixture_id, patterns in hard_positive_hits.items() if rule_name in patterns
            ),
            "hard_positive_hits": positive_hits,
            "hard_positive_misses": positive_misses,
            "hard_negative_hits": negative_hits,
            "confusion_matrix": {
                "true_positive": len(positive_hits),
                "false_negative": len(positive_misses),
                "false_positive": len(negative_hits),
                "true_negative": hard_negative_total - len(negative_hits),
            },
            "soft_negative_hit_count": len(soft_hits),
            "soft_negative_hit_samples": soft_hits[:20],
            "sole_hit_frequency": {
                source_name: sole_counts_by_source[source_name].get(rule_name, 0)
                for source_name, _ in audit_result_files
            },
            "sole_hit_examples": [
                example
                for source_name, _ in audit_result_files
                for example in sole_examples_by_source[source_name].get(rule_name, [])
            ][:20],
            "final_verdict": policy.default_verdict,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_counts": {
            "hard_positives": len(manifest["hard_positives"]),
            "hard_negatives": len(manifest["hard_negatives"]),
            "soft_review_negatives": len(manifest["soft_review_negatives"]),
        },
        "rule_order": AUDIT_RULE_ORDER,
        "rules": rules,
    }


def write_rule_audit_report(output_dir: str, manifest: dict, report: dict) -> tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "classifier_fixture_manifest.json")
    report_json_path = os.path.join(output_dir, "rule_audit_report.json")
    report_md_path = os.path.join(output_dir, "rule_audit_report.md")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Rule Audit Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Hard positives: {report['manifest_counts']['hard_positives']}",
        f"- Hard negatives: {report['manifest_counts']['hard_negatives']}",
        f"- Soft review negatives: {report['manifest_counts']['soft_review_negatives']}",
        "",
    ]
    for rule_name in report["rule_order"]:
        rule = report["rules"][rule_name]
        lines.extend([
            f"## {rule_name}",
            f"- Technique family: `{rule['technique_family']}`",
            f"- Evidence tier: `{rule['evidence_tier']}`",
            f"- Max outcome: `{rule['max_outcome']}`",
            f"- Final verdict: `{rule['final_verdict']}`",
            f"- Hard positive hits: {len(rule['hard_positive_hits'])}",
            f"- Hard negative hits: {len(rule['hard_negative_hits'])}",
            f"- Confusion matrix: {json.dumps(rule['confusion_matrix'], sort_keys=True)}",
            f"- Soft negative hits: {rule['soft_negative_hit_count']}",
            f"- Sole-hit frequency: {json.dumps(rule['sole_hit_frequency'], sort_keys=True)}",
            "",
        ])
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return manifest_path, report_json_path, report_md_path


# ---------------------------------------------------------------------------
# Mode A: NvidiaArchive directory scan
# ---------------------------------------------------------------------------

def scan_nvidia_archive(directory: str, output_path: str):
    """Scan NvidiaArchive directory and output detection results."""
    py_files = sorted(glob.glob(os.path.join(directory, "nv_sub_*.py")))
    print(f"Scanning {len(py_files)} Python files in {directory}")

    results = []
    hash_groups = defaultdict(list)
    struct_groups = defaultdict(list)

    for filepath in py_files:
        basename = os.path.basename(filepath)
        # Parse submission ID: nv_sub_{id}_{name}.py
        parts = basename.split("_", 3)
        sub_id = parts[2] if len(parts) >= 3 else "unknown"

        code = open(filepath, encoding="utf-8", errors="replace").read()

        # Try loading runs.json for score metadata
        runs_path = os.path.join(directory, f"nv_sub_{sub_id}_runs.json")
        metadata = {"submission_id": sub_id}
        if os.path.exists(runs_path):
            try:
                with open(runs_path) as f:
                    runs = json.load(f)
                scores = _extract_nvidia_archive_scores(runs)
                if scores:
                    metadata["scores"] = scores
            except Exception:
                pass

        result = analyze_code(code, metadata, field="submission")
        result["submission_id"] = sub_id
        result["filename"] = basename
        result["lines"] = len(code.splitlines())

        ch = result["code_hash"]
        sh = result["structural_hash"]
        hash_groups[ch].append(sub_id)
        struct_groups[sh].append(sub_id)

        results.append(result)

    # Add duplicate / near-clone info
    for r in results:
        ch = r["code_hash"]
        sh = r["structural_hash"]
        exact_group = hash_groups[ch]
        struct_group = struct_groups[sh]
        if DUPLICATE_CONFIG.get("exact", True) and len(exact_group) > 1:
            r["matched_patterns"].append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"Code hash {ch} shared by {len(exact_group)} submissions",
                "field": "submission",
            })
            r["duplicate_count"] = len(exact_group)
        elif DUPLICATE_CONFIG.get("near_clone", True) and len(struct_group) > 1:
            # Near-clone: same structure, different identifier names
            r["matched_patterns"].append({
                "pattern": "NEAR_CLONE_SPAM",
                "severity": "medium",
                "evidence": f"Structural hash {sh} shared by {len(struct_group)} submissions (trivial rename)",
                "field": "submission",
            })
            r["near_clone_count"] = len(struct_group)
        # Reclassify after any new dedup pattern
        r["classification"], r["should_filter"] = classify(r["matched_patterns"])
        r["filter_reason"] = filter_reason(r["matched_patterns"]) if r["should_filter"] else None

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    classifications = Counter(r["classification"] for r in results)
    patterns = Counter()
    for r in results:
        for p in r["matched_patterns"]:
            patterns[p["pattern"]] += 1

    print(f"\n{'='*60}")
    print(f"NvidiaArchive Scan Results ({len(results)} files)")
    print(f"{'='*60}")
    print(f"\nClassifications:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nPattern hits:")
    for pat, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}")
    filtered_count = sum(1 for r in results if r["should_filter"])
    print(f"\nShould filter: {filtered_count}/{len(results)}")

    # Filter reason breakdown
    reason_counts = Counter(
        r.get("filter_reason", "none") for r in results if r["should_filter"]
    )
    if reason_counts:
        print(f"\nFilter reason breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # List valid (no-signal) files for manual spot-check
    valid = [r for r in results if r["classification"] == "valid"]
    if valid:
        print(f"\nValid (no signals) files ({len(valid)}):")
        for r in valid:
            print(f"  {r['filename']} ({r['lines']} lines)")

    print(f"\nResults written to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Mode B: JSONL dataset scan
# ---------------------------------------------------------------------------

def scan_jsonl(jsonl_path: str, results_path: str, cleaned_path: str, summary_path: str,
               workers: int = 0):
    """Stream-process JSONL dataset and output detection results + cleaned file.

    workers: number of parallel worker processes (0 = os.cpu_count()).
    """
    n_workers = workers if workers > 0 else (os.cpu_count() or 1)
    print(f"Scanning {jsonl_path}  [{n_workers} workers]")

    # Read all lines upfront so we can distribute to the pool.
    # Memory cost: ~300MB for 44K entries; acceptable.
    with open(jsonl_path, "r") as fin:
        raw_lines = list(enumerate(fin, 1))  # [(line_num, line), ...]

    total_lines = len(raw_lines)
    print(f"  Loaded {total_lines} lines, dispatching to pool...")

    all_results: list[dict] = []
    hash_groups_improved: dict = defaultdict(list)
    hash_groups_baseline: dict = defaultdict(list)
    patterns_improved: Counter = Counter()
    patterns_baseline: Counter = Counter()
    patterns_metadata: Counter = Counter()
    per_user = defaultdict(lambda: {"total": 0, "filtered": 0})
    per_problem = defaultdict(lambda: {"total": 0, "filtered": 0})

    chunksize = max(1, total_lines // (n_workers * 8))
    done = 0

    with mp.Pool(n_workers, initializer=_worker_pool_init, initargs=(ACTIVE_RUNTIME_CONFIG,)) as pool:
        for result in pool.imap_unordered(_worker_jsonl, raw_lines, chunksize=chunksize):
            if result is None:
                continue
            all_results.append(result)
            hash_groups_improved[result["code_hash_improved"]].append(result["id"])
            hash_groups_baseline[result["code_hash_baseline"]].append(result["id"])
            for p in result["matched_patterns"]:
                if p["field"] == "improved_code":
                    patterns_improved[p["pattern"]] += 1
                elif p["field"] == "baseline_code":
                    patterns_baseline[p["pattern"]] += 1
                else:
                    patterns_metadata[p["pattern"]] += 1
            per_user[result["user"]]["total"] += 1
            per_problem[result["problem_name"]]["total"] += 1
            done += 1
            if done % 5000 == 0:
                print(f"  Processed {done}/{total_lines}...")

    # Sort by original line order for deterministic output
    all_results.sort(key=lambda r: r["_line_num"])
    total = len(all_results)
    print(f"  Processed {total} entries total")

    filtered = 0
    kept = 0
    classifications: Counter = Counter()

    # Add exact-duplicate detection (NEAR_CLONE_SPAM skipped in JSONL mode for performance)
    for r in all_results:
        extra_patterns = []
        ch_imp = r["code_hash_improved"]
        ch_base = r["code_hash_baseline"]
        if DUPLICATE_CONFIG.get("exact", True) and len(hash_groups_improved.get(ch_imp, [])) > 1:
            extra_patterns.append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"improved_code hash {ch_imp} shared by {len(hash_groups_improved[ch_imp])} entries",
                "field": "improved_code",
            })
        if DUPLICATE_CONFIG.get("exact", True) and len(hash_groups_baseline.get(ch_base, [])) > 1:
            extra_patterns.append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"baseline_code hash {ch_base} shared by {len(hash_groups_baseline[ch_base])} entries",
                "field": "baseline_code",
            })
        if extra_patterns:
            r["matched_patterns"].extend(extra_patterns)
            decision_patterns = [
                p for p in r["matched_patterns"]
                if p.get("field") in ("improved_code", "metadata")
            ]
            r["classification"], r["should_filter"] = classify(decision_patterns)
            r["filter_reason"] = (
                filter_reason(decision_patterns) if r["should_filter"] else None
            )

    # Second pass: write results and cleaned JSONL
    print(f"  Writing results...")
    with open(results_path, "w") as fres:
        for r in all_results:
            fres.write(json.dumps(r) + "\n")
            classifications[r["classification"]] += 1
            if r["should_filter"]:
                per_user[r["user"]]["filtered"] += 1
                per_problem[r["problem_name"]]["filtered"] += 1

    # Write cleaned JSONL (re-read original and skip filtered lines)
    filtered_lines = {r["_line_num"] for r in all_results if r["should_filter"]}
    with open(jsonl_path, "r") as fin, open(cleaned_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            if line_num not in filtered_lines:
                fout.write(line)
                kept += 1
            else:
                filtered += 1

    # Build filter reason breakdown (counts post-dedup final decisions)
    filter_reason_counts: Counter = Counter()
    for r in all_results:
        if r["should_filter"] and r.get("filter_reason"):
            filter_reason_counts[r["filter_reason"]] += 1

    # Write summary
    summary = {
        "source_file": jsonl_path,
        "total_entries": total,
        "classifications": dict(classifications),
        "filtered": filtered,
        "kept": kept,
        "filter_reason_breakdown": dict(sorted(filter_reason_counts.items(), key=lambda x: -x[1])),
        "pattern_hits_improved_code": dict(sorted(patterns_improved.items(), key=lambda x: -x[1])),
        "pattern_hits_baseline_code": dict(sorted(patterns_baseline.items(), key=lambda x: -x[1])),
        "pattern_hits_metadata": dict(sorted(patterns_metadata.items(), key=lambda x: -x[1])),
        "per_problem": {
            k: v for k, v in sorted(per_problem.items())
        },
        "top_users_by_filtered": dict(
            sorted(
                ((u, d) for u, d in per_user.items() if d["filtered"] > 0),
                key=lambda x: -x[1]["filtered"],
            )[:20]
        ),
        "duplicate_clusters_improved": {
            h: len(ids) for h, ids in hash_groups_improved.items() if len(ids) > 5
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"JSONL Scan Results")
    print(f"{'='*60}")
    print(f"Total entries: {total}")
    print(f"\nClassifications:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nPattern hits (improved_code):")
    for pat, count in sorted(patterns_improved.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}")
    print(f"\nPattern hits (baseline_code):")
    for pat, count in sorted(patterns_baseline.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}")
    if patterns_metadata:
        print(f"\nPattern hits (metadata):")
        for pat, count in sorted(patterns_metadata.items(), key=lambda x: -x[1]):
            print(f"  {pat}: {count}")
    print(f"\nFiltered: {filtered}, Kept: {kept}")
    print(f"\nFilter reason breakdown:")
    for reason, count in sorted(filter_reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nResults: {results_path}")
    print(f"Cleaned: {cleaned_path}")
    print(f"Summary: {summary_path}")

    return all_results


# ---------------------------------------------------------------------------
# Mode C: Parquet dataset scan
# ---------------------------------------------------------------------------

def _iter_parquet_args(pf, cols: list, batch_size: int):
    """Generator: yield worker arg tuples one row at a time, reading the parquet in
    arrow batches so only one batch is live in the main process at a time."""
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for r in batch.to_pylist():
            yield (
                int(r["submission_id"]),
                int(r["leaderboard_id"]),
                str(r["user_id"]),
                str(r["user_name"]),
                str(r["problem_name"]),
                float(r["score"]) if r["score"] is not None else None,
                bool(r["passed"]),
                str(r["code"]) if r["code"] else "",
            )


def scan_parquet(parquet_path: str, results_path: str, best_path: str, summary_path: str,
                 workers: int = 0, batch_size: int = 2000):
    """Scan a submission parquet file using parallel workers.

    Streams via a generator so only batch_size rows are loaded at a time in the
    main process, keeping forked worker memory usage low.

    workers: parallel workers (0 = min(8, cpu_count) — capped to avoid OOM).
    batch_size: rows per arrow read batch.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required for --parquet mode: pip install pyarrow")

    # Default to min(8, cpu_count) — 32 workers OOMs on large files
    n_workers = workers if workers > 0 else min(8, os.cpu_count() or 1)
    print(f"Scanning {parquet_path}  [{n_workers} workers, batch_size={batch_size}]")

    pf = pq.ParquetFile(parquet_path)
    cols = ["submission_id", "leaderboard_id", "user_id", "user_name",
            "problem_name", "score", "passed", "code"]
    total = pf.metadata.num_rows
    print(f"  {total:,} rows in file")

    all_results: list[dict] = []
    hash_groups: dict = defaultdict(list)
    done = 0

    # Single imap_unordered over a generator — workers stay saturated, no idle
    # gaps between batches.  chunksize=50 keeps IPC overhead low.
    with mp.Pool(n_workers, initializer=_worker_pool_init, initargs=(ACTIVE_RUNTIME_CONFIG,)) as pool:
        for result in pool.imap_unordered(
            _worker_parquet,
            _iter_parquet_args(pf, cols, batch_size),
            chunksize=50,
        ):
            all_results.append(result)
            hash_groups[result["code_hash"]].append(result["submission_id"])
            done += 1
            if done % 10000 == 0:
                print(f"  Processed {done}/{total}...")

    print(f"  Processed {done:,} submissions total")

    # Attach duplicate spam
    for r in all_results:
        ch = r["code_hash"]
        if DUPLICATE_CONFIG.get("exact", True) and len(hash_groups.get(ch, [])) > 1:
            r["matched_patterns"].append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"code hash {ch} shared by {len(hash_groups[ch])} submissions",
                "field": "code",
            })
            r["classification"], r["should_filter"] = classify(r["matched_patterns"])
            r["filter_reason"] = filter_reason(r["matched_patterns"]) if r["should_filter"] else None

    # Sort by submission_id for deterministic output
    all_results.sort(key=lambda r: r["submission_id"])

    # Write per-submission results
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Best-per-user-per-problem: highest passing score, else highest score
    best: dict[tuple, dict] = {}
    for r in all_results:
        key = (r["user_id"], r["problem_name"])
        prev = best.get(key)
        sc = r.get("score") or 0.0
        if prev is None:
            best[key] = r
        else:
            prev_sc = prev.get("score") or 0.0
            # Prefer passing; among passing prefer lower score (faster); non-passing prefer lower too
            if (r.get("passed") and not prev.get("passed")) or \
               (r.get("passed") == prev.get("passed") and sc < prev_sc):
                best[key] = r

    with open(best_path, "w") as f:
        for r in sorted(best.values(), key=lambda r: (r["problem_name"], r.get("score") or 0)):
            f.write(json.dumps(r) + "\n")

    # Summary stats
    classifications: Counter = Counter(r["classification"] for r in all_results)
    filtered_count = sum(1 for r in all_results if r["should_filter"])
    patterns_all: Counter = Counter(
        p["pattern"] for r in all_results for p in r["matched_patterns"]
    )
    filter_reason_counts: Counter = Counter(
        r["filter_reason"] for r in all_results if r["should_filter"] and r.get("filter_reason")
    )
    per_problem: Counter = Counter(r["problem_name"] for r in all_results)
    best_classifications: Counter = Counter(r["classification"] for r in best.values())
    best_filtered = sum(1 for r in best.values() if r["should_filter"])

    summary = {
        "source_file": parquet_path,
        "total_submissions": total,
        "classifications": dict(classifications),
        "filtered": filtered_count,
        "filter_reason_breakdown": dict(sorted(filter_reason_counts.items(), key=lambda x: -x[1])),
        "pattern_hits": dict(sorted(patterns_all.items(), key=lambda x: -x[1])),
        "per_problem": dict(per_problem),
        "best_per_user_total": len(best),
        "best_per_user_classifications": dict(best_classifications),
        "best_per_user_filtered": best_filtered,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Parquet Scan Results")
    print(f"{'='*60}")
    print(f"Total submissions: {total}")
    print(f"\nClassifications:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nPattern hits:")
    for pat, count in sorted(patterns_all.items(), key=lambda x: -x[1])[:15]:
        print(f"  {pat}: {count}")
    print(f"\nFiltered: {filtered_count} ({100*filtered_count/total:.1f}%)")
    print(f"\nFilter reason breakdown:")
    for reason, count in sorted(filter_reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nBest-per-user ({len(best)} entries): {best_filtered} filtered ({100*best_filtered/len(best):.1f}%)")
    print(f"\nResults:  {results_path}")
    print(f"Best:     {best_path}")
    print(f"Summary:  {summary_path}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hacky Kernel Fingerprinting Pipeline")
    parser.add_argument("--jsonl", type=str, help="Path to JSONL dataset")
    parser.add_argument("--parquet", type=str, help="Path to submission parquet file")
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE_NAME,
        choices=BUILTIN_PROFILES,
        help="Built-in runtime profile to use before applying --config / --set overrides",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a TOML runtime config file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help='Override config values with dotted key=value syntax, e.g. entrypoints.names=["kernel"]',
    )
    parser.add_argument(
        "--export-config",
        nargs="?",
        const="-",
        help="Write the resolved runtime config as TOML and exit. Omit the path or use - for stdout.",
    )
    parser.add_argument(
        "--audit-rules",
        action="store_true",
        help="Build the precision-audit fixture manifest and rule audit report",
    )
    parser.add_argument(
        "--audit-archive-dir",
        dest="audit_archive_dir",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--audit-ground-truth-dir",
        dest="audit_ground_truth_dir",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--audit-manual-review",
        dest="audit_manual_review",
        action="append",
        default=[],
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--audit-filtered-results",
        dest="audit_filtered_results",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--audit-result",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker processes (default: os.cpu_count())")
    parser.add_argument(
        "--api-mode",
        action="store_true",
        help="Read submission code from stdin, output JSON result to stdout (for sidecar integration)",
    )
    args = parser.parse_args()

    try:
        resolved_config = configure_runtime(
            profile=args.profile,
            config_path=args.config,
            overrides=args.overrides,
        )
    except Exception as exc:
        parser.exit(2, f"Configuration error: {exc}\n")

    if args.export_config is not None:
        rendered = runtime_config_to_toml(resolved_config)
        if args.export_config == "-":
            sys.stdout.write(rendered)
        else:
            with open(args.export_config, "w", encoding="utf-8") as f:
                f.write(rendered)
            print(f"Exported runtime config to {args.export_config}")
        return

    if args.api_mode:
        import sys as _sys
        code = _sys.stdin.read()
        result = analyze_code(code, compute_structural_hash=False)
        print(json.dumps(result))
        return

    if not args.jsonl and not args.parquet and not args.audit_rules:
        parser.error("Must specify at least one of --jsonl, --parquet, or --audit-rules")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.audit_rules:
        try:
            audit_cfg = resolved_config.get("audit", {})
            audit_result_specs = args.audit_result or _audit_result_specs_from_config(resolved_config)
            audit_result_files = resolve_audit_result_files(audit_result_specs)
            audit_archive_dir = (
                _nonempty_str(args.audit_archive_dir)
                or _nonempty_str(audit_cfg.get("archive_dir"))
            )
            audit_ground_truth_dir = (
                _nonempty_str(args.audit_ground_truth_dir)
                or _nonempty_str(audit_cfg.get("ground_truth_dir"))
            )
            audit_filtered_results = (
                _nonempty_str(args.audit_filtered_results)
                or _nonempty_str(audit_cfg.get("filtered_results_path"))
            )
            audit_manual_review_paths = tuple(args.audit_manual_review) or tuple(
                path for path in audit_cfg.get("manual_review_files", []) if path
            ) or None
            manifest = build_classifier_fixture_manifest(
                nvidia_archive_dir=audit_archive_dir,
                amd_dir=audit_ground_truth_dir,
                filtered_nvidia_archive_path=audit_filtered_results,
                manual_judgments_path=(audit_manual_review_paths[0] if audit_manual_review_paths else None),
                extra_manual_review_paths=(audit_manual_review_paths[1:] if audit_manual_review_paths and len(audit_manual_review_paths) > 1 else None),
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"{exc}\n")
        manifest_counts = _manifest_fixture_counts(manifest)
        if not any(manifest_counts.values()):
            parser.exit(
                2,
                "No audit fixtures discovered. Run from a workspace containing the audit corpora "
                "or pass explicit audit corpus locations before using --audit-rules.\n",
            )
        report = generate_rule_audit_report(manifest, audit_result_files=audit_result_files)
        manifest_path, report_json_path, report_md_path = write_rule_audit_report(
            args.output_dir, manifest, report,
        )
        print("Precision audit complete")
        print(f"  Manifest: {manifest_path}")
        print(f"  Report:   {report_json_path}")
        print(f"  Summary:  {report_md_path}")

    if args.jsonl:
        results_path = os.path.join(args.output_dir, "detection_results_jsonl.jsonl")
        cleaned_path = os.path.join(args.output_dir, "cleaned_pairs.jsonl")
        summary_path = os.path.join(args.output_dir, "detection_summary.json")
        scan_jsonl(args.jsonl, results_path, cleaned_path, summary_path, workers=args.workers)

    if args.parquet:
        stem = os.path.splitext(os.path.basename(args.parquet))[0]
        results_path = os.path.join(args.output_dir, f"detection_results_{stem}.jsonl")
        best_path = os.path.join(args.output_dir, f"detection_results_{stem}_best.jsonl")
        summary_path = os.path.join(args.output_dir, f"detection_summary_{stem}.json")
        scan_parquet(args.parquet, results_path, best_path, summary_path, workers=args.workers)


if __name__ == "__main__":
    main()
