"""
Answer equivalence checking and grouping.

Provides mathematical equivalence checking for answers and a consolidated
grouping function that replaces 4 separate implementations from the old codebase.
"""

from functools import lru_cache
from math import isclose
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynasor.core.evaluator import math_equal

# Lazy import for SymPy (expensive to load)
_sympy_loaded = False
_parse_latex = None
_latex2sympy = None
_sympy_N = None


def _ensure_sympy():
    """Lazy load SymPy modules."""
    global _sympy_loaded, _parse_latex, _latex2sympy, _sympy_N
    if not _sympy_loaded:
        from sympy import N as sympy_N
        from sympy.parsing.latex import parse_latex
        from latex2sympy2 import latex2sympy
        _parse_latex = parse_latex
        _latex2sympy = latex2sympy
        _sympy_N = sympy_N
        _sympy_loaded = True


@lru_cache(maxsize=10000)
def _math_equal_cached(a: str, b: str) -> bool:
    """Cached wrapper around math_equal to avoid redundant SymPy calls."""
    return math_equal(a, b)


def _normalize_quick(s: str) -> str:
    """Quick normalization for fast path comparison."""
    s = s.strip().lower()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace(" ", "")
    return s


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse string as a simple number."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


@lru_cache(maxsize=10000)
def canonical_numeric_key(answer: str) -> Tuple[Optional[str], bool]:
    """Try to canonicalize answer to a numeric key.

    Returns:
        (canonical_key, is_numeric): If numeric, key is a rounded float string.
    """
    if answer is None:
        return (None, False)

    simple_num = _try_parse_number(answer)
    if simple_num is not None:
        return (f"{simple_num:.9g}", True)

    _ensure_sympy()

    s = answer.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    for parser in [_parse_latex, _latex2sympy]:
        try:
            expr = parser(s)
            val = complex(_sympy_N(expr, 15))
            if val.imag == 0:
                return (f"{val.real:.9g}", True)
        except:
            continue

    return (None, False)


def _is_plausible_math(s: str) -> bool:
    """Quick check whether a string could be a mathematical expression.

    Rejects long strings with no mathematical content to avoid expensive
    SymPy parsing on garbage. Checks the first 50 chars so answers with
    correct values followed by trailing garbage still pass.
    """
    s = s.strip()
    prefix = s[:50]
    if any(c.isdigit() for c in prefix):
        return True
    if any(c in prefix for c in "\\^{}/_"):
        return True
    return len(s) <= 30


def answers_equivalent(a: str, b: str) -> bool:
    """Check if two answers are mathematically equivalent.

    Fast paths tried first:
    1. Exact string match
    2. Case-insensitive match
    3. Quick normalized match
    4. Numeric comparison
    5. Plausibility filter (skip garbage strings)
    6. Full Dynasor math_equal (symbolic)
    """
    if a is None or b is None:
        return False

    if a == b:
        return True

    a_stripped = a.strip().lower()
    b_stripped = b.strip().lower()
    if a_stripped == b_stripped:
        return True

    a_norm = _normalize_quick(a)
    b_norm = _normalize_quick(b)
    if a_norm == b_norm:
        return True

    a_num = _try_parse_number(a)
    b_num = _try_parse_number(b)
    if a_num is not None and b_num is not None:
        return isclose(a_num, b_num, rel_tol=1e-4)

    if not _is_plausible_math(a) or not _is_plausible_math(b):
        return False

    return _math_equal_cached(a, b)


def group_equivalent_answers(
    answers: List[Optional[str]],
    max_symbolic_pairwise: int = 50,
) -> Tuple[Dict[int, str], np.ndarray]:
    """Group equivalent answers into integer IDs.

    Consolidated from 4 separate implementations:
    - deepconf_voting._hybrid_grouping_counts/scores
    - adaptive_margin.build_hybrid_canonical_map
    - deepconf_offline_vectorized._group_equivalent_answers

    Algorithm:
        1. Numeric fast path: canonicalize via canonical_numeric_key, hash-group  O(n)
        2. Symbolic fallback: pairwise answers_equivalent on remaining            O(m^2)
        3. Cross-check: symbolic answers against numeric representatives          O(m*k)

    Args:
        answers: List of answer strings (may contain None)
        max_symbolic_pairwise: Max symbolic answers for O(m^2) comparison

    Returns:
        canonical_map: {group_id: representative_string}
        answer_ids: ndarray[int16] of shape [len(answers)], -1 for None answers
    """
    n = len(answers)
    answer_ids = np.full(n, -1, dtype=np.int16)

    if n == 0:
        return {}, answer_ids

    # Collect unique non-None answers
    unique_answers = set()
    for a in answers:
        if a is not None:
            unique_answers.add(a)

    if not unique_answers:
        return {}, answer_ids

    # Step 1: Separate numeric and symbolic
    numeric_groups: Dict[str, List[str]] = {}  # canonical_key -> list of equivalent strings
    symbolic_answers: List[str] = []

    for ans in unique_answers:
        key, is_numeric = canonical_numeric_key(ans)
        if is_numeric:
            if key not in numeric_groups:
                numeric_groups[key] = []
            numeric_groups[key].append(ans)
        else:
            symbolic_answers.append(ans)

    # Step 2: Build groups list
    groups: List[List[str]] = []

    # Add numeric groups
    for key, members in numeric_groups.items():
        groups.append(members)

    # Step 3: Symbolic grouping — pairwise comparison O(m^2)
    max_cross_check_k = 20
    if len(symbolic_answers) > max_symbolic_pairwise:
        for ans in symbolic_answers:
            groups.append([ans])
    else:
        used = set()
        for ans in symbolic_answers:
            if ans in used:
                continue
            group = [ans]
            used.add(ans)

            for other in symbolic_answers:
                if other not in used:
                    try:
                        if answers_equivalent(ans, other):
                            group.append(other)
                            used.add(other)
                    except:
                        continue

            # Cross-check against the top-K largest existing groups.
            # The correct answer is almost always among the most popular
            # groups, so this catches real merges without O(m*n) cost
            # on the long tail of rare garbage answers.
            merged = False
            if groups:
                top_k_indices = sorted(
                    range(len(groups)),
                    key=lambda i: len(groups[i]),
                    reverse=True,
                )[:max_cross_check_k]
                for gid in top_k_indices:
                    try:
                        if answers_equivalent(ans, groups[gid][0]):
                            groups[gid].extend(group)
                            merged = True
                            break
                    except:
                        continue

            if not merged:
                groups.append(group)

    # Step 4: Build mappings
    canonical_map: Dict[int, str] = {}
    answer_to_id: Dict[str, int] = {}

    for group_id, group in enumerate(groups):
        representative = group[0]
        canonical_map[group_id] = representative
        for ans in group:
            answer_to_id[ans] = group_id

    # Step 5: Map all answers to IDs
    for i, ans in enumerate(answers):
        if ans is not None and ans in answer_to_id:
            answer_ids[i] = answer_to_id[ans]

    return canonical_map, answer_ids
