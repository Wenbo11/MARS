"""
Probed trace data structures for adaptive margin-based early stopping.

Provides dataclasses and loaders for probed trace data, where each trace has
been probed at specific token positions to extract intermediate answers.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProbedTrace:
    """A single trace with probed intermediate answers.

    Attributes:
        trace_id: Unique identifier matching the NPZ trace index
        final_answer: The final answer extracted from complete trace
        num_tokens: Total number of tokens in the trace
        probe_results: Mapping from probe position to extracted answer
    """
    trace_id: int
    final_answer: Optional[str]
    num_tokens: int
    probe_results: Dict[int, str] = field(default_factory=dict)

    def get_answer_at_position(self, position: int) -> Optional[str]:
        """Get the answer at a specific probe position.

        Returns answer at position if probed, final_answer if position >= num_tokens,
        None otherwise.
        """
        if position in self.probe_results:
            return self.probe_results[position]
        if position >= self.num_tokens:
            return self.final_answer
        return None


@dataclass
class ProbedQuestion:
    """A question with all its probed traces.

    Attributes:
        question_id: Unique identifier matching the NPZ question ID
        traces: List of probed traces for this question
        probe_positions: Sorted list of all probe positions across traces
    """
    question_id: int
    traces: List[ProbedTrace] = field(default_factory=list)
    probe_positions: List[int] = field(default_factory=list)

    @property
    def num_traces(self) -> int:
        return len(self.traces)


def load_probed_trace(trace_path: Path) -> ProbedTrace:
    """Load a single probed trace from a JSON file."""
    with open(trace_path) as f:
        data = json.load(f)

    probe_results = {}
    for pos_str, result in data.get('prob_token_results', {}).items():
        pos = int(pos_str)
        answer = result.get('answer')
        if answer is not None:
            probe_results[pos] = answer

    return ProbedTrace(
        trace_id=data['trace_id'],
        final_answer=data.get('final_answer'),
        num_tokens=data.get('num_tokens', 0),
        probe_results=probe_results,
    )


def load_probed_question(data_dir: Path, qid: int) -> Optional[ProbedQuestion]:
    """Load all probed traces for a question.

    Searches for probed trace directories matching the pattern
    `probed_traces/*_qid{qid}_*/trace_*.json`.
    """
    probed_base = data_dir / "probed_traces"
    if not probed_base.exists():
        return None

    target_dir = None
    for d in probed_base.iterdir():
        if d.is_dir() and f"_qid{qid}_" in d.name:
            target_dir = d
            break

    if target_dir is None:
        return None

    traces = []
    all_positions = set()

    for trace_file in sorted(target_dir.glob("trace_*.json")):
        trace = load_probed_trace(trace_file)
        traces.append(trace)
        all_positions.update(trace.probe_results.keys())

    if not traces:
        return None

    return ProbedQuestion(
        question_id=qid,
        traces=traces,
        probe_positions=sorted(all_positions),
    )
