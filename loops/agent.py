"""Bounded LLM agent client (S22 D6).

Invokes the Claude Code CLI (`claude --print`) as a subprocess for judgment
calls (capital allocation), the CodeGraph-AI parallel_review.py pattern — no SDK
dependency. The agent ONLY selects among engine-enumerated options; its output is
re-validated against constraints before use, and ANY failure (CLI absent, parse
error, constraint violation) returns None so the caller falls back deterministically.
Every request/response is persisted for audit when a LoopState is provided.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class AgentRequest:
    schema_version: int
    decision_type: str
    loop_id: str
    period: str
    mandate: dict
    inputs: dict
    options: list[str]          # legal children the agent may weight
    constraints: dict


@dataclass
class AgentResponse:
    selection: dict             # e.g. {"weights": {"L2.crypto": 0.6, ...}}
    confidence: float = 0.0
    rationale: str = ""
    flags: list[str] = field(default_factory=list)


class AgentClient:
    def __init__(
        self, command: str = "claude", timeout: int = 300,
        enabled: bool = True, state: Any = None,
    ) -> None:
        self.command = command
        self.timeout = timeout
        self.enabled = enabled
        self.state = state

    # ── public judgment call ──────────────────────────────────────────────
    def allocate(self, mandate, prior_reports, children) -> Optional[dict]:
        """Return validated weights dict, or None to trigger deterministic fallback."""
        if not self.enabled or shutil.which(self.command) is None:
            return None
        options = [c.loop_id for c in children]
        req = AgentRequest(
            schema_version=1, decision_type="allocate",
            loop_id=getattr(mandate, "issued_by", "") or "",
            period=getattr(mandate, "period", ""),
            mandate=mandate.to_dict() if mandate else {},
            inputs={cid: (r.to_dict() if r else None) for cid, r in prior_reports.items()},
            options=options,
            constraints={"weights_sum_to": 1.0, "min": 0.0, "max": 1.0, "keys": options},
        )
        resp = self._invoke(req)
        weights = self._validate(resp, options) if resp else None
        if self.state is not None:
            self.state.record_agent_call(
                req.period, req.decision_type, asdict(req),
                asdict(resp) if resp else {"error": "no_response"},
            )
        return weights

    # ── subprocess + parsing ──────────────────────────────────────────────
    def _invoke(self, req: AgentRequest) -> Optional[AgentResponse]:
        prompt = self._build_prompt(req)
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)  # avoid nested-session guard
        try:
            proc = subprocess.run(
                [self.command, "--print"], input=prompt, capture_output=True,
                text=True, timeout=self.timeout, env=env,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None
        return self._parse(proc.stdout)

    @staticmethod
    def _build_prompt(req: AgentRequest) -> str:
        return (
            "You are a capital allocator. Given the mandate and each child loop's "
            "last report, return ONLY a JSON object of the form "
            '{"weights": {"<loop_id>": <0..1>, ...}, "confidence": <0..1>, '
            '"rationale": "<short>"}. Weights MUST cover exactly these loop_ids '
            f"and sum to 1.0: {req.options}.\n\nREQUEST:\n"
            + json.dumps(asdict(req), indent=2)
        )

    @staticmethod
    def _parse(stdout: str) -> Optional[AgentResponse]:
        s = stdout.strip()
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            d = json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
        if "weights" not in d:
            return None
        return AgentResponse(
            selection={"weights": d["weights"]},
            confidence=float(d.get("confidence", 0.0)),
            rationale=str(d.get("rationale", "")),
            flags=list(d.get("flags", [])),
        )

    # ── constraint re-validation (engine has final say) ───────────────────
    @staticmethod
    def _validate(resp: AgentResponse, options: list[str]) -> Optional[dict]:
        w = resp.selection.get("weights", {})
        if set(w) != set(options):
            return None
        try:
            vals = [float(v) for v in w.values()]
        except (TypeError, ValueError):
            return None
        if any(v < 0.0 or v > 1.0 for v in vals):
            return None
        total = sum(vals)
        if total <= 0 or abs(total - 1.0) > 0.02:
            return None
        return {k: float(v) / total for k, v in w.items()}  # renormalize
