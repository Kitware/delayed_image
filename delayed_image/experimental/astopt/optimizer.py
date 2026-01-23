"""Experimental AST-based optimizer driver."""

from __future__ import __annotations__

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Tuple

from delayed_image.experimental.astopt import rules
from delayed_image.experimental.astopt.signature import node_signature
from delayed_image.experimental.astopt.transformer import get_children, rebuild


@dataclass
class OptimizeTrace:
    applied: list[dict] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def record(self, rule_name, node, new_node):
        self.applied.append({
            "rule": rule_name,
            "node": node.__class__.__name__,
            "new_node": new_node.__class__.__name__,
        })
        self.counts[rule_name] += 1


class ASTOptimizer:
    """AST optimizer with rule-based local rewrites."""

    def __init__(self, trace: OptimizeTrace | None = None, legacy_fallback: bool = True):
        self.trace = trace
        self.legacy_fallback = legacy_fallback
        self._memo: Dict[int, Tuple[object, str]] = {}

    def optimize(self, node):
        return self._optimize_node(node)

    def _optimize_node(self, node):
        node_id = id(node)
        if node_id in self._memo:
            return self._memo[node_id][0]

        replacements = {}
        child_signatures = []
        for path, child in get_children(node):
            new_child = self._optimize_node(child)
            if new_child is not child:
                replacements[path] = new_child
            child_signatures.append(node_signature(new_child, []))

        if replacements:
            node = rebuild(node, replacements)

        node = self._apply_rules(node)

        if self.legacy_fallback:
            node = node.optimize()

        signature = node_signature(node, child_signatures)
        self._memo[node_id] = (node, signature)
        return node

    def _apply_rules(self, node):
        max_iters = 20
        for _ in range(max_iters):
            changed = False
            for rule in rules.rules_for(node):
                new_node, did_change, name = rule(node)
                if did_change:
                    if self.trace is not None and name:
                        self.trace.record(name, node, new_node)
                    node = new_node
                    changed = True
                    break
            if not changed:
                break
        return node


def optimize(node, **kwargs):
    """Optimize a delayed node using the experimental AST optimizer."""
    trace = kwargs.pop("trace", None)
    legacy_fallback = kwargs.pop("legacy_fallback", True)
    optimizer = ASTOptimizer(trace=trace, legacy_fallback=legacy_fallback)
    return optimizer.optimize(node)


def optimize_trace(node, **kwargs):
    """Optimize and return the trace of applied rules."""
    trace = OptimizeTrace()
    kwargs["trace"] = trace
    result = optimize(node, **kwargs)
    return result, trace
