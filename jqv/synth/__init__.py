"""Synthetic decision items for the JevBench hard families that jqv is weak on.

Every item's answer is produced by a solver (rule engine, calendar arithmetic, exact probability), never by a
language model. The solver records a derivation trace, from which `dependency_hops` (number of derived
intermediate facts) and `reasoning_depth` (longest derivation chain) are computed, so multi-hop difficulty is an
attribute of every item rather than a separate family.
"""

from jqv.synth.common import Trace, SynthItem  # noqa: F401

FAMILIES = ("long_policy", "temporal_numeric", "probability", "temporal_v2")
