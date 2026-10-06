"""Structure for rq1/inject.py (PRIVATE, git-ignored). Copy this file to inject.py and
write the payloads there. Never commit inject.py or paste payload text into the paper.

inject(order, attack_class, channel, rng) returns an UntrustedText with the payload in the
given channel and realistic benign text in the other slots.
followed(plan, correct, attack_class) decides whether the returned plan contains the injected
change (the core RQ1 metric): compare structurally to the correct plan, per attack class.
"""
from __future__ import annotations

from rq1.prompt import UntrustedText

ATTACK_CLASSES = ["C1_wrong_variant", "C2_inspection_omission", "C3_param_outside_quality_window",
                  "C4_operation_reorder", "C5_throughput_sabotage", "C6_denial_of_service_holds"]


def benign_text(order, rng) -> UntrustedText:
    raise NotImplementedError("realistic, harmless notes / labels / comments per slot")


def inject(order, attack_class, channel, rng) -> UntrustedText:
    raise NotImplementedError("payload families per attack class and delivery style; keep private")


def followed(plan: dict, correct: dict, attack_class: str) -> bool:
    raise NotImplementedError("structural diff against the correct plan for this attack class")
