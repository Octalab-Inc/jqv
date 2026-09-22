"""Solver checks for the synthetic hard-family generators: hand-computed cases through the `facts` override, helper
functions on known values, item invariants, and the loader round trip."""

from __future__ import annotations

import random
from datetime import date, time
from fractions import Fraction
from pathlib import Path

import pytest

from jqv.synth import policy, probability, temporal, temporal_v2
from jqv.synth.common import Names, SplitWriter, Trace
from jqv.synth.generate import generate_family


def _rng(seed=0):
    r = random.Random(seed)
    return r, Names(r)


# ----------------------------------------------------------------------------- Trace


def test_trace_hops_and_depth():
    t = Trace()
    t.given("a", "a"); t.given("b", "b")
    t.derive("c", ["a", "b"], "c"); t.derive("d", ["c"], "d"); t.derive("e", ["a"], "e")
    assert t.hops == 3 and t.depth == 2
    assert t.rationale().startswith("(1) c (2) d")
    with pytest.raises(KeyError):
        t.derive("f", ["zzz"], "f")


# ----------------------------------------------------------------------------- temporal helpers and scenarios


def test_add_months_month_end_rule():
    assert temporal.add_months_same_day(date(2026, 8, 31), 18) == (date(2028, 2, 29), True)   # leap year
    assert temporal.add_months_same_day(date(2025, 8, 31), 18) == (date(2027, 2, 28), True)
    assert temporal.add_months_same_day(date(2026, 11, 30), 30) == (date(2029, 5, 30), False)
    assert temporal.add_months_same_day(date(2026, 1, 31), 1) == (date(2026, 2, 28), True)


def test_business_days_after_skips_weekends_and_holidays():
    # Thu 2027-08-19 + 5 business days with Wed 2027-08-25 a holiday -> Fri 2027-08-27
    due, skipped = temporal.business_days_after(date(2027, 8, 19), 5, {date(2027, 8, 25)})
    assert due == date(2027, 8, 27)
    assert any("holiday" in s for s in skipped) and sum("weekend" in s for s in skipped) == 2


def test_warranty_leap_day_edge():
    rng, names = _rng(1)
    base = {"delivery": date(2026, 8, 31), "n_months": 18, "home_city": ("Lisbon", "Europe/Lisbon"), "same_tz": True, "variant": 0.1}
    early = temporal.warranty_month_end_tz(rng, names, {**base, "delta_minutes": -30})
    assert early.expected == "yes" and "29 February 2028" in early.trace.rationale()
    late = temporal.warranty_month_end_tz(rng, names, {**base, "delta_minutes": 30})
    assert late.expected == "no"
    choice = temporal.warranty_month_end_tz(rng, names, {**base, "delta_minutes": -30, "variant": 0.6})
    assert choice.qtype == "choice" and choice.expected == "2028_02_29"


def test_warranty_timezone_conversion():
    rng, names = _rng(2)
    # home Oslo (UTC+1 in Feb), claim from Toronto (UTC-5): 30 minutes before the Oslo midnight deadline is in time
    f = {"delivery": date(2026, 8, 31), "n_months": 18, "home_city": ("Oslo", "Europe/Oslo"), "same_tz": False, "variant": 0.1}
    it = temporal.warranty_month_end_tz(rng, names, {**f, "delta_minutes": -30})
    assert it.expected == "yes"
    it = temporal.warranty_month_end_tz(rng, names, {**f, "delta_minutes": 90})
    assert it.expected == "no"


def test_business_day_deadline_after_hours():
    rng, names = _rng(3)
    f = {"received": date(2027, 8, 19), "rec_time": time(20, 30), "cutoff": time(16, 0), "n_days": 5, "holidays": {date(2027, 8, 25)}}
    # effective receipt Fri 20 Aug; +5 business days skipping Wed 25 Aug and the weekend -> Fri 27 Aug (20->23,24,26,27,30) = 30 Aug
    it = temporal.business_day_deadline(rng, names, {**f, "sent": date(2027, 8, 30)})
    if it.qtype == "noul":
        assert it.expected == "yes"
    else:
        assert it.expected == "2027_08_30"
    it2 = temporal.business_day_deadline(rng, names, {**f, "sent": date(2027, 8, 31)})
    if it2.qtype == "noul":
        assert it2.expected == "no"


def test_service_months_leave_reset():
    rng, names = _rng(4)
    f = {"hire": date(2024, 1, 31), "ref": date(2026, 3, 30), "limit": 30, "a": 6, "b": 12, "c": 24}
    it = temporal.service_months_band(rng, names, {**f, "leave_days": 0})
    # 31 Jan 2024 -> anniversaries on the last day of short months; 25 complete months by 30 Mar 2026 (the 26th completes 31 Mar)
    assert ": 25 (" in it.trace.rationale()
    it = temporal.service_months_band(rng, names, {**f, "leave_days": 45})
    assert "restarts" in it.trace.rationale()


def test_prorated_invoice_arithmetic():
    rng, names = _rng(5)
    from decimal import Decimal
    f = {"fee": Decimal(1200), "start": date(2025, 3, 1), "end": date(2025, 4, 1), "inclusive": False, "discount": Decimal(10), "admin": Decimal(50)}
    it = temporal.prorated_invoice(rng, names, f)
    # 31 days / 365 * 1200 = 101.9178..; -10% = 91.7260..; +50 = 141.73
    assert it.expected == "amount_141_73"


def test_dst_cutoff_conversion():
    rng, names = _rng(6)
    f = {"home_city": ("Dubai", "Asia/Dubai"), "other_city": ("Helsinki", "Europe/Helsinki"), "cutoff": time(17, 30), "d0": date(2027, 11, 2)}
    from datetime import timedelta
    it = temporal.dst_cutoff(rng, names, {**f, "delta": timedelta(minutes=-30)})
    assert it.expected in ("yes", "t_17_00")
    it = temporal.dst_cutoff(rng, names, {**f, "delta": timedelta(minutes=30)})
    assert it.expected in ("no", "t_18_00")


# ----------------------------------------------------------------------------- probability


def test_hypergeom_and_binomial():
    assert probability.hypergeom_at_least(12, 3, 3, 1) == Fraction(34, 55)   # JevBench-style example
    assert probability.hypergeom_at_least(12, 3, 2, 1) == Fraction(60, 132)
    assert probability.binom_at_least(3, Fraction(1, 2), 2) == Fraction(1, 2)


def test_acceptance_sampling_uses_the_version_in_force():
    rng, names = _rng(7)
    f = {"N": 12, "D": 3, "n_old": 1, "n_new": 3, "c": 1, "cutover": date(2026, 9, 1), "received": date(2026, 9, 18), "r": 0.1, "_retry": 0}
    it = probability.acceptance_sampling(rng, names, f)
    assert it is not None and it.qtype == "noul" and it.expected == "yes"
    assert abs(it.target_distribution["yes"] - 34 / 55) < 1e-9
    it = probability.acceptance_sampling(rng, names, {**f, "received": date(2026, 8, 18)})
    assert it is not None and abs(it.target_distribution["yes"] - 3 / 12) < 1e-9 and it.expected == "no"


def test_screening_posterior_bayes():
    rng, names = _rng(8)
    f = {"prev": [2, 15, 30], "g": 2, "sens": Fraction(92, 100), "spec": Fraction(88, 100), "positive": True, "r": 0.1, "_retry": 0}
    it = probability.screening_posterior(rng, names, f)
    assert it is not None and it.expected == "no"
    post = (0.03 * 0.92) / (0.03 * 0.92 + 0.97 * 0.12)
    assert abs(it.target_distribution["yes"] - post) < 1e-9


def test_redundancy_binomial():
    rng, names = _rng(9)
    f = {"n": 3, "tol": 1, "years_per_fail": 10, "months": 24, "_retry": 0}  # _retry skips the "distractor must disagree" resampling
    it = probability.redundancy(rng, names, f)
    assert it is not None
    p = Fraction(1, 5)
    expect = 3 * p ** 2 * (1 - p) + p ** 3
    assert f"{float(expect):.4f}" in it.trace.rationale()


def test_draw_outcomes_distribution_sums_to_one():
    rng, names = _rng(10)
    it = probability.draw_outcomes(rng, names, {"n_red": 9, "n_blue": 6})
    assert it is not None
    if it.qtype == "choice":
        assert abs(sum(it.target_distribution.values()) - 1) < 1e-9 and it.expected == "one_of_each"
    else:
        assert abs(it.target_distribution["yes"] - (1 - Fraction(6, 15) * Fraction(5, 14))) < 1e-9


# ----------------------------------------------------------------------------- long_policy rule engines


HW = {"R": 14, "V": 60, "S1": 10000, "S2": 15000, "ded": 1000, "endorsement_date": date(2026, 1, 1), "renewal": date(2026, 3, 15),
      "duration": 35, "concealed": True, "known": False, "furnished": True, "absence": 75, "permit": False, "loss": 22400, "r": 0.1}


def test_home_water_rule_engine():
    rng, names = _rng(11)
    it = policy.home_water(rng, names, HW)  # furnished -> unoccupied, not vacant; concealed seepage -> sublimit 15,000 (endorsed)
    assert it.expected == "pay_within_15000_sublimit" and it.extra["payable"] == 15000
    it = policy.home_water(rng, names, {**HW, "renewal": date(2025, 12, 1)})
    assert it.expected == "pay_within_10000_sublimit" and it.extra["payable"] == 10000
    it = policy.home_water(rng, names, {**HW, "furnished": False})
    assert it.expected == "deny_vacant_dwelling"
    it = policy.home_water(rng, names, {**HW, "furnished": False, "permit": True})
    assert it.expected == "pay_within_15000_sublimit"
    it = policy.home_water(rng, names, {**HW, "known": True})
    assert it.expected == "deny_repeated_leakage"
    it = policy.home_water(rng, names, {**HW, "duration": 5})
    assert it.expected == "pay_estimate_less_deductible" and it.extra["payable"] == 21400
    it = policy.home_water(rng, names, {**HW, "absence": 40, "furnished": False, "duration": 5, "loss": 3000})
    assert it.expected == "pay_estimate_less_deductible" and it.extra["payable"] == 2000


def test_equipment_breakdown_rule_engine():
    rng, names = _rng(12)
    f = {"W": 6, "M": 45, "S1": 20000, "S2": 30000, "ded": 2500, "endorsement_date": date(2026, 1, 1), "inception": date(2026, 2, 1),
         "signs_months": 8, "surge": True, "overdue": 10, "booked": False, "loss": 41000, "r": 0.1}
    it = policy.equipment_breakdown(rng, names, f)
    assert it.expected == "pay_within_30000_sublimit" and it.extra["payable"] == 30000
    it = policy.equipment_breakdown(rng, names, {**f, "surge": False})
    assert it.expected == "deny_wear_and_tear"
    it = policy.equipment_breakdown(rng, names, {**f, "overdue": 60})
    assert it.expected == "deny_maintenance_lapse"
    it = policy.equipment_breakdown(rng, names, {**f, "overdue": 60, "booked": True})
    assert it.expected == "pay_within_30000_sublimit"
    it = policy.equipment_breakdown(rng, names, {**f, "signs_months": 2})
    assert it.expected == "pay_full_repair_less_deductible" and it.extra["payable"] == 38500


def test_trip_cancellation_rule_engine():
    rng, names = _rng(13)
    f = {"L": 90, "K": 60, "D": 14, "S": 5000, "cost": 8000, "deposit": date(2026, 4, 1), "purchase": date(2026, 4, 10), "cfar": True,
         "reason": "medical", "last_treat_days_before": 30, "stable_days": 70, "r": 0.1}
    it = policy.trip_cancellation(rng, names, f)
    assert it.expected == "pay_within_5000_sublimit"
    it = policy.trip_cancellation(rng, names, {**f, "stable_days": 20})
    assert it.expected == "pay_cfar_75_percent" and it.extra["payable"] == 6000
    it = policy.trip_cancellation(rng, names, {**f, "stable_days": 20, "purchase": date(2026, 4, 20)})  # rider bought 19 days after deposit -> void
    assert it.expected == "deny_pre_existing_condition"
    it = policy.trip_cancellation(rng, names, {**f, "last_treat_days_before": 120})
    assert it.expected == "pay_full_nonrefundable_costs"
    it = policy.trip_cancellation(rng, names, {**f, "reason": "advisory", "adv_offset": -5})
    assert it.expected == "pay_cfar_75_percent"
    it = policy.trip_cancellation(rng, names, {**f, "reason": "advisory", "adv_offset": 5})
    assert it.expected == "pay_full_nonrefundable_costs"


def test_relocation_rule_engine():
    rng, names = _rng(14)
    f = {"Dkm": 50, "T": 90, "caps_old": {"G1": 3000, "G2": 6000, "G3": 10000}, "bump": 1500, "supplement": 1500, "amend_date": date(2026, 1, 1),
         "grade": "G2", "start": date(2026, 2, 1), "old_commute": 10, "new_commute": 65, "dependants": True, "leave_days": 0, "claim_delay": 80, "expenses": 9500}
    it = policy.relocation_reimbursement(rng, names, f)
    assert it.expected == "reimburse_capped_at_9000" and it.extra["payable"] == 9000   # 7,500 amended cap + 1,500 supplement
    it = policy.relocation_reimbursement(rng, names, {**f, "expenses": 8000})
    assert it.expected == "reimburse_in_full"
    it = policy.relocation_reimbursement(rng, names, {**f, "new_commute": 55})
    assert it.expected == "deny_distance_test"
    it = policy.relocation_reimbursement(rng, names, {**f, "claim_delay": 95})
    assert it.expected == "deny_late_claim"
    it = policy.relocation_reimbursement(rng, names, {**f, "claim_delay": 95, "leave_days": 14})
    assert it.expected == "reimburse_capped_at_9000"
    it = policy.relocation_reimbursement(rng, names, {**f, "start": date(2025, 12, 1)})
    assert it.expected == "reimburse_capped_at_7500"


def test_sla_rule_engine():
    rng, names = _rng(15)
    inc = [("INC-1", "unplanned", 30), ("INC-2", "scheduled_in_window", 120), ("INC-3", "customer_caused", 60), ("INC-4", "force_majeure", 240)]
    f = {"month_minutes": 43200, "amend_date": date(2026, 1, 1), "renewal": date(2025, 6, 1), "incidents": inc, "r": 0.1}
    it = policy.sla_credits(rng, names, f)
    # denominator 43200-120-240 = 42840; downtime 30 -> uptime 99.930% -> no credit on the original tiers (99.9)
    assert it.expected == "no_credit" and it.extra["uptime"] == "99.930"
    it = policy.sla_credits(rng, names, {**f, "renewal": date(2026, 2, 1)})   # amended tiers: 99.95 threshold -> 5% credit
    assert it.expected == "credit_5_percent"


# ----------------------------------------------------------------------------- invariants and loader


@pytest.mark.parametrize("mod", [temporal, probability, policy, temporal_v2])
def test_generated_items_are_valid(mod):
    rng, names = _rng(99)
    hops = []
    for _ in range(40):
        it = mod.make_item(rng, names)
        it.validate()
        assert it.expected in it.labels
        assert it.trace.hops >= 1 and it.trace.depth >= 1 and it.trace.rationale()
        assert it.surface_answer == "" or it.surface_answer in it.labels
        hops.append(it.trace.hops)
    assert len(set(hops)) >= 2


def test_split_writer_dedups_signatures(tmp_path):
    rng, names = _rng(5)
    w = SplitWriter("temporal_numeric", tmp_path, 0)
    it = temporal.make_item(rng, names)
    assert w.add("train", it) and not w.add("dev", it)


def test_generate_and_load_round_trip(tmp_path):
    from jqv.data import load_synth

    generate_family("probability", {"train": 6, "dev": 3, "test": 3}, 0, tmp_path)
    items = load_synth("probability", "dev", root=tmp_path)
    assert len(items) == 3
    for it in items:
        assert it["choices"][it["answer"]].split(":")[0] == it["labels"][it["answer"]]
        assert "dependency_hops" in it["meta"] and it["qtype"] in ("choice", "noul")
    assert (tmp_path / "probability" / "summary.json").exists()


# ----------------------------------------------------------------------------- temporal_v2 solvers (hand-computed cases)


def test_v2_term_vs_cap_classification():
    from decimal import Decimal  # noqa: F401
    base = {"made": date(2024, 9, 2), "n_term": 24, "cap_months": 30, "delivery": date(2024, 12, 10), "has_repair": True,
            "r_in": date(2025, 6, 1), "r_out": date(2025, 6, 10), "code_style": False, "shift_date": False, "variant": 0.1, "_retry": 0,
            "desk_city": ("Oslo", "Europe/Oslo"), "cust_city": ("Osaka", "Asia/Tokyo")}
    # term: 10 Dec 2026 + 10 repair days (1-10 June inclusive) = 20 Dec 2026; cap: 30 months from 2 Sep 2024 = 2 Mar 2027
    rng, names = _rng(1)
    it = temporal_v2.term_vs_cap(rng, names, {**base, "claim": date(2026, 12, 15)})
    assert it.expected == "covered" and it.scenario == "term_vs_cap"
    rng, names = _rng(1)
    it = temporal_v2.term_vs_cap(rng, names, {**base, "claim": date(2026, 12, 25)})
    assert it.expected == "expired_term"
    # cap binds: 27 months from 1 Jan 2024 = 1 Apr 2026 < term 1 June 2026 (no repair); claim 1 May 2026
    rng, names = _rng(2)
    it = temporal_v2.term_vs_cap(rng, names, {**base, "made": date(2024, 1, 1), "cap_months": 27, "delivery": date(2024, 6, 1),
                                               "has_repair": False, "claim": date(2026, 5, 1)})
    assert it.expected == "expired_cap"
    rng, names = _rng(3)
    it = temporal_v2.term_vs_cap(rng, names, {**base, "claim": date(2026, 12, 15), "variant": 0.9})
    assert it.qtype == "choice" and it.expected == "2026_12_20"


def test_v2_fx_lines_cap_amount():
    from decimal import Decimal

    rates = {date(2026, 10, 7): Decimal("165.20"), date(2026, 10, 8): Decimal("164.10"), date(2026, 10, 9): Decimal("163.50"),
             date(2026, 10, 12): Decimal("160.30"), date(2026, 10, 13): Decimal("161.00")}
    f = {"home": "EUR", "cur": ("JPY", Decimal("162"), Decimal("0"), 0), "check_in": date(2026, 10, 8), "nights": 3, "rates": rates, "tx_rule": "checkout",
         "cap": Decimal(170), "room_lines": [(date(2026, 10, 8), Decimal(26000)), (date(2026, 10, 9), Decimal(29500)), (date(2026, 10, 10), Decimal(24000))],
         "tax_fx": Decimal(300), "excluded": [("Minibar", Decimal(1800))], "variant": 0.1, "_retry": 0}
    rng, names = _rng(4)
    it = temporal_v2.fx_lines_cap(rng, names, f)
    # check-out Sunday 11 Oct -> Friday's 163.50; 159.02 + min(180.43, 170) + 146.79 + 3 x 1.83 = 481.30
    assert it.expected == "eur_481_30" and it.qtype == "choice" and "eur_481_30" in it.criteria
    rng, names = _rng(4)
    it = temporal_v2.fx_lines_cap(rng, names, {**f, "variant": 0.6, "thr_lo": Decimal(400), "thr_hi": Decimal(500)})
    assert it.expected == "manager"


def test_v2_effective_expiry_earliest_condition():
    f = {"eff": date(2026, 3, 15), "k_months": 24, "has_transfer": True, "transfer": date(2027, 1, 10), "grace": 30, "has_notice": True,
         "notice_date": date(2026, 12, 1), "received": date(2026, 12, 3), "notice_period": 90, "has_fee": True, "paid": None,
         "query": date(2027, 2, 20), "variant": 0.1, "_retry": 0}
    rng, names = _rng(5)
    it = temporal_v2.effective_expiry(rng, names, f)  # transfer + 30 = 9 Feb 2027 < notice 3 Mar 2027 < fee 15 Mar 2027 < fixed 15 Mar 2028
    assert it.expected == "transfer_event"
    rng, names = _rng(5)
    it = temporal_v2.effective_expiry(rng, names, {**f, "variant": 0.5})
    assert it.expected == "2027_02_09"
    rng, names = _rng(5)
    it = temporal_v2.effective_expiry(rng, names, {**f, "variant": 0.9})
    assert it.qtype == "noul" and it.expected == "no"
    rng, names = _rng(6)
    it = temporal_v2.effective_expiry(rng, names, {**f, "has_transfer": False, "variant": 0.1})  # notice from receipt: 3 Dec + 90 = 3 Mar 2027
    assert it.expected == "termination_notice"


def test_v2_deadline_boolean_counting_roll_and_zone():
    f = {"event": date(2026, 3, 2), "n_days": 30, "count_event_day": False, "roll": True, "holidays": {date(2026, 4, 1)},
         "office_city": ("Toronto", "America/Toronto"), "filer_city": ("Warsaw", "Europe/Warsaw"), "same_tz": False, "cutoff": time(17, 0),
         "grace": 5, "variant": 0.5, "_retry": 0}
    # day 30 = 1 April (Wed, closed) -> Thursday 2 April 2026, 17:00 Toronto (EDT) = 23:00 Warsaw (CEST)
    rng, names = _rng(7)
    it = temporal_v2.deadline_boolean(rng, names, {**f, "delta_min": -30})
    assert it.expected == "on_time" and "2 April 2026" in it.trace.rationale()
    rng, names = _rng(7)
    it = temporal_v2.deadline_boolean(rng, names, {**f, "delta_min": 60})
    assert it.expected == "late_with_fee"
    rng, names = _rng(7)
    it = temporal_v2.deadline_boolean(rng, names, {**f, "delta_min": 8 * 24 * 60})
    assert it.expected == "rejected"
    rng, names = _rng(8)
    it = temporal_v2.deadline_boolean(rng, names, {**f, "count_event_day": True, "roll": False, "delta_min": -30, "variant": 0.9})
    assert it.expected == "2026_03_31"  # counting 2 March as day 1, day 30 is 31 March; no roll


def test_v2_multi_condition_tiers():
    from decimal import Decimal

    quarters = [(date(2027, 6, 21), Decimal(5300)), (date(2027, 3, 22), Decimal(5900)), (date(2026, 12, 21), Decimal(3700)),
                (date(2026, 9, 21), Decimal(5300)), (date(2026, 6, 22), Decimal(1500))]
    payments = [(date(2027, 6, 5), date(2027, 6, 5)), (date(2027, 8, 3), date(2027, 8, 2)), (date(2026, 7, 3), date(2026, 7, 10))]  # the late one is outside the window
    f = {"assess": date(2027, 9, 20), "opened": date(2025, 5, 20), "quarters": quarters, "payments": payments, "referrals": 3, "suspended": False,
         "A": Decimal(20000), "B": 12, "C": 4, "structure": "and_with_or", "variant": 0.1, "_retry": 0}
    rng, names = _rng(9)
    it = temporal_v2.multi_condition(rng, names, f)  # spend in window 20,200 >= 20,000; tenure 28 >= 12; no late in window -> gold
    assert it.expected == "gold"
    rng, names = _rng(9)
    it = temporal_v2.multi_condition(rng, names, {**f, "suspended": True})
    assert it.expected == "standard"
    rng, names = _rng(9)
    it = temporal_v2.multi_condition(rng, names, {**f, "structure": "or_of_ands", "referrals": 5, "A": Decimal(25000)})  # (spend fails) OR (5 refs AND no late)
    assert it.expected == "gold"
    rng, names = _rng(9)
    it = temporal_v2.multi_condition(rng, names, {**f, "variant": 0.9})
    assert it.qtype == "score" and it.expected == "3"


def test_v2_program_keys_are_split_disjoint(tmp_path):
    import json

    generate_family("temporal_v2", {"train": 15, "dev": 6, "test": 6}, 0, tmp_path, apply_paraphrase=False)
    keys = {}
    for split in ("train", "dev", "test"):
        for line in (tmp_path / "temporal_v2" / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            assert r["meta"]["program_key"] and r["meta"]["version"] == "v2"
            keys.setdefault(split, set()).add(r["meta"]["program_key"])
    assert not (keys["train"] & keys["test"]) and not (keys["train"] & keys["dev"]) and not (keys["dev"] & keys["test"])


# ----------------------------------------------------------------------------- paraphrase helpers (no model)


def test_paraphrase_candidates_and_acceptance():
    from jqv.synth import paraphrase as pp

    rng, names = _rng(31)
    it = policy.home_water(rng, names)
    cands = pp.candidate_paragraphs(it.state)
    lines = it.state.split("\n")
    assert cands and all(not lines[i].strip().isupper() and not lines[i].startswith("=") for i in cands)
    orig = "The insured returned after 75 consecutive days away on 10 September 2025 and found damage; the estimate is $22,400 (claim CLM-1796389)."
    assert pp.accept(orig, "After 75 consecutive days away, the insured came back on 10 September 2025 to find damage estimated at $22,400 (claim CLM-1796389).")
    assert not pp.accept(orig, "After 57 consecutive days away, the insured came back on 10 September 2025 to find damage estimated at $22,400 (claim CLM-1796389).")
    assert not pp.accept(orig, "Here is the rewritten paragraph: " + orig)
    assert not pp.accept(orig, "Short.")


# ----------------------------------------------------------------------------- mixed training sources


def test_mix_parsing_and_per_batch_source_sampling():
    from jqv.train.data import MixSampler, load_source, parse_mix

    assert parse_mix("synth:long_policy=0.25, mmlu=0.3") == [("synth:long_policy", 0.25), ("mmlu", 0.3)]
    items = load_source("synth:probability:dev")
    assert len(items) == 300 and set(items[0]) >= {"state", "question", "choices", "answer"}
    rng = random.Random(0)
    sampler = MixSampler({"a": [{"v": i} for i in range(10)], "b": [{"v": 100 + i} for i in range(5)]}, {"a": 0.5, "b": 0.5}, rng)
    for _ in range(20):
        batch = sampler.next(4, lambda it: it["v"])
        assert len(batch) == 4 and (all(v < 100 for v in batch) or all(v >= 100 for v in batch))  # one source per batch
    assert sampler.next(3, lambda it: None if it["v"] % 2 else it["v"]) and all(v % 2 == 0 for v in sampler.next(3, lambda it: None if it["v"] % 2 else it["v"]))
