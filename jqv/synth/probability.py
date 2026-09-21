"""probability: acceptance sampling, screening posteriors, redundancy, expected value, supplier mix, draws.

All probabilities are exact Fractions from the stated facts. noul items that ask whether a random event will happen
carry `target_distribution` (the true P(yes)); deterministic questions (which option, which range) do not.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

from jqv.synth.common import Names, SynthItem, Trace, money, pct
from jqv.synth.temporal import fmt_date, rand_date, filler_lines

def procedure_paragraphs(rng: random.Random, names: Names, kind: str) -> list[str]:
    """Realistic procedural text that does not change the answer (JevBench probability states run 270-1,200 tokens)."""
    pool = {
        "inspection": [
            "Purpose. Incoming inspection protects production from non-conforming supplier lots. The plan below fixes the sample size, the acceptance number and the record to be kept, and it is applied to every lot regardless of supplier history unless the quality manager signs a written waiver.",
            f"Procedure. (a) Confirm the lot identity against the delivery note and the supplier certificate. (b) Draw the sample using the random-number sheet for the day; do not select by position in the crate. (c) Test each drawn unit on bench {rng.randint(1, 6)} using the released test program. (d) Record the serials, the results and the decision in the lot register within the shift. (e) Quarantine a rejected lot in area Q-{rng.randint(1, 9)} and raise a supplier corrective action request.",
            f"Records. The lot register is retained for {rng.choice([3, 5, 7])} years. Deviations from the plan, including sample sizes other than those specified, are reported to the quality manager ({names.person()}) and noted on the certificate of conformity.",
            "Supplier data. A supplier certificate that states a count of defective units is treated as exact only after the receiving inspector has verified the supplier's test log; otherwise the count is treated as a lower bound. For this lot the verification was completed and signed.",
        ],
        "screening": [
            "Programme background. The screening programme invites members of the enrolled population at fixed intervals. Results are reviewed by a clinician before any follow-up is arranged, and the review note must state the estimated probability of the condition given the result, using the programme's published prevalence figures and the validated test characteristics.",
            f"Test validation. Sensitivity and specificity were estimated in a validation study with the reference standard applied to every participant; the study report (version {rng.randint(2, 5)}) states that the estimates did not differ materially between age groups, so a single pair of values is used for all groups.",
            f"Follow-up policy. Where the estimated probability of the condition after the test exceeds {rng.choice([20, 25, 30])}%, a confirmatory examination is booked within {rng.choice([10, 14, 21])} days; otherwise the person is returned to routine recall. The estimate must combine the prior for the person's age group with the test result; it must not be read directly from the sensitivity or specificity.",
        ],
        "reliability": [
            "Scope. This review estimates the probability that the installation is unavailable at any point during the review period, for the purpose of deciding whether an additional standby unit is justified. The estimate uses the vendor's field reliability data and the redundancy rule in the maintenance contract.",
            f"Assumptions stated in the contract: failures of different units are independent; the failure rate is constant over the period; no repairs or replacements are carried out during the period; the review period starts on {fmt_date(rand_date(rng, 2025, 2027))}.",
            f"Reporting. The probability is reported to three decimals in the quarterly asset report, together with the number of units, the tolerated failures and the period length. The report is signed by the reliability engineer ({names.person()}).",
        ],
        "contract": [
            "Background. The procurement committee reviews contract options for the coming quarter on the basis of expected value. Historical frequencies from the previous eight to sixteen quarters are treated as probabilities for the coming quarter unless a documented change in operations makes the history irrelevant; no such change is recorded for this service.",
            f"Documentation. Each option's fee structure is copied from the supplier's offer letter dated {fmt_date(rand_date(rng, 2025, 2027))}. Bonuses and revenue shares are paid at quarter end; there is no cap and no floor beyond those stated.",
            "Committee rule. The committee does not apply a risk premium or a preference for fixed fees. Where two options have expected values within 1% of each other the incumbent supplier is preferred; this tie-break is recorded here for completeness.",
        ],
        "supplier": [
            "Purpose. When a defective unit is found in mixed stock the quality team estimates which supplier most likely produced it, so that the corrective-action request goes to the right supplier. Units carry no supplier marking once they are in stock.",
            f"Method. The estimate combines each supplier's share of the units in stock with that supplier's outgoing-inspection defect rate for the current quarter. Shares are taken from the stock ledger as of {fmt_date(rand_date(rng, 2025, 2027))}; defect rates are taken from the supplier scorecard (sample sizes above {rng.choice([500, 800, 1200])} units each).",
            "Note on interpretation. A high defect rate and a large share both matter; the team's standing instruction is to compute the joint probability for each supplier before comparing them.",
        ],
        "draw": [
            f"Procedure. The draw is used to select items for a spot check without any pattern that the floor staff could anticipate. The container is shaken for at least {rng.randint(10, 30)} seconds before each draw and the auditor draws without looking.",
            f"Record. The auditor records the sequence of drawn items on form SC-{rng.randint(10, 99)} and returns all items to the container only after the check is complete. Two witnesses initial the form.",
        ],
    }[kind]
    out = list(pool)
    rng.shuffle(out)
    return out[: rng.randint(2, len(out))]


def disagrees(p, surface_p, as_bin: bool) -> bool:
    """True when the distractor's number leads to a different answer than the truth."""
    if surface_p is None:
        return True
    if as_bin:
        return bin_of(p) is not None and bin_of(p) != bin_of(surface_p)
    return (p > Fraction(1, 2)) != (surface_p > Fraction(1, 2))


BINS = [("under_25_percent", Fraction(0), Fraction(1, 4)), ("25_to_50_percent", Fraction(1, 4), Fraction(1, 2)),
        ("50_to_75_percent", Fraction(1, 2), Fraction(3, 4)), ("over_75_percent", Fraction(3, 4), Fraction(1))]


def hypergeom_pmf(N: int, D: int, n: int, k: int) -> Fraction:
    return Fraction(comb(D, k) * comb(N - D, n - k), comb(N, n))


def hypergeom_at_least(N: int, D: int, n: int, c: int) -> Fraction:
    return sum((hypergeom_pmf(N, D, n, k) for k in range(c, min(n, D) + 1)), Fraction(0))


def binom_at_least(n: int, p: Fraction, k: int) -> Fraction:
    return sum((Fraction(comb(n, j)) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)), Fraction(0))


def f2(x: Fraction) -> str:
    return f"{float(x):.4f}"


def bin_of(p: Fraction) -> str | None:
    for name, lo, hi in BINS:
        if abs(p - lo) < Fraction(1, 50) or abs(p - hi) < Fraction(1, 50):
            return None  # too close to a boundary
        if lo <= p < hi:
            return name
    return None


def noul_event(family_state: str, instructions: str, yes: str, no: str, p: Fraction, tr: Trace, scenario: str, sig: str,
               distractor: str, surface_p: Fraction | None, names=None) -> SynthItem | None:
    if abs(p - Fraction(1, 2)) < Fraction(2, 25) or p < Fraction(3, 100) or p > Fraction(97, 100):
        return None
    expected = "yes" if p > Fraction(1, 2) else "no"
    surface = "" if surface_p is None else ("yes" if surface_p > Fraction(1, 2) else "no")
    return SynthItem(family="probability", scenario=scenario, state=family_state, qtype="noul", instructions=instructions,
                     criteria={"true": yes, "false": no}, expected=expected, trace=tr, signature=sig, distractor=distractor,
                     surface_answer=surface, target_distribution={"yes": float(p), "no": float(1 - p)})


def bin_choice(state: str, instructions: str, p: Fraction, tr: Trace, scenario: str, sig: str, distractor: str,
               surface_p: Fraction | None, what: str) -> SynthItem | None:
    b = bin_of(p)
    if b is None or p < Fraction(3, 100) or p > Fraction(97, 100):
        return None
    crit = {name: f"{what} is {'below 25%' if name == 'under_25_percent' else 'from 25% up to 50%' if name == '25_to_50_percent' else 'from 50% up to 75%' if name == '50_to_75_percent' else '75% or more'}."
            for name, _, _ in BINS}
    sb = bin_of(surface_p) if surface_p is not None else None
    return SynthItem(family="probability", scenario=scenario, state=state, qtype="choice", instructions=instructions,
                     criteria=crit, expected=b, trace=tr, signature=sig + "|bin", distractor=distractor, surface_answer=sb or "")


# ----------------------------------------------------------------------------- P1 acceptance sampling


def acceptance_sampling(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem | None:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Electronics", "Components", "Medical", "Manufacturing"])))
    lot = f.get("lot", names.ident(rng.choice(["LOT", "PSU", "BATCH"]), 6))
    N = f.get("N", rng.randint(8, 40))
    D = f.get("D", rng.randint(1, max(1, min(6, N // 3))))
    n_old, n_new = rng.sample([2, 3, 4, 5], 2)
    n_old, n_new = f.get("n_old", n_old), f.get("n_new", n_new)
    c = f.get("c", rng.choice([1, 1, 2]))
    cutover = f.get("cutover", rand_date(rng, 2025, 2027))
    received = f.get("received", cutover + (rng.randint(1, 20) if rng.random() < 0.6 else -rng.randint(1, 20)) * __import__("datetime").timedelta(days=1))
    n = n_new if received >= cutover else n_old
    if c > n or c > D:
        return None
    tr.given("lot", f"Lot of {N} units, exactly {D} defective.")
    tr.given("plans", f"Plan v{2}: draw {n_old}; plan v3: draw {n_new}; v3 applies to lots received on or after {fmt_date(cutover)}.")
    tr.given("received", f"Lot received {fmt_date(received)}.")
    tr.derive("plan", ["plans", "received"], f"The lot was received {'on or after' if received >= cutover else 'before'} {fmt_date(cutover)}, so the "
              f"{'v3' if n == n_new else 'v2'} plan applies: draw {n}, reject if at least {c} defective.")
    p0 = hypergeom_pmf(N, D, n, 0)
    chain = " x ".join(f"{N - D - i}/{N - i}" for i in range(n))
    tr.derive("p_none", ["plan", "lot"], f"P(no defective in {n} draws without replacement) = {chain} = {p0} = {f2(p0)}.")
    if c == 1:
        p = 1 - p0
        tr.derive("p_reject", ["p_none"], f"P(at least one defective) = 1 - {f2(p0)} = {f2(p)}.")
    else:
        p1 = hypergeom_pmf(N, D, n, 1)
        tr.derive("p_one", ["plan", "lot"], f"P(exactly one defective) = {p1} = {f2(p1)}.")
        p = 1 - p0 - p1
        tr.derive("p_reject", ["p_none", "p_one"], f"P(at least two defective) = 1 - {f2(p0)} - {f2(p1)} = {f2(p)}.")
    # distractors
    kind = f.get("kind", rng.choice(["old_plan", "with_replacement"]))
    n_wrong = n_old if n == n_new else n_new
    if kind == "old_plan":
        wp = hypergeom_at_least(N, D, n_wrong, c) if c <= n_wrong else Fraction(0)
        note = (f"Inspector's note ({names.person()}): with the {n_wrong}-unit draw the chance of catching at least {c} defective is about "
                f"{pct(round(float(wp), 3))}, which matches my gut feeling.")
    else:
        q = Fraction(D, N)
        wp = binom_at_least(n, q, c)
        note = (f"Inspector's note ({names.person()}): each draw has a {D}/{N} chance of being defective, so I treat the {n} draws as independent: "
                f"about {pct(round(float(wp), 3))} chance of at least {c} defective.")
    state = "\n".join([
        f"{org.upper()} — INCOMING INSPECTION, LOT {lot} ({N} {rng.choice(['power supply units', 'sensor modules', 'valve bodies', 'pump cartridges', 'PCB assemblies'])})",
        "",
        *filler_lines(rng, names, 2),
        *procedure_paragraphs(rng, names, "inspection"),
        f"Supplier certificate: the supplier's outgoing test recorded exactly {D} of the {N} units as defective ({rng.choice(['ripple outside the limit', 'leak rate over the limit', 'solder voids', 'flow out of tolerance'])}). "
        f"Defective units carry no marking and look identical to good ones; only a bench test tells them apart. Receiving has verified the count.",
        f"Sampling plan SP-{rng.randint(2, 9)}: under version 2 the inspector takes {n_old} units from the lot at random, one at a time, never putting a unit back, and tests them; "
        f"under version 3 the inspector takes {n_new} units the same way. With either version the lot is rejected when at least {c} of the tested units "
        f"{'is' if c == 1 else 'are'} defective. Version 3 applies to lots received on or after {fmt_date(cutover)}; earlier lots stay on version 2.",
        f"Lot {lot} received: {fmt_date(received, rng.randint(0, 2))}.",
        "",
        note,
    ])
    sig = f"{N}|{D}|{n_old}|{n_new}|{c}|{received}|{cutover}"
    r = f.get("r", rng.random())
    if "_retry" not in f and rng.random() < 0.9 and not disagrees(p, wp, as_bin=r >= 0.3):
        return None  # the caller draws again: we want items where the shortcut gives a different answer
    if r < 0.3:
        return noul_event(state, f"Will the inspection sample for lot {lot} contain at least {c} defective unit{'s' if c > 1 else ''} (so that the lot is rejected)? "
                          f"Answer with probabilities that follow from the facts above.",
                          f"At least {c} of the drawn units {'is' if c == 1 else 'are'} defective and the lot is rejected.",
                          f"Fewer than {c} of the drawn units {'is' if c == 1 else 'are'} defective and the lot is accepted.",
                          p, tr, "acceptance_sampling", sig, kind, wp)
    return bin_choice(state, f"Under the applicable sampling plan, which range contains the probability that lot {lot} is rejected?",
                      p, tr, "acceptance_sampling", sig, kind, wp, "The rejection probability")


# ----------------------------------------------------------------------------- P2 screening posterior


def screening_posterior(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem | None:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Medical", "Services", "Partners"])))
    person = names.person()
    groups = ["under 40", "40 to 59", "60 and over"]
    prev = f.get("prev", sorted(rng.sample([2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120], 3)))  # per 1,000
    g = f.get("g", rng.randrange(3))
    pi = Fraction(prev[g], 1000)
    sens = f.get("sens", Fraction(rng.choice([80, 85, 90, 92, 95, 97, 99]), 100))
    spec = f.get("spec", Fraction(rng.choice([85, 88, 90, 92, 94, 95, 97, 98, 99]), 100))
    positive = f.get("positive", rng.random() < 0.8)
    tr.given("groups", "Prevalence by age group: " + ", ".join(f"{gr}: {pv} per 1,000" for gr, pv in zip(groups, prev)) + ".")
    tr.given("person", f"{person} is in the {groups[g]} group; test result {'positive' if positive else 'negative'}.")
    tr.given("test", f"Sensitivity {pct(sens)}, specificity {pct(spec)}.")
    tr.derive("prior", ["groups", "person"], f"Prior for the {groups[g]} group: {prev[g]} per 1,000 = {f2(pi)}.")
    if positive:
        tp = pi * sens
        fp = (1 - pi) * (1 - spec)
        tr.derive("true_pos", ["prior", "test"], f"P(condition and positive) = {f2(pi)} x {pct(sens)} = {f2(tp)}.")
        tr.derive("false_pos", ["prior", "test"], f"P(no condition and positive) = {f2(1 - pi)} x {pct(1 - spec)} = {f2(fp)}.")
        post = tp / (tp + fp)
        tr.derive("posterior", ["true_pos", "false_pos"], f"P(condition | positive) = {f2(tp)} / ({f2(tp)} + {f2(fp)}) = {f2(post)}.")
        naive = sens
        naive_text = f"the test is {pct(sens)} sensitive, so a positive result means about a {pct(sens)} chance"
    else:
        fn = pi * (1 - sens)
        tn = (1 - pi) * spec
        tr.derive("false_neg", ["prior", "test"], f"P(condition and negative) = {f2(pi)} x {pct(1 - sens)} = {f2(fn)}.")
        tr.derive("true_neg", ["prior", "test"], f"P(no condition and negative) = {f2(1 - pi)} x {pct(spec)} = {f2(tn)}.")
        post = fn / (fn + tn)
        tr.derive("posterior", ["false_neg", "true_neg"], f"P(condition | negative) = {f2(fn)} / ({f2(fn)} + {f2(tn)}) = {f2(post)}.")
        naive = 1 - sens
        naive_text = f"the test misses {pct(1 - sens)} of cases, so the chance is about {pct(1 - sens)}"
    note = f"Clinic note ({names.person()}): {naive_text}."
    state = "\n".join([
        f"{org.upper()} — SCREENING PROGRAMME, RESULT REVIEW FOR {person.upper()}",
        "",
        *filler_lines(rng, names, 2),
        *procedure_paragraphs(rng, names, "screening"),
        f"Programme data sheet: prevalence of the condition in the screened population by age group — " +
        ", ".join(f"{gr}: {pv} per 1,000" for gr, pv in zip(groups, prev)) + ".",
        f"Test characteristics (validation study, n = {rng.randint(1200, 9800):,}): sensitivity {pct(sens)} (share of people with the condition who test positive); "
        f"specificity {pct(spec)} (share of people without the condition who test negative). Results are independent of age group given the condition status.",
        "",
        f"{person}, age group {groups[g]}, screened on {fmt_date(rand_date(rng, 2025, 2027), rng.randint(0, 2))}: result {'POSITIVE' if positive else 'NEGATIVE'}. "
        f"No other risk information is on file.",
        "",
        note,
    ])
    sig = f"{prev}|{g}|{sens}|{spec}|{positive}"
    r = f.get("r", rng.random())
    if "_retry" not in f and rng.random() < 0.85 and not disagrees(post, naive, as_bin=r >= 0.55):
        return None
    if r < 0.3:
        return noul_event(state, f"Does {person} actually have the condition? Answer with probabilities that follow from the facts above.",
                          "The person has the condition.", "The person does not have the condition.", post, tr, "screening_posterior", sig,
                          "accuracy_as_posterior", naive)
    if r < 0.55:
        more = post > Fraction(1, 2)
        tr.derive("verdict", ["posterior"], f"{f2(post)} is {'above' if more else 'below'} 0.5.")
        return SynthItem(family="probability", scenario="screening_posterior", state=state, qtype="noul",
                         instructions=f"Given the result, is it more likely than not that {person} has the condition?",
                         criteria={"true": "The probability of the condition given the result exceeds 50%.",
                                   "false": "The probability of the condition given the result is 50% or less."},
                         expected="yes" if more else "no", trace=tr, signature=sig + "|mln", distractor="accuracy_as_posterior",
                         surface_answer="yes" if naive > Fraction(1, 2) else "no")
    return bin_choice(state, f"Which range contains the probability that {person} has the condition, given the result?",
                      post, tr, "screening_posterior", sig, "accuracy_as_posterior", naive, "The probability of the condition")


# ----------------------------------------------------------------------------- P3 redundancy


def redundancy(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem | None:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Energy", "Systems", "Technologies", "Logistics"])))
    n = f.get("n", rng.choice([3, 4, 5, 6, 7, 8]))
    tol = f.get("tol", rng.randint(1, n - 1))  # tolerates up to tol failures; fails if >= tol+1 fail
    k = tol + 1
    years_per_fail = f.get("years_per_fail", rng.choice([3, 4, 5, 6, 8, 10, 12, 15, 20, 25]))
    months = f.get("months", rng.choice([3, 6, 9, 12, 15, 18, 24, 30, 36, 48]))
    p = Fraction(months, 12 * years_per_fail)
    if p >= 1:
        return None
    tr.given("units", f"{n} identical units; system fails if at least {k} fail (tolerates {tol}).")
    tr.given("rate", f"One failure per {years_per_fail} unit-years; period {months} months; failures independent.")
    tr.derive("p_unit", ["rate"], f"Per-unit failure probability over {months} months: ({months}/12)/{years_per_fail} = {p} = {f2(p)}.")
    pf = binom_at_least(n, p, k)
    terms = [f"C({n},{j}) {f2(p)}^{j} {f2(1 - p)}^{n - j}" for j in range(k, n + 1)]
    tr.derive("p_system", ["p_unit", "units"], f"P(at least {k} of {n} fail) = " + " + ".join(terms) + f" = {f2(pf)}.")
    kind = f.get("kind", rng.choice(["sum_rates", "single_unit"]))
    if kind == "sum_rates":
        wp = min(Fraction(1), n * p)
        note = f"Ops note ({names.person()}): {n} units at {pct(round(float(p), 4))} each is roughly {pct(round(float(n * p), 4))} for the system."
    else:
        wp = p
        note = f"Ops note ({names.person()}): the system is as reliable as one unit, so the risk is about {pct(round(float(p), 4))}."
    sysname = f.get("sysname", rng.choice(["chiller bank", "pump set", "UPS string", "compressor group", "server cluster"]))
    state = "\n".join([
        f"{org.upper()} — RELIABILITY REVIEW, {sysname.upper()} {names.ident('SYS', 4)}",
        "",
        *filler_lines(rng, names, 2),
        *procedure_paragraphs(rng, names, "reliability"),
        f"Configuration: {n} identical units. The {sysname} keeps operating as long as no more than {tol} unit{'s' if tol > 1 else ''} {'have' if tol > 1 else 'has'} failed; "
        f"it fails when {k} or more units have failed within the review period. Failures are independent between units.",
        f"Vendor reliability data: on average one failure per {years_per_fail} unit-years. Treat the failure probability of a unit over a period as "
        f"(period in years) / {years_per_fail}. Review period: the next {months} months, with no repairs during the period.",
        "",
        note,
    ])
    sig = f"{n}|{k}|{years_per_fail}|{months}"
    r = f.get("r", rng.random())
    if "_retry" not in f and rng.random() < 0.9 and not disagrees(pf, wp, as_bin=r >= 0.3):
        return None
    if r < 0.3:
        return noul_event(state, f"Will the {sysname} fail during the review period? Answer with probabilities that follow from the facts above.",
                          f"At least {k} of the {n} units fail within the period.", f"Fewer than {k} units fail and the {sysname} keeps operating.",
                          pf, tr, "redundancy", sig, kind, wp)
    return bin_choice(state, f"Which range contains the probability that the {sysname} fails during the review period?",
                      pf, tr, "redundancy", sig, kind, wp, "The system failure probability")


# ----------------------------------------------------------------------------- P4 expected value


def ev_choice(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem | None:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Logistics", "Services", "Partners", "Energy"])))
    fixed = f.get("fixed", rng.choice([1800, 2000, 2400, 2500, 3000, 3600, 4000]))
    base = f.get("base", int(fixed * rng.uniform(0.55, 0.85) / 50) * 50)
    bonus = f.get("bonus", int(fixed * rng.uniform(0.4, 1.2) / 50) * 50)
    hits, trials = rng.randint(2, 9), rng.randint(8, 16)
    q = Fraction(hits, trials)
    tr.given("a", f"Option A: fixed {money(fixed)}.")
    tr.given("b", f"Option B: {money(base)} plus {money(bonus)} if the target is met; target met in {hits} of the last {trials} quarters.")
    tr.derive("q", ["b"], f"P(target met) = {hits}/{trials} = {f2(q)}.")
    ev_b = base + bonus * q
    tr.derive("ev_b", ["q"], f"E[B] = {money(base)} + {money(bonus)} x {f2(q)} = {money(round(float(ev_b), 2))}.")
    three = f.get("three", rng.random() < 0.7)
    if three:
        share = Fraction(rng.choice([10, 12, 15, 20]), 100)
        rev_lo, rev_hi = sorted(rng.sample([8000, 10000, 12000, 15000, 18000, 20000, 25000], 2))
        p_hi = Fraction(rng.randint(2, 8), 10)
        tr.given("c", f"Option C: {pct(share)} of revenue; revenue {money(rev_hi)} with probability {f2(p_hi)}, otherwise {money(rev_lo)}.")
        ev_c = share * (p_hi * rev_hi + (1 - p_hi) * rev_lo)
        tr.derive("ev_c", ["c"], f"E[C] = {pct(share)} x ({f2(p_hi)} x {money(rev_hi)} + {f2(1 - p_hi)} x {money(rev_lo)}) = {money(round(float(ev_c), 2))}.")
        evs = {"option_a": Fraction(fixed), "option_b": ev_b, "option_c": ev_c}
    else:
        evs = {"option_a": Fraction(fixed), "option_b": ev_b}
    best = max(evs, key=evs.get)
    vals = sorted(evs.values(), reverse=True)
    if vals[0] - vals[1] < 40:
        return None
    tr.derive("best", list(k for k in ["a", "ev_b", "ev_c"] if k in {"a", "ev_b", "ev_c"} and (k != "ev_c" or three)),
              "Highest expected value: " + best.replace("_", " ").title() + ".")
    kind = f.get("kind", rng.choice(["ignore_bonus", "assume_bonus"]))
    if kind == "ignore_bonus":
        naive_b = Fraction(base)
        note = f"Manager's note ({names.person()}): the bonus is uncertain, so I compare {money(fixed)} with {money(base)} and pick accordingly."
    else:
        naive_b = Fraction(base + bonus)
        note = f"Manager's note ({names.person()}): if the target is met B pays {money(base + bonus)}, which beats A, so B it is."
    naive_evs = dict(evs); naive_evs["option_b"] = naive_b
    naive_best = max(naive_evs, key=naive_evs.get)
    lines = [f"Option A: a fixed quarterly fee of {money(fixed)}.",
             f"Option B: a quarterly fee of {money(base)} plus a bonus of {money(bonus)} in any quarter in which the service-level target is met. "
             f"The target was met in {hits} of the last {trials} quarters; treat that frequency as the probability for the coming quarter."]
    if three:
        lines.append(f"Option C: {pct(share)} of the quarter's attributable revenue. Finance estimates revenue of {money(rev_hi)} with probability {f2(p_hi)} "
                     f"and {money(rev_lo)} otherwise.")
    state = "\n".join([
        f"{org.upper()} — CONTRACT OPTION MEMO, {rng.choice(['MAINTENANCE', 'HAULAGE', 'SUPPORT', 'FACILITIES'])} SERVICES",
        "",
        *filler_lines(rng, names, 2),
        *procedure_paragraphs(rng, names, "contract"),
        "Decision rule from the procurement policy: choose the option with the highest expected value for the coming quarter; risk preferences are not considered.",
        *lines,
        "",
        note,
    ])
    sig = f"{fixed}|{base}|{bonus}|{hits}|{trials}|{three}|{sorted(evs.values())}"
    if f.get("r", rng.random()) < 0.7:
        crit = {k: f"{k.replace('_', ' ').title()} has the highest expected value." for k in evs}
        return SynthItem(family="probability", scenario="ev_choice", state=state, qtype="choice",
                         instructions="Under the procurement policy, which option should be chosen?", criteria=crit, expected=best,
                         trace=tr, signature=sig, distractor=kind, surface_answer=naive_best)
    higher = ev_b > fixed
    tr.derive("b_vs_a", ["ev_b", "a"], f"E[B] is {'higher' if higher else 'not higher'} than {money(fixed)}.")
    return SynthItem(family="probability", scenario="ev_choice", state=state, qtype="noul",
                     instructions="Does Option B have a higher expected value than Option A for the coming quarter?",
                     criteria={"true": "E[B] exceeds the fixed fee of Option A.", "false": "E[B] is at or below the fixed fee of Option A."},
                     expected="yes" if higher else "no", trace=tr, signature=sig + "|ba", distractor=kind,
                     surface_answer="yes" if naive_b > fixed else "no")


# ----------------------------------------------------------------------------- P5 supplier mix


def supplier_mix(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem | None:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Manufacturing", "Components", "Industries"])))
    k = f.get("k", rng.choice([3, 3, 3, 2]))
    sups = f.get("sups", [names.org(rng.choice(["Components", "Textiles", "Foods", "Engineering"])) for _ in range(k)])
    while len(set(sups)) < k:
        sups = [names.org(rng.choice(["Components", "Textiles", "Foods", "Engineering"])) for _ in range(k)]
    if k == 2:
        s1 = rng.choice([55, 60, 65, 70, 75, 80])
        shares = [Fraction(s1, 100), Fraction(100 - s1, 100)]
    else:
        a = rng.choice([40, 50, 60]); b = rng.choice([20, 25, 30]); shares = [Fraction(a, 100), Fraction(b, 100), Fraction(100 - a - b, 100)]
    rates = f.get("rates", [Fraction(rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12]), 100) for _ in range(k)])
    if len(set(rates)) < k:
        return None
    tr.given("shares", "Shares: " + ", ".join(f"{s} {pct(sh)}" for s, sh in zip(sups, shares)) + ".")
    tr.given("rates", "Defect rates: " + ", ".join(f"{s} {pct(r)}" for s, r in zip(sups, rates)) + ".")
    joints = [sh * r for sh, r in zip(shares, rates)]
    for i, (s, j) in enumerate(zip(sups, joints)):
        tr.derive(f"joint_{i}", ["shares", "rates"], f"P({s} and defective) = {pct(shares[i])} x {pct(rates[i])} = {f2(j)}.")
    total = sum(joints, Fraction(0))
    tr.derive("total", [f"joint_{i}" for i in range(k)], f"P(defective) = {' + '.join(f2(j) for j in joints)} = {f2(total)}.")
    posts = [j / total for j in joints]
    i_best = max(range(k), key=lambda i: posts[i])
    tr.derive("posterior", ["total"], "P(supplier | defective): " + ", ".join(f"{s} {f2(p)}" for s, p in zip(sups, posts)) + ".")
    i_rate = max(range(k), key=lambda i: rates[i])
    i_share = max(range(k), key=lambda i: shares[i])
    if "_retry" not in f and i_rate == i_best and i_share == i_best and rng.random() < 0.97:
        return None  # both shortcuts agree with the truth: not informative
    kinds = [kk for kk, ii in (("highest_rate", i_rate), ("largest_share", i_share)) if ii != i_best] or ["highest_rate", "largest_share"]
    kind = f.get("kind", rng.choice(kinds))
    if kind == "highest_rate":
        i_wrong = max(range(k), key=lambda i: rates[i])
        note = f"Quality note ({names.person()}): {sups[i_wrong]} has the worst defect rate, so a defective unit almost certainly came from them."
    else:
        i_wrong = max(range(k), key=lambda i: shares[i])
        note = f"Quality note ({names.person()}): most units come from {sups[i_wrong]}, so that is where a defective unit comes from."
    state = "\n".join([
        f"{org.upper()} — SUPPLIER QUALITY REVIEW, PART {names.ident('P', 5)}",
        "",
        *filler_lines(rng, names, 2),
        *procedure_paragraphs(rng, names, "supplier"),
        "Sourcing (share of units in stock): " + "; ".join(f"{s} {pct(sh)}" for s, sh in zip(sups, shares)) + ". Units are mixed in stock and unmarked.",
        "Outgoing-inspection history (defect rate by supplier): " + "; ".join(f"{s} {pct(r)}" for s, r in zip(sups, rates)) + ".",
        f"A unit drawn at random from stock on {fmt_date(rand_date(rng, 2025, 2027), rng.randint(0, 2))} was found defective.",
        "",
        note,
    ])
    sig = f"{shares}|{rates}|{k}"
    r = f.get("r", rng.random())
    labels = [f"supplier_{'abc'[i]}" for i in range(k)]
    if r < 0.3:
        crit = {l: f"The defective unit most likely came from {s}." for l, s in zip(labels, sups)}
        return SynthItem(family="probability", scenario="supplier_mix", state=state, qtype="choice",
                         instructions="Which supplier most likely produced the defective unit?", criteria=crit, expected=labels[i_best],
                         trace=tr, signature=sig, distractor=kind, surface_answer=labels[i_wrong])
    if r < 0.55:
        i_q = rng.randrange(k)
        return noul_event(state, f"Was the defective unit supplied by {sups[i_q]}? Answer with probabilities that follow from the facts above.",
                          f"The defective unit came from {sups[i_q]}.", f"The defective unit came from another supplier.",
                          posts[i_q], tr, "supplier_mix", sig + f"|s{i_q}", kind, Fraction(1) if i_q == i_wrong else Fraction(0))
    return bin_choice(state, "Which range contains the probability that a unit drawn at random from stock is defective?",
                      total, tr, "supplier_mix", sig + "|total", kind, None, "The defect probability")


# ----------------------------------------------------------------------------- P6 draw outcomes (distribution items)


def draw_outcomes(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem | None:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Foods", "Logistics", "Services"])))
    n_red, n_blue = f.get("n_red", rng.randint(2, 15)), f.get("n_blue", rng.randint(2, 15))
    N = n_red + n_blue
    thing = f.get("thing", rng.choice(["tokens", "sample vials", "sealed envelopes", "keys", "pallets tags"]))
    cA, cB = rng.choice([("red", "blue"), ("marked", "unmarked"), ("priority", "standard"), ("sealed", "open"), ("green", "yellow"), ("numbered", "blank")])
    tr.given("bag", f"{n_red} {cA} and {n_blue} {cB} {thing}; two drawn without replacement.")
    p_aa = Fraction(n_red, N) * Fraction(n_red - 1, N - 1)
    p_bb = Fraction(n_blue, N) * Fraction(n_blue - 1, N - 1)
    p_ab = 1 - p_aa - p_bb
    tr.derive("p_aa", ["bag"], f"P(both {cA}) = {n_red}/{N} x {n_red - 1}/{N - 1} = {f2(p_aa)}.")
    tr.derive("p_bb", ["bag"], f"P(both {cB}) = {n_blue}/{N} x {n_blue - 1}/{N - 1} = {f2(p_bb)}.")
    tr.derive("p_ab", ["p_aa", "p_bb"], f"P(one of each) = 1 - {f2(p_aa)} - {f2(p_bb)} = {f2(p_ab)}.")
    dist = {"both_first_kind": p_aa, "one_of_each": p_ab, "both_second_kind": p_bb}
    ranked = sorted(dist.values(), reverse=True)
    if ranked[0] - ranked[1] < Fraction(1, 20):
        return None
    best = max(dist, key=dist.get)
    # distractor: with-replacement numbers
    q_aa = Fraction(n_red, N) ** 2; q_bb = Fraction(n_blue, N) ** 2; q_ab = 1 - q_aa - q_bb
    wdist = {"both_first_kind": q_aa, "one_of_each": q_ab, "both_second_kind": q_bb}
    wbest = max(wdist, key=wdist.get)
    note = (f"Floor note ({names.person()}): each draw is {n_red}/{N} {cA}, so both {cA} is {pct(round(float(q_aa), 3))}, both {cB} is "
            f"{pct(round(float(q_bb), 3))}, and mixed is {pct(round(float(q_ab), 3))}.")
    state = "\n".join([
        f"{org.upper()} — RANDOM DRAW PROCEDURE, {rng.choice(['DAILY AUDIT', 'SPOT CHECK', 'ALLOCATION ROUND'])} {names.ident('RD', 5)}",
        "",
        *filler_lines(rng, names, 2),
        *procedure_paragraphs(rng, names, "draw"),
        f"The container holds {n_red} {cA} {thing} and {n_blue} {cB} {thing}, thoroughly mixed. The auditor draws two {thing} at random, one after the other, "
        f"without putting the first one back.",
        "",
        note,
    ])
    sig = f"{n_red}|{n_blue}|{cA}"
    if f.get("r", rng.random()) < 0.6:
        tr.derive("most_likely", ["p_ab"], f"Most likely outcome: {best.replace('_', ' ')}.")
        crit = {"both_first_kind": f"Both drawn {thing} are {cA}.", "one_of_each": f"One {cA} and one {cB}.",
                "both_second_kind": f"Both drawn {thing} are {cB}."}
        return SynthItem(family="probability", scenario="draw_outcomes", state=state, qtype="choice",
                         instructions=f"Which outcome will the draw produce? Answer with probabilities that match the chances of each outcome.",
                         criteria=crit, expected=best, trace=tr, signature=sig, distractor="with_replacement", surface_answer=wbest,
                         target_distribution={k: float(v) for k, v in dist.items()})
    p_any = 1 - p_bb
    tr.derive("p_any", ["p_bb"], f"P(at least one {cA}) = 1 - {f2(p_bb)} = {f2(p_any)}.")
    return noul_event(state, f"Will at least one of the two drawn {thing} be {cA}? Answer with probabilities that follow from the facts above.",
                      f"At least one drawn item is {cA}.", f"Both drawn items are {cB}.", p_any, tr, "draw_outcomes", sig + "|any",
                      "with_replacement", 1 - q_bb)


SCENARIOS = [acceptance_sampling, screening_posterior, redundancy, ev_choice, supplier_mix, draw_outcomes]
WEIGHTS = [0.22, 0.26, 0.18, 0.14, 0.14, 0.06]


def make_item(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    """Pick a scenario by weight, then redraw its parameters until it yields an item, so that scenarios with a high
    rejection rate (the distractor must disagree with the truth) keep their intended share."""
    fn = rng.choices(SCENARIOS, weights=WEIGHTS)[0]
    for _ in range(60):
        item = fn(rng, names, facts)
        if item is not None:
            return item
    for _ in range(20):  # fallback: any scenario
        item = rng.choices(SCENARIOS, weights=WEIGHTS)[0](rng, names, facts)
        if item is not None:
            return item
    raise RuntimeError("no valid probability item after 80 draws")
