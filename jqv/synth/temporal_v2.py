"""temporal_v2: the computations that decide a choice in JevBench-style temporal / numeric items.

Written after the targeted-LoRA runs (docs/report.ja.md, 32B section): heads trained on the v1 temporal scenarios
(month-end rule, business days, DST, pro-rating, unit conversion) got *worse* on JevBench temporal_numeric at 14B and
32B. The items they lost need a different kind of arithmetic: a contract term against an absolute cap counted from
another date, foreign-currency lines converted at the right table rate and capped line by line, an expiry that is
the earliest of several conditions, before/after-deadline booleans with inclusive/exclusive counting, and tiers
defined by AND / OR over several derived numbers. Each scenario here solves such a computation with the standard
library, records a Trace, and adds a desk note that argues for a wrong reading.

The signature of an item is its program key (the solved inputs only, no names, fillers or question variant), so the
split writer keeps one computation in exactly one split.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from jqv.synth.common import Names, SynthItem, Trace
from jqv.synth.temporal import MONTHS, add_months_same_day, filler_lines, fmt_date, fmt_time, rand_date, snake_date, utc_offset_str

_ID5 = __import__("re").compile(r"\b(INV|REF|CASE)-(\d{5})\b")


def fillers(rng: random.Random, names: Names) -> list[str]:
    """The shared filler lines with six-digit reference numbers, so they cannot collide with JevBench's five-digit ids."""
    return [_ID5.sub(lambda m: f"{m.group(1)}-{m.group(2)}{rng.randint(0, 9)}", line) for line in filler_lines(rng, names)]

FAMILY = "temporal_v2"
CENT = Decimal("0.01")


def q2(x) -> Decimal:
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def amt(cur: str, x) -> str:
    return f"{cur} {q2(x):,.2f}"


def amt_label(cur: str, x) -> str:
    return f"{cur.lower()}_{str(q2(x)).replace('.', '_').replace('-', 'minus_')}"


def next_business_day(d: date, holidays: set[date]) -> date:
    while d.weekday() >= 5 or d in holidays:
        d += timedelta(days=1)
    return d


def prev_business_day(d: date, holidays: set[date]) -> date:
    while d.weekday() >= 5 or d in holidays:
        d -= timedelta(days=1)
    return d


def weekday_date(rng: random.Random, y0: int = 2025, y1: int = 2027) -> date:
    d = rand_date(rng, y0, y1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def pick_labels(rng: random.Random, correct: str, wrongs: list[str], k: int = 5) -> list[str]:
    """Correct label plus distinct wrong labels, shuffled; at most k labels."""
    seen, out = {correct}, [correct]
    for w in wrongs:
        if w not in seen:
            seen.add(w)
            out.append(w)
    out = out[:k]
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------- scenario 1: contract term vs absolute cap


def term_vs_cap(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    """N-month term from delivery, extended by repair days (both ends inclusive), against a cap of C months from the
    manufacture date that no extension can move. Claim date from a portal timestamp in the desk's time zone."""
    f = facts or {}
    tr = Trace()
    org = names.org(rng.choice(["Audio", "Electronics", "Instruments", "Medical", "Components"]))
    product = rng.choice(["studio monitor", "ultrasound probe", "network switch", "espresso machine", "e-bike drive unit",
                          "lab centrifuge", "projector", "power station", "robot vacuum", "3D printer"])
    case_id = names.ident(rng.choice(["WD", "WC", "SR"]), 6)
    customer = f"{names.org(rng.choice(['Studios', 'Clinic', 'Partners', 'Labs']))}"
    made = f.get("made", rand_date(rng, 2023, 2026))
    made = date.fromisocalendar(made.isocalendar()[0], made.isocalendar()[1], 1)  # Monday of its ISO week
    n_term = f.get("n_term", rng.choice([12, 18, 24, 24, 36]))
    cap_months = f.get("cap_months", rng.choice([27, 30, 30, 36, 42]))
    delivery = f.get("delivery", made + timedelta(days=rng.randint(60, 400)))
    code_style = f.get("code_style", rng.random() < 0.55)
    iso_y, iso_w, _ = made.isocalendar()
    if code_style:
        tr.given("code", f"Batch code {iso_y % 100:02d}{iso_w:02d}: ISO year {iso_y}, week {iso_w}.")
        tr.derive("made", ["code"], f"Manufacture date taken as the Monday of that week, {fmt_date(made)}.")
    else:
        tr.given("made", f"Manufacture date {fmt_date(made)}.")
    tr.given("delivery", f"Delivery {fmt_date(delivery)}, term {n_term} months.")
    term_plain, _ = add_months_same_day(delivery, n_term)
    tr.derive("term_plain", ["delivery"], f"{n_term} months from delivery: {fmt_date(term_plain)}.")
    cap_end_pre, _ = add_months_same_day(made, cap_months)
    latest_repair = min(term_plain, cap_end_pre) - timedelta(days=50)
    has_repair = f.get("has_repair", (latest_repair - delivery).days > 40 and rng.random() < 0.65)
    ext_days = 0
    r_in = r_out = None
    if has_repair:
        r_in = f.get("r_in", delivery + timedelta(days=rng.randint(30, max(31, (latest_repair - delivery).days))))
        r_out = f.get("r_out", r_in + timedelta(days=rng.randint(4, 45)))
        ext_days = (r_out - r_in).days + 1
        tr.given("repair", f"Repair: received {fmt_date(r_in)}, dispatched back {fmt_date(r_out)}.")
        tr.derive("ext", ["repair"], f"Extension, both days inclusive: {ext_days} days.")
        term_end = term_plain + timedelta(days=ext_days)
        tr.derive("term_end", ["term_plain", "ext"], f"Term ends {fmt_date(term_end)}.")
    else:
        term_end = term_plain
        tr.derive("term_end", ["term_plain"], f"No repair extension; term ends {fmt_date(term_end)}.")
    cap_end, _ = add_months_same_day(made, cap_months)
    tr.derive("cap_end", ["made"], f"Cap: {cap_months} months from manufacture, {fmt_date(cap_end)}, not extended.")
    cov_end = min(term_end, cap_end)
    binding = "term" if term_end <= cap_end else "cap"
    tr.derive("cov_end", ["term_end", "cap_end"], f"Coverage ends on the earlier date, {fmt_date(cov_end)} (the {binding}).")

    # claim date: aim at a class, near a boundary
    target = f.get("target", rng.choices(["covered", "expired_term", "expired_cap"], weights=[0.4, 0.3, 0.3])[0])
    if target == "expired_cap" and binding != "cap":
        target = rng.choice(["covered", "expired_term"])
    if target == "covered":
        claim = cov_end - timedelta(days=rng.randint(0, 25))
    elif target == "expired_term":
        claim = term_end + timedelta(days=rng.randint(1, 25))
    else:
        claim = cap_end + timedelta(days=rng.randint(1, min(25, max(1, (term_end - cap_end).days))))
    claim = f.get("claim", claim)
    if has_repair and claim <= r_out and f.get("_retry", 0) < 8:  # the claim must come after the repair
        return term_vs_cap(rng, names, {**f, "_retry": f.get("_retry", 0) + 1})
    # portal timestamp: sometimes the customer's local date differs from the desk's date
    desk_city, desk_tz = f.get("desk_city", names.city())
    cust_city, cust_tz = f.get("cust_city", names.city())
    while cust_tz == desk_tz:
        cust_city, cust_tz = names.city()
    shift_date = f.get("shift_date", rng.random() < 0.4)
    desk_dt = datetime.combine(claim, time(rng.randint(0, 2) if shift_date else rng.randint(9, 20), rng.choice([5, 15, 30, 45])), tzinfo=ZoneInfo(desk_tz))
    cust_dt = desk_dt.astimezone(ZoneInfo(cust_tz))
    if shift_date and cust_dt.date() == claim:  # push the customer's clock across midnight the other way
        desk_dt = datetime.combine(claim, time(rng.randint(21, 23), rng.choice([5, 15, 30, 45])), tzinfo=ZoneInfo(desk_tz))
        cust_dt = desk_dt.astimezone(ZoneInfo(cust_tz))
    tr.given("portal", f"Portal submission {fmt_date(cust_dt.date())} {fmt_time(cust_dt.time())} {cust_city} time.")
    if cust_dt.date() != claim:
        tr.derive("claim_date", ["portal"], f"In {desk_city} time that is {fmt_date(claim)} {fmt_time(desk_dt.time())}, the claim date.")
    else:
        tr.derive("claim_date", ["portal"], f"Claim date in {desk_city} time: {fmt_date(claim)}.")
    if claim > term_end:
        verdict = "expired_term"
    elif claim > cap_end:
        verdict = "expired_cap"
    else:
        verdict = "covered"
    tr.derive("verdict", ["claim_date", "cov_end"], f"Claim {fmt_date(claim)} vs term end {fmt_date(term_end)} and cap {fmt_date(cap_end)}: {verdict.replace('_', ' ')}.")

    # desk notes that read the rules wrongly
    def classify(t_end: date, c_end: date, d: date) -> str:
        return "expired_term" if d > t_end else ("expired_cap" if d > c_end else "covered")

    restart_end = add_months_same_day(r_out, n_term)[0] if has_repair else term_end
    grace_end = term_end + timedelta(days=30)
    thirty_end = delivery + timedelta(days=30 * n_term + ext_days)
    cap_early = add_months_same_day(made, cap_months - 1)[0]
    wrongs = {
        "thirty_day_months": classify(thirty_end, cap_end, claim),
        "cap_month_start": classify(term_end, cap_early, claim),
        "ignores_cap": classify(term_end, date.max, cust_dt.date()),
        "ignores_extension": classify(term_plain, cap_end, claim),
        "cap_from_delivery": classify(term_end, add_months_same_day(delivery, cap_months)[0], claim),
        "customer_date": classify(term_end, cap_end, cust_dt.date()),
        "repair_restarts_term": classify(restart_end, cap_end, claim),
        "assumed_grace": classify(grace_end, cap_end, claim),
    }
    if not has_repair:
        wrongs.pop("ignores_extension")
        wrongs.pop("repair_restarts_term")
    if cust_dt.date() == claim:
        wrongs.pop("customer_date")
    kinds = [k for k, v in wrongs.items() if v != verdict]
    if not kinds and f.get("_retry", 0) < 8:
        return term_vs_cap(rng, names, {**f, "_retry": f.get("_retry", 0) + 1})
    kind = f.get("kind", rng.choice(kinds) if kinds else rng.choice(list(wrongs)))
    surface = wrongs[kind]
    texts = {
        "ignores_cap": f"Warranty desk note ({names.person()}): term runs to {fmt_date(term_end)}, claim logged {fmt_date(cust_dt.date())}, so {surface.replace('_', ' ')}. The manufacture limit is for the factory's records only.",
        "ignores_extension": f"Warranty desk note ({names.person()}): {n_term} months from {fmt_date(delivery)} is {fmt_date(term_plain)}; the repair does not change the term. Claim of {fmt_date(claim)}: {surface.replace('_', ' ')}.",
        "cap_from_delivery": f"Warranty desk note ({names.person()}): the {cap_months}-month limit counts from delivery, giving {fmt_date(add_months_same_day(delivery, cap_months)[0])}; the claim is therefore {surface.replace('_', ' ')}.",
        "customer_date": f"Warranty desk note ({names.person()}): the customer's form shows {fmt_date(cust_dt.date())}, so the claim date is {fmt_date(cust_dt.date())} and the case is {surface.replace('_', ' ')}.",
        "thirty_day_months": f"Warranty desk note ({names.person()}): {n_term} months is {30 * n_term} days{' plus ' + str(ext_days) + ' repair days' if ext_days else ''}, so the term runs to {fmt_date(thirty_end)}; the claim of {fmt_date(claim)} is {surface.replace('_', ' ')}.",
        "cap_month_start": f"Warranty desk note ({names.person()}): the {cap_months}th month after manufacture begins on {fmt_date(cap_early)}, and claims from then on are outside the limit; the claim is {surface.replace('_', ' ')}.",
        "repair_restarts_term": f"Warranty desk note ({names.person()}): the unit came back from repair on {fmt_date(r_out) if r_out else ''}, which restarts the {n_term}-month term; it now runs to {fmt_date(restart_end)}, so the claim is {surface.replace('_', ' ')}.",
        "assumed_grace": f"Warranty desk note ({names.person()}): desk practice allows a 30-day grace period after the term ends, i.e. until {fmt_date(grace_end)}; the claim of {fmt_date(claim)} is therefore {surface.replace('_', ' ')}.",
    }
    note = texts[kind]
    made_line = (f"Serial {names.ident('SN', 5)}, batch code {iso_y % 100:02d}{iso_w:02d} (the first two digits are the ISO year, the last two the ISO week of manufacture; "
                 f"the manufacture date is taken as the Monday that starts that week)." if code_style else f"Serial {names.ident('SN', 5)}, manufactured {fmt_date(made, rng.randint(0, 2))}.")
    repair_line = (f"Repair {names.ident('RMA', 5)}: accepted as a covered repair; unit arrived at the repair centre {fmt_date(r_in, rng.randint(0, 2))}, "
                   f"dispatched back to the customer {fmt_date(r_out, rng.randint(0, 2))}." if has_repair else "Repairs: none.")
    rule_terms = rng.choice([
        f"Clause 1. Defects are covered for {n_term} months from delivery to the first end customer. Clause 2. That period grows by the days the unit was held "
        f"by an approved repair centre for a covered repair, from the day it arrived there to the day it was sent back, both counted.",
        f"Section A. Coverage: {n_term} months counted from the delivery date. Section B. Days the unit was held by an approved repair centre for a covered repair (arrival day "
        f"through dispatch day, both counted) are added to the coverage period.",
    ])
    rule_cap = rng.choice([
        f"Clause 6. Whatever clauses 1 and 2 say, a claim made after the {cap_months}th month from the manufacture date is refused; repairs do not move this limit.",
        f"Section F. Absolute limit: nothing in this statement extends coverage beyond {cap_months} months after the manufacture date, repairs or not.",
    ])
    state = "\n".join([
        f"{org.upper()} — WARRANTY DESK ({desk_city}), CASE {case_id}",
        "",
        *fillers(rng, names),
        f"Product: {org.split()[0]} {product}. {made_line}",
        f"Customer: {customer}, {cust_city}.",
        "",
        "Warranty statement (extract)",
        rule_terms,
        rule_cap,
        f"Clause 8. The claim date is the day the customer first logs the fault on the service portal, read in {desk_city} time (the portal keeps {desk_city} time for every region). "
        f"A period stated in months runs to the same day number in its last month; when that month is shorter, it runs to that month's last day.",
        "",
        f"Case {case_id}: delivered to the customer {fmt_date(delivery, rng.randint(0, 2))} (delivery confirmed by the carrier). {repair_line} "
        f"Fault reported through the portal at {fmt_time(cust_dt.time())} on {fmt_date(cust_dt.date(), rng.randint(0, 2))}, {cust_city} time "
        f"({utc_offset_str(cust_dt)}; {desk_city} is {utc_offset_str(desk_dt)} on that date). Reported fault: {rng.choice(['no power', 'intermittent output', 'error E', 'overheating', 'dead pixel cluster'])}.",
        "",
        note,
    ])
    key = f"{made}|{n_term}|{cap_months}|{delivery}|{r_in}|{r_out}|{claim}|{desk_tz}|{cust_dt.isoformat()}"
    extra = {"program_key": key, "version": "v2"}
    variant = f.get("variant", rng.random())
    if variant < 0.5:
        return SynthItem(
            family=FAMILY, scenario="term_vs_cap", state=state, qtype="choice",
            instructions=f"Which classification applies to warranty claim {case_id}?",
            criteria={"covered": "Warranty applies: the claim date is inside the term and inside the manufacture limit.",
                      "expired_term": f"Refused because the {n_term}-month term (with any repair extension) had already ended on the claim date.",
                      "expired_cap": f"Refused because the claim date is past the {cap_months}-month manufacture limit, although the term itself was still running."},
            expected=verdict, trace=tr, signature=key, distractor=kind, surface_answer=surface, extra=extra)
    if variant < 0.75:
        return SynthItem(
            family=FAMILY, scenario="term_vs_cap", state=state, qtype="noul",
            instructions=f"Does warranty apply to claim {case_id}?",
            criteria={"true": "Yes: the claim date is inside both the (extended) term and the manufacture limit.",
                      "false": "No: the claim date is past the term or past the manufacture limit."},
            expected="yes" if verdict == "covered" else "no", trace=tr, signature=key, distractor=kind,
            surface_answer="yes" if surface == "covered" else "no", extra=extra)
    wrong_dates = [term_plain, cap_end if binding == "term" else term_end, add_months_same_day(delivery, cap_months)[0],
                   term_plain + timedelta(days=max(0, ext_days - 1)), cov_end + timedelta(days=1), min(thirty_end, cap_end), min(term_end, cap_early)]
    labels = pick_labels(rng, snake_date(cov_end), [snake_date(d) for d in wrong_dates])
    surf_date = {"ignores_cap": term_end, "ignores_extension": min(term_plain, cap_end), "cap_from_delivery": min(term_end, add_months_same_day(delivery, cap_months)[0]),
                 "customer_date": cov_end, "repair_restarts_term": min(restart_end, cap_end), "assumed_grace": min(grace_end, cap_end),
                 "thirty_day_months": min(thirty_end, cap_end), "cap_month_start": min(term_end, cap_early)}[kind]
    return SynthItem(
        family=FAMILY, scenario="term_vs_cap", state=state, qtype="choice",
        instructions=f"On which date does warranty coverage for case {case_id} end?",
        criteria={l: f"Coverage ends on {l.replace('_', '-')}." for l in labels},
        expected=snake_date(cov_end), trace=tr, signature=key, distractor=kind,
        surface_answer=snake_date(surf_date) if snake_date(surf_date) in labels and surf_date != cov_end else "", extra=extra)


# ----------------------------------------------------------------------------- scenario 2: foreign-currency lines, table rate, per-line cap, threshold

CURRENCIES = [("JPY", Decimal("162"), Decimal("0"), 0), ("GBP", Decimal("0.855"), Decimal("0.001"), 2), ("USD", Decimal("1.09"), Decimal("0.001"), 2),
              ("CHF", Decimal("0.955"), Decimal("0.001"), 2), ("SEK", Decimal("11.4"), Decimal("0.01"), 2), ("PLN", Decimal("4.31"), Decimal("0.001"), 2),
              ("CAD", Decimal("1.48"), Decimal("0.001"), 2), ("AUD", Decimal("1.64"), Decimal("0.001"), 2), ("INR", Decimal("90.2"), Decimal("0.01"), 0),
              ("NOK", Decimal("11.7"), Decimal("0.01"), 2), ("CZK", Decimal("25.1"), Decimal("0.01"), 0), ("MXN", Decimal("19.8"), Decimal("0.01"), 2)]


def fx_lines_cap(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    """Hotel folio in a foreign currency: reference rate of the transaction date (else the most recent published
    rate), each night capped separately, local tax reimbursed in full, some lines excluded; then a threshold."""
    f = facts or {}
    tr = Trace()
    home = f.get("home", rng.choice(["EUR", "EUR", "EUR", "USD", "GBP"]))
    cur, base_rate, tick, fx_dec = f.get("cur", rng.choice([c for c in CURRENCIES if c[0] != home]))
    fxq = Decimal("1") if fx_dec == 0 else CENT
    org = names.org(rng.choice(["Robotics", "Systems", "Engineering", "Technologies", "Medical"]))
    employee = names.person()
    claim_id = names.ident(rng.choice(["TE", "TX", "XP"]), 6)
    city, _ = names.city()
    check_in = f.get("check_in", weekday_date(rng))
    nights = f.get("nights", rng.choice([2, 3, 3, 4]))
    check_out = check_in + timedelta(days=nights)
    # rate table: business days from check_in - 2 to check_out + 2, random walk
    days = []
    d = check_in - timedelta(days=3)
    while d <= check_out + timedelta(days=3):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rates: dict[date, Decimal] = {}
    r = base_rate * Decimal(str(rng.uniform(0.97, 1.03)))
    for d in days:
        r = r * Decimal(str(rng.uniform(0.985, 1.015)))
        rates[d] = r.quantize(tick if tick else Decimal("0.01")) if tick else r.quantize(Decimal("0.01"))
    rates = f.get("rates", rates)
    tx_rule = f.get("tx_rule", rng.choice(["checkout", "checkout", "per_night"]))
    holidays = set()
    if "rates" not in f and rng.random() < 0.3:  # a table gap on a weekday
        gap = rng.choice([d for d in days if check_in <= d <= check_out] or days)
        rates.pop(gap, None)
        holidays.add(gap)

    def rate_for(d: date) -> tuple[date, Decimal]:
        while d not in rates:
            d -= timedelta(days=1)
        return d, rates[d]

    cap = f.get("cap", Decimal(rng.choice([120, 140, 150, 160, 170, 180, 200])))
    room_lines = []
    for i in range(nights):
        night = check_in + timedelta(days=i)
        room_home = cap * Decimal(str(rng.uniform(0.8, 1.25)))
        room_fx = (room_home * rate_for(check_out if tx_rule == "checkout" else night)[1]).quantize(fxq)
        room_lines.append((night, room_fx))
    room_lines = f.get("room_lines", room_lines)
    tax_fx = f.get("tax_fx", (cap * Decimal(str(rng.uniform(0.02, 0.06))) * base_rate).quantize(fxq))
    excluded = f.get("excluded", [(rng.choice(["Minibar", "Laundry", "Breakfast (restaurant)", "Spa access"]), (cap * Decimal(str(rng.uniform(0.05, 0.2))) * base_rate).quantize(fxq))
                                  for _ in range(rng.randint(1, 2))])
    tr.given("rule", f"Rate of the transaction date, else the most recent published rate; cap {amt(home, cap)} per night; tax in full; extras excluded.")
    total = Decimal(0)
    total_nocap = Decimal(0)
    total_capsum = Decimal(0)
    conv_lines = []
    tx_dates = set()
    for i, (night, room_fx) in enumerate(room_lines):
        tx = check_out if tx_rule == "checkout" else night
        rd, rate = rate_for(tx)
        tx_dates.add((tx, rd))
        room_home = q2(room_fx / rate)
        reimb = min(room_home, cap)
        total += reimb
        total_nocap += room_home
        total_capsum += room_home
        conv_lines.append((night, room_fx, rate, room_home, reimb))
        tr.given(f"room{i}", f"Night {fmt_date(night)}: {cur} {room_fx:,}.")
        tr.derive(f"conv{i}", [f"room{i}", "rule"], f"At {rate} ({fmt_date(rd)} rate): {amt(home, room_home)}; capped to {amt(home, reimb)}.")
    tax_total = Decimal(0)
    for i, (night, _) in enumerate(room_lines):
        tx = check_out if tx_rule == "checkout" else night
        rd, rate = rate_for(tx)
        tax_total += q2(tax_fx / rate)
    tr.given("tax", f"City tax {cur} {tax_fx:,} per night x {nights}.")
    tr.derive("tax_home", ["tax", "rule"], f"Tax converted line by line: {amt(home, tax_total)}, reimbursed in full.")
    total += tax_total
    tr.derive("total", [f"conv{i}" for i in range(nights)] + ["tax_home"], f"Lodging refund: {amt(home, total)}.")
    # wrong readings
    post_date = next_business_day(check_out + timedelta(days=1), set())
    post_rd, post_rate = rate_for(post_date) if post_date in rates else (max(rates), rates[max(rates)])
    w_post = sum(min(q2(x / post_rate), cap) for _, x in room_lines) + sum(q2(tax_fx / post_rate) for _ in room_lines)
    w_nocap = total_nocap + tax_total
    w_capsum = min(total_capsum, cap * nights) + tax_total
    w_excl = total + sum(q2(x / rate_for(check_out)[1]) for _, x in excluded)
    ci_rd, ci_rate = rate_for(check_in)
    w_checkin = sum(min(q2(x / ci_rate), cap) for _, x in room_lines) + sum(q2(tax_fx / ci_rate) for _ in room_lines)
    w_notax = total - tax_total
    wrongs = {"posting_date_rate": w_post, "no_cap": w_nocap, "cap_on_total": w_capsum, "extras_included": w_excl, "checkin_rate": w_checkin, "tax_in_cap": w_notax}
    wrongs = {k: q2(v) for k, v in wrongs.items() if q2(v) != q2(total)}
    if len(wrongs) < 3 and f.get("_retry", 0) < 8:
        return fx_lines_cap(rng, names, {**f, "_retry": f.get("_retry", 0) + 1})
    tgt = rng.choice(["auto", "manager", "director"])  # thresholds placed so each approval level is equally likely
    if tgt == "auto":
        lo = total * Decimal(str(rng.uniform(1.03, 1.18))); hi = lo * Decimal(str(rng.uniform(1.25, 1.6)))
    elif tgt == "manager":
        lo = total * Decimal(str(rng.uniform(0.8, 0.97))); hi = total * Decimal(str(rng.uniform(1.03, 1.25)))
    else:
        hi = total * Decimal(str(rng.uniform(0.82, 0.97))); lo = hi * Decimal(str(rng.uniform(0.6, 0.85)))
    thr_lo = f.get("thr_lo", lo.quantize(Decimal("10"), rounding=ROUND_HALF_UP) if lo >= 100 else lo.quantize(Decimal("1")))
    thr_hi = f.get("thr_hi", hi.quantize(Decimal("10"), rounding=ROUND_HALF_UP) if hi >= 100 else hi.quantize(Decimal("1")))
    if thr_hi <= thr_lo:
        thr_hi = thr_lo + Decimal(10)
    variant = f.get("variant", rng.random())
    thr = f.get("thr", (total * Decimal(str(rng.uniform(0.9, 1.1)))).quantize(Decimal("10"), rounding=ROUND_HALF_UP))

    def level_of(x: Decimal) -> str:
        return "auto" if x <= thr_lo else ("manager" if x <= thr_hi else "director")

    def disagrees(v: Decimal) -> bool:  # the wrong amount must change the answer of the question actually asked
        if variant < 0.55:
            return True
        if variant < 0.8:
            return level_of(v) != level_of(total)
        return (v > thr) != (total > thr)

    kinds = [k for k, v in wrongs.items() if disagrees(v)]
    if not kinds and f.get("_retry", 0) < 8:
        return fx_lines_cap(rng, names, {**f, "_retry": f.get("_retry", 0) + 1, "variant": variant})
    kind = f.get("kind", rng.choice(kinds) if kinds else (rng.choice(list(wrongs)) if wrongs else "no_cap"))
    wrong = wrongs.get(kind, q2(total_nocap + tax_total))
    who = names.person()
    texts = {
        "posting_date_rate": f"Audit desk note ({who}): the card posted the charge on {fmt_date(post_date)}, so I converted everything at that day's rate ({post_rate}) and applied the caps: {amt(home, wrong)}.",
        "no_cap": f"Audit desk note ({who}): converted each room night and the tax at the correct rate; the ceiling is a booking guideline, so the refund is {amt(home, wrong)}.",
        "cap_on_total": f"Audit desk note ({who}): the room nights total {amt(home, total_capsum)} against a combined ceiling of {amt(home, cap * nights)} for {nights} nights, so the lodging refund is {amt(home, wrong)} including tax.",
        "extras_included": f"Audit desk note ({who}): all folio lines converted at the correct rate and the nightly cap applied to the room lines: {amt(home, wrong)}.",
        "tax_in_cap": f"Audit desk note ({who}): city tax is part of the cost of the night and falls under the nightly ceiling, so the refund is the capped room charges alone: {amt(home, wrong)}.",
        "checkin_rate": f"Audit desk note ({who}): used the rate of the arrival day {fmt_date(check_in)} ({ci_rate}) for the whole folio, caps applied: {amt(home, wrong)}.",
    }
    note = texts[kind]
    table = "\n".join(f"{d.isoformat()} ({d.strftime('%a').lower()})  {rates[d]}" for d in sorted(rates))
    tx_text = ("The transaction date is the date the merchant charged the expense: for a hotel folio closed and charged at check-out, that is the check-out date for every line."
               if tx_rule == "checkout" else "For hotel folios each night is a separate transaction dated on that night.")
    folio = [f"- Room, night of {fmt_date(n, 1)}: {cur} {x:,}" for n, x in room_lines]
    folio += [f"- City tax, night of {fmt_date(n, 1)}: {cur} {tax_fx:,}" for n, _ in room_lines]
    folio += [f"- {name}: {cur} {x:,}" for name, x in excluded]
    rng.shuffle(folio)
    state = "\n".join([
        f"{org.upper()} — TRAVEL EXPENSE AUDIT, CLAIM {claim_id}",
        "",
        *fillers(rng, names),
        f"Employee {employee}, home currency {home}. Trip: {rng.choice(['client workshop', 'site survey', 'trade fair', 'training course'])}, {city}, "
        f"{fmt_date(check_in)} to {fmt_date(check_out)} ({nights} nights). Hotel folio {names.ident('F', 5)} closed at check-out on {fmt_date(check_out)}; card charged the same day and posted on {fmt_date(post_date)}.",
        "",
        "Travel policy (extract)",
        f"7.2 Amounts paid in another currency are translated into {home} with the treasury reference rate of the transaction date; when the table has no entry for that date, the latest earlier entry applies. Card statement conversions are never used.",
        f"7.3 {tx_text}",
        "7.4 Each folio line is converted on its own and rounded to the cent after conversion.",
        f"8.1 Room charges are refunded up to a per-night ceiling of {amt(home, cap)}. Each night is compared with the ceiling on its own; a saving on one night cannot be carried to another night.",
        "8.2 City or accommodation taxes are paid back in full on top of the capped room charges.",
        "8.3 Minibar, laundry, restaurant breakfast and spa charges do not belong to the lodging line and are not paid back under it.",
        f"9.1 A lodging refund up to {amt(home, thr_lo)} is approved automatically; above that and up to {amt(home, thr_hi)} it needs the line manager; above {amt(home, thr_hi)} the divisional director.",
        "",
        f"Treasury table ({cur} per 1 {home}; entries exist for business days only):",
        table,
        "",
        f"Hotel folio lines (all amounts {cur}):",
        *folio,
        "",
        note,
    ])
    key = f"{home}|{cur}|{sorted((d.isoformat(), str(r)) for d, r in rates.items())}|{tx_rule}|{[(n.isoformat(), str(x)) for n, x in room_lines]}|{tax_fx}|{cap}"
    extra = {"program_key": key, "version": "v2"}
    level, w_level = level_of(total), level_of(wrong)
    if variant < 0.55:
        labels = pick_labels(rng, amt_label(home, total), [amt_label(home, v) for v in wrongs.values()])
        by_label = {amt_label(home, total): total, **{amt_label(home, v): v for v in wrongs.values()}}
        return SynthItem(
            family=FAMILY, scenario="fx_lines_cap", state=state, qtype="choice",
            instructions=f"How much of the lodging on claim {claim_id} is refundable under the policy?",
            criteria={l: f"Lodging refund of {amt(home, by_label[l])}." for l in labels},
            expected=amt_label(home, total), trace=tr, signature=key, distractor=kind,
            surface_answer=amt_label(home, wrong) if amt_label(home, wrong) in labels else "", extra=extra)
    tr.derive("level", ["total"], f"Against the thresholds {amt(home, thr_lo)} / {amt(home, thr_hi)}: {level}.")
    if variant < 0.8:
        return SynthItem(
            family=FAMILY, scenario="fx_lines_cap", state=state, qtype="choice",
            instructions=f"Which approval level does the lodging refund on claim {claim_id} need under 9.1?",
            criteria={"auto": f"Automatic approval (refund up to {amt(home, thr_lo)}).",
                      "manager": f"Line manager (refund above {amt(home, thr_lo)}, up to {amt(home, thr_hi)}).",
                      "director": f"Divisional director (refund above {amt(home, thr_hi)})."},
            expected=level, trace=tr, signature=key, distractor=kind, surface_answer=w_level if w_level != level else "", extra=extra)
    return SynthItem(
        family=FAMILY, scenario="fx_lines_cap", state=state, qtype="noul",
        instructions=f"Is the lodging refund on claim {claim_id} more than {amt(home, thr)}?",
        criteria={"true": f"The refund comes to more than {amt(home, thr)}.", "false": f"The refund is {amt(home, thr)} or less."},
        expected="yes" if total > thr else "no", trace=tr, signature=key, distractor=kind,
        surface_answer=("yes" if wrong > thr else "no") if (wrong > thr) != (total > thr) else "", extra=extra)


# ----------------------------------------------------------------------------- scenario 3: effective date + expiry conditions


def effective_expiry(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    """A licence in force from its effective date until the EARLIEST of: the fixed term, a transfer event plus a grace
    period, a termination notice plus a notice period counted from receipt, or a fee lapse on the due date."""
    f = facts or {}
    tr = Trace()
    kind_doc, holder_kind = rng.choice([("operating licence", "operator"), ("site permit", "permit holder"), ("vendor accreditation", "vendor"),
                                        ("software subscription", "subscriber"), ("fleet policy", "policyholder")])
    org = names.org(rng.choice(["Authority", "Holdings", "Systems", "Assurance", "Services"]))
    holder = names.org(rng.choice(["Logistics", "Foods", "Components", "Textiles", "Energy"]))
    ref = names.ident(rng.choice(["LIC", "PRM", "ACC", "SUB", "POL"]), 6)
    eff = f.get("eff", rand_date(rng, 2025, 2027))
    k_months = f.get("k_months", rng.choice([12, 18, 24, 24, 36]))
    fixed_end, _ = add_months_same_day(eff, k_months)
    tr.given("eff", f"Effective {fmt_date(eff)}, fixed term {k_months} months.")
    tr.derive("fixed_end", ["eff"], f"Fixed term ends {fmt_date(fixed_end)}.")
    ends: dict[str, date] = {"fixed_term": fixed_end}
    span = (fixed_end - eff).days
    has_transfer = f.get("has_transfer", rng.random() < 0.55)
    grace = f.get("grace", rng.choice([14, 30, 30, 60, 90]))
    transfer = None
    if has_transfer:
        transfer = f.get("transfer", eff + timedelta(days=rng.randint(30, max(31, span + 60))))
        ends["transfer_event"] = transfer + timedelta(days=grace)
        tr.given("transfer", f"Transfer of the business on {fmt_date(transfer)}.")
        tr.derive("transfer_end", ["transfer"], f"Ends {grace} days after the transfer: {fmt_date(ends['transfer_event'])}.")
    has_notice = f.get("has_notice", rng.random() < 0.5)
    notice_period = f.get("notice_period", rng.choice([30, 60, 90]))
    notice_date = received = None
    if has_notice:
        notice_date = f.get("notice_date", eff + timedelta(days=rng.randint(30, max(31, span + 60))))
        received = f.get("received", notice_date + timedelta(days=rng.choice([0, 1, 2, 3, 4])))
        ends["termination_notice"] = received + timedelta(days=notice_period)
        tr.given("notice", f"Termination notice dated {fmt_date(notice_date)}, received {fmt_date(received)}.")
        tr.derive("notice_end", ["notice"], f"Takes effect {notice_period} days after receipt: {fmt_date(ends['termination_notice'])}.")
    has_fee = f.get("has_fee", k_months > 12 and rng.random() < 0.5)
    fee_due = paid = None
    if has_fee:
        fee_due, _ = add_months_same_day(eff, 12)
        paid = f.get("paid", None if rng.random() < 0.3 else fee_due + timedelta(days=rng.randint(-20, 25)))
        if paid is None or paid > fee_due:
            ends["fee_lapse"] = fee_due
            tr.given("fee", f"Annual fee due {fmt_date(fee_due)}, {'not paid' if paid is None else 'paid ' + fmt_date(paid)}.")
            tr.derive("fee_end", ["fee"], f"Not paid by the due date: lapses at the end of {fmt_date(fee_due)}.")
        else:
            tr.given("fee", f"Annual fee due {fmt_date(fee_due)}, paid {fmt_date(paid)} (on time).")
    if len(ends) == 1 and f.get("_retry", 0) < 8:  # want at least one competing condition
        return effective_expiry(rng, names, {**f, "_retry": f.get("_retry", 0) + 1})
    which = min(ends, key=lambda k: (ends[k], k))
    expiry = ends[which]
    tr.derive("expiry", [n for n in ("fixed_end", "transfer_end", "notice_end", "fee_end") if n in tr._names],
              f"Earliest of {', '.join(f'{k.replace(chr(95), ' ')} {fmt_date(v)}' for k, v in ends.items())}: {fmt_date(expiry)} ({which.replace('_', ' ')}).")
    query = f.get("query", expiry + timedelta(days=rng.choice([-1, 1]) * rng.randint(0, 20)))
    in_force = query <= expiry
    tr.derive("in_force", ["expiry"], f"On {fmt_date(query)} the {kind_doc} is {'in force' if in_force else 'no longer in force'}.")

    # wrong readings: (condition, date)
    wrongs: dict[str, tuple[str, date]] = {"fixed_only": ("fixed_term", fixed_end)}
    if has_transfer:
        alt = {k: v for k, v in ends.items()}
        alt["transfer_event"] = transfer
        w = min(alt, key=lambda k: (alt[k], k))
        wrongs["no_grace"] = (w, alt[w])
    if has_notice and received != notice_date:
        alt = {k: v for k, v in ends.items()}
        alt["termination_notice"] = notice_date + timedelta(days=notice_period)
        w = min(alt, key=lambda k: (alt[k], k))
        wrongs["notice_from_date"] = (w, alt[w])
    if "fee_lapse" in ends:
        alt = {k: v for k, v in ends.items() if k != "fee_lapse"}
        w = min(alt, key=lambda k: (alt[k], k))
        wrongs["ignores_lapse"] = (w, alt[w])
    if len(ends) >= 2:
        w = max(ends, key=lambda k: (ends[k], k))
        wrongs["latest_event"] = (w, ends[w])
    variant = f.get("variant", rng.random())
    qv = "which" if variant < 0.4 else ("date" if variant < 0.7 else "noul")

    def surface_of(k: str):
        cond, d = wrongs[k]
        if qv == "which":
            return cond
        if qv == "date":
            return snake_date(d)
        return "yes" if query <= d else "no"

    truth = which if qv == "which" else (snake_date(expiry) if qv == "date" else ("yes" if in_force else "no"))
    kinds = [k for k in wrongs if surface_of(k) != truth]
    if not kinds and f.get("_retry", 0) < 8:
        return effective_expiry(rng, names, {**f, "_retry": f.get("_retry", 0) + 1, "variant": variant})
    kind = f.get("kind", rng.choice(kinds) if kinds else rng.choice(list(wrongs)))
    w_cond, w_date = wrongs[kind]
    who = names.person()
    texts = {
        "fixed_only": f"Registry note ({who}): the {kind_doc} runs for its fixed term to {fmt_date(fixed_end)}; the other correspondence on file does not change the expiry date.",
        "no_grace": f"Registry note ({who}): the {kind_doc} ended with the transfer on {fmt_date(transfer) if transfer else ''}; nothing survives a change of ownership. Expiry {fmt_date(w_date)} ({w_cond.replace('_', ' ')}).",
        "notice_from_date": f"Registry note ({who}): {notice_period} days from the notice date {fmt_date(notice_date) if notice_date else ''} gives {fmt_date(notice_date + timedelta(days=notice_period)) if notice_date else ''}, so expiry is {fmt_date(w_date)} ({w_cond.replace('_', ' ')}).",
        "ignores_lapse": f"Registry note ({who}): fee reminders are administrative; the {kind_doc} stays in force until {fmt_date(w_date)} ({w_cond.replace('_', ' ')}).",
        "latest_event": f"Registry note ({who}): several conditions apply, so the {kind_doc} continues until the last of them, {fmt_date(w_date)} ({w_cond.replace('_', ' ')}).",
    }
    note = texts[kind]
    events = []
    if has_transfer:
        events.append(f"- Change of control: the {holder_kind}'s business was transferred to a new owner on {fmt_date(transfer, rng.randint(0, 2))} (transfer deed on file).")
    if has_notice:
        events.append(f"- Written termination notice from the {holder_kind} dated {fmt_date(notice_date, rng.randint(0, 2))}, received by the registry on {fmt_date(received, rng.randint(0, 2))} (date stamp).")
    if has_fee:
        events.append(f"- Annual fee of {rng.choice(['EUR', 'USD', 'GBP'])} {rng.choice([450, 600, 750, 900, 1200]):,} due {fmt_date(fee_due, rng.randint(0, 2))}: "
                      + ("no payment received." if paid is None else f"payment received {fmt_date(paid, rng.randint(0, 2))}."))
    rng.shuffle(events)
    conds = [f"(a) at the end of the day {k_months} months after the effective date"]
    conds.append(f"(b) {grace} days after any transfer of the {holder_kind}'s business to another person")
    conds.append(f"(c) {notice_period} days after the registry receives a written termination notice from the {holder_kind}; the notice date itself is not relevant")
    if k_months > 12:
        conds.append("(d) at the end of the annual fee due date, if the fee has not been received by then")
    state = "\n".join([
        f"{org.upper()} — {kind_doc.upper()} REGISTER, {ref}",
        "",
        *fillers(rng, names),
        f"{kind_doc.capitalize()} {ref} issued to {holder}; effective date {fmt_date(eff, rng.randint(0, 2))}.",
        "",
        "Conditions (extract)",
        f"Term. The {kind_doc} is in force from the effective date and ends at the earliest of the following: " + "; ".join(conds) + ".",
        "Once ended, it is not revived by later payments or correspondence.",
        "",
        "Events on file:",
        *(events or ["- none"]),
        "",
        note,
    ])
    key = f"{eff}|{k_months}|{transfer}|{grace}|{notice_date}|{received}|{notice_period}|{fee_due}|{paid}|{query}"
    extra = {"program_key": key, "version": "v2"}
    surface = surface_of(kind)
    if qv == "which":
        labels = {"fixed_term": f"The fixed term of {k_months} months ran out.", "transfer_event": "The transfer of the business ended it (after the grace period).",
                  "termination_notice": "The termination notice took effect.", "fee_lapse": "It lapsed because the annual fee was not paid by the due date."}
        if k_months <= 12:
            labels.pop("fee_lapse")
        return SynthItem(family=FAMILY, scenario="effective_expiry", state=state, qtype="choice",
                         instructions=f"Which condition ended {kind_doc} {ref} (or will end it first)?",
                         criteria=labels, expected=which, trace=tr, signature=key, distractor=kind,
                         surface_answer=surface if surface != which and surface in labels else "", extra=extra)
    if qv == "date":
        cands = [d for _, d in wrongs.values()] + [expiry + timedelta(days=1), expiry - timedelta(days=1)]
        labels = pick_labels(rng, snake_date(expiry), [snake_date(d) for d in cands])
        return SynthItem(family=FAMILY, scenario="effective_expiry", state=state, qtype="choice",
                         instructions=f"On which date does {kind_doc} {ref} end?",
                         criteria={l: f"The {kind_doc} ends on {l.replace('_', '-')}." for l in labels},
                         expected=snake_date(expiry), trace=tr, signature=key, distractor=kind,
                         surface_answer=surface if surface != snake_date(expiry) and surface in labels else "", extra=extra)
    return SynthItem(family=FAMILY, scenario="effective_expiry", state=state, qtype="noul",
                     instructions=f"Is {kind_doc} {ref} in force on {fmt_date(query)}?",
                     criteria={"true": f"The {kind_doc} is still in force on that date.", "false": f"The {kind_doc} has ended before that date."},
                     expected="yes" if in_force else "no", trace=tr, signature=key, distractor=kind,
                     surface_answer=surface if surface != ("yes" if in_force else "no") else "", extra=extra)


# ----------------------------------------------------------------------------- scenario 4: before / after a deadline


def deadline_boolean(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    """A filing deadline N days after an event with an explicit counting rule (event day counted or not), an optional
    roll to the next business day, a cutoff time in the office's zone, and a submission stamped in another zone."""
    f = facts or {}
    tr = Trace()
    org = names.org(rng.choice(["Assurance", "Authority", "Mutual", "Underwriters", "Indemnity"]))
    ref = names.ident(rng.choice(["FIL", "APL", "NOT", "CLM"]), 6)
    event_kind = rng.choice(["the decision was notified", "the loss occurred", "the invoice was issued", "the defect was discovered", "the goods were delivered"])
    filing_kind = rng.choice(["appeal", "claim form", "notice of dispute", "defect report", "reimbursement request"])
    event = f.get("event", rand_date(rng, 2025, 2027))
    n_days = f.get("n_days", rng.choice([10, 14, 15, 20, 21, 30, 30, 45, 60]))
    count_event_day = f.get("count_event_day", rng.random() < 0.4)
    roll = f.get("roll", rng.random() < 0.55)
    office_city, office_tz = f.get("office_city", names.city())
    filer_city, filer_tz = f.get("filer_city", names.city())
    same_tz = f.get("same_tz", rng.random() < 0.35)
    if same_tz:
        filer_city, filer_tz = office_city, office_tz
    else:
        while filer_tz == office_tz:
            filer_city, filer_tz = names.city()
    cutoff = f.get("cutoff", time(rng.choice([16, 17, 17, 18]), rng.choice([0, 0, 30])))
    grace = f.get("grace", rng.choice([3, 5, 7, 10]))
    holidays = f.get("holidays", set())
    tr.given("event", f"{event_kind[0].upper() + event_kind[1:]} on {fmt_date(event)}; {n_days} days to file.")
    raw_end = event + timedelta(days=n_days - (1 if count_event_day else 0))
    tr.derive("raw_end", ["event"], f"{'Counting the event day' if count_event_day else 'Not counting the event day'}, day {n_days} is {fmt_date(raw_end)}.")
    if not holidays and rng.random() < 0.5:
        hd = raw_end if raw_end.weekday() < 5 and rng.random() < 0.5 else raw_end + timedelta(days=rng.randint(1, 3))
        holidays = {hd}
    deadline = next_business_day(raw_end, holidays) if roll else raw_end
    if roll and deadline != raw_end:
        tr.derive("deadline", ["raw_end"], f"{fmt_date(raw_end)} is not a business day; the deadline moves to {fmt_date(deadline)}.")
    elif roll:
        tr.derive("deadline", ["raw_end"], f"{fmt_date(raw_end)} is a business day, so the deadline stays.")
    else:
        tr.derive("deadline", ["raw_end"], f"No extension for weekends or holidays: the deadline is {fmt_date(deadline)}.")
    office = ZoneInfo(office_tz)
    deadline_dt = datetime.combine(deadline, cutoff, tzinfo=office)
    tr.derive("cutoff", ["deadline"], f"Filings must reach the office by {fmt_time(cutoff)} {office_city} time on {fmt_date(deadline)}.")
    # submission near the deadline, in the filer's zone
    band = rng.random()
    if band < 0.45:
        draw = -int(abs(rng.gauss(0, 20 * 60)))  # on time, mostly within a day of the cutoff
    elif band < 0.75:
        draw = rng.randint(15, grace * 24 * 60)  # late, within the grace window
    else:
        draw = rng.randint(grace * 24 * 60 + 30, (grace + 8) * 24 * 60)  # too late
    delta_min = f.get("delta_min", draw)
    delta_min = max(-4 * 24 * 60, min((grace + 9) * 24 * 60, delta_min))
    if abs(delta_min) < 15:
        delta_min = 15 if delta_min >= 0 else -15
    sub_utc = deadline_dt.astimezone(ZoneInfo("UTC")) + timedelta(minutes=delta_min)
    sub_local = sub_utc.astimezone(ZoneInfo(filer_tz)).replace(second=0, microsecond=0)
    sub_utc = sub_local.astimezone(ZoneInfo("UTC"))
    sub_office = sub_utc.astimezone(office)
    tr.given("submitted", f"Submitted {fmt_date(sub_local.date())} {fmt_time(sub_local.time())} {filer_city} time.")
    if not same_tz:
        tr.derive("sub_office", ["submitted"], f"In {office_city} time: {fmt_date(sub_office.date())} {fmt_time(sub_office.time())}.")
        last = "sub_office"
    else:
        last = "submitted"
    on_time = sub_utc <= deadline_dt.astimezone(ZoneInfo("UTC"))
    late_days = 0 if on_time else (sub_office.date() - deadline).days + (1 if sub_office.time() > cutoff or sub_office.date() > deadline else 0)
    late_days = max(late_days, 1) if not on_time else 0
    tier = "on_time" if on_time else ("late_with_fee" if late_days <= grace else "rejected")
    tr.derive("verdict", [last, "cutoff"], f"{'On time' if on_time else f'Late by {late_days} day(s)'}: {tier.replace('_', ' ')}.")

    def tier_of(dl: date, cut: time, sub: datetime) -> tuple[bool, str]:
        dl_dt = datetime.combine(dl, cut, tzinfo=office)
        ok = sub.astimezone(ZoneInfo("UTC")) <= dl_dt.astimezone(ZoneInfo("UTC"))
        so = sub.astimezone(office)
        ld = 0 if ok else max(1, (so.date() - dl).days + (1 if so.time() > cut or so.date() > dl else 0))
        return ok, ("on_time" if ok else ("late_with_fee" if ld <= grace else "rejected"))

    wrongs: dict[str, tuple[date, bool, str]] = {}
    alt_raw = event + timedelta(days=n_days - (0 if count_event_day else 1))
    alt_dl = next_business_day(alt_raw, holidays) if roll else alt_raw
    wrongs["event_day_miscount"] = (alt_dl, *tier_of(alt_dl, cutoff, sub_utc))
    if roll and deadline != raw_end:
        wrongs["no_roll"] = (raw_end, *tier_of(raw_end, cutoff, sub_utc))
    if not roll and next_business_day(raw_end, holidays) != raw_end:
        rd = next_business_day(raw_end, holidays)
        wrongs["rolls_anyway"] = (rd, *tier_of(rd, cutoff, sub_utc))
    if not same_tz:
        naive = sub_local.replace(tzinfo=office)
        wrongs["no_tz_conversion"] = (deadline, *tier_of(deadline, cutoff, naive))
    end_of_day = (deadline, *tier_of(deadline, time(23, 59), sub_utc))
    wrongs["ignores_cutoff"] = end_of_day
    bd = event
    n_bd = 0
    while n_bd < n_days:
        bd += timedelta(days=1)
        if bd.weekday() < 5 and bd not in holidays:
            n_bd += 1
    wrongs["business_days"] = (bd, *tier_of(bd, cutoff, sub_utc))
    variant = f.get("variant", rng.random())
    qv = "noul" if variant < 0.4 else ("tier" if variant < 0.75 else "date")
    truth = ("yes" if on_time else "no") if qv == "noul" else (tier if qv == "tier" else snake_date(deadline))

    def surface_of(k: str) -> str:
        dl, ok, t = wrongs[k]
        return ("yes" if ok else "no") if qv == "noul" else (t if qv == "tier" else snake_date(dl))

    kinds = [k for k in wrongs if surface_of(k) != truth]
    if not kinds and f.get("_retry", 0) < 8:
        return deadline_boolean(rng, names, {**f, "_retry": f.get("_retry", 0) + 1, "variant": variant})
    kind = f.get("kind", rng.choice(kinds) if kinds else rng.choice(list(wrongs)))
    w_dl, w_ok, w_tier = wrongs[kind]
    who = names.person()
    texts = {
        "event_day_miscount": f"Intake note ({who}): day {n_days} after {fmt_date(event)} is {fmt_date(alt_raw)}{', moved to ' + fmt_date(alt_dl) if alt_dl != alt_raw else ''}; the filing is {w_tier.replace('_', ' ')}.",
        "no_roll": f"Intake note ({who}): the period ends {fmt_date(raw_end)}; weekends and holidays do not move a statutory deadline, so the filing is {w_tier.replace('_', ' ')}.",
        "rolls_anyway": f"Intake note ({who}): {fmt_date(raw_end)} was not a business day, so the deadline became {fmt_date(w_dl)} and the filing is {w_tier.replace('_', ' ')}.",
        "no_tz_conversion": f"Intake note ({who}): the portal stamp reads {fmt_time(sub_local.time())} on {fmt_date(sub_local.date())}, which is {'before' if w_ok else 'after'} the {fmt_time(cutoff)} cutoff of {fmt_date(deadline)}; {w_tier.replace('_', ' ')}.",
        "business_days": f"Intake note ({who}): {n_days} business days from {fmt_date(event)} run to {fmt_date(bd)}, so the filing is {w_tier.replace('_', ' ')}.",
        "ignores_cutoff": f"Intake note ({who}): received on {fmt_date(sub_office.date())}, {'within' if w_ok else 'after'} the deadline day of {fmt_date(deadline)}, so {w_tier.replace('_', ' ')}.",
    }
    note = texts[kind]
    hol_text = "; ".join(f"{fmt_date(h)} (office closed)" for h in sorted(holidays)) if holidays else "none in the period"
    rule_count = ("The day on which the event happens counts as day 1." if count_event_day else "The day on which the event happens is not counted; the following day is day 1.")
    rule_roll = ("If the last day is a Saturday, a Sunday or a day on which the office is closed, the deadline moves to the next business day."
                 if roll else "The deadline is not extended when the last day is a Saturday, a Sunday or a day on which the office is closed.")
    state = "\n".join([
        f"{org.upper()} — {filing_kind.upper()} INTAKE, FILE {ref}",
        "",
        *fillers(rng, names),
        "Procedure (extract)",
        f"Rule 3. The {filing_kind} must be received within {n_days} days after {event_kind}. {rule_count} {rule_roll}",
        f"Rule 4. The office is in {office_city}; a filing is received when it reaches the portal, read in {office_city} time. On the last day the cutoff is {fmt_time(cutoff)}; anything later is late.",
        f"Rule 5. A late {filing_kind} is accepted with a late fee if it is received no more than {grace} days after the deadline; later filings are rejected.",
        f"Office closures in the period: {hol_text}.",
        "",
        f"File {ref}: {event_kind[0].upper() + event_kind[1:]} on {fmt_date(event, rng.randint(0, 2))}. The {filing_kind} was submitted through the portal at "
        f"{fmt_time(sub_local.time())} on {fmt_date(sub_local.date(), rng.randint(0, 2))}, {filer_city} time ({utc_offset_str(sub_local)}"
        f"{'' if same_tz else '; ' + office_city + ' is ' + utc_offset_str(sub_office) + ' on that date'}). Filed by {names.person()}.",
        "",
        note,
    ])
    key = f"{event}|{n_days}|{count_event_day}|{roll}|{sorted(h.isoformat() for h in holidays)}|{cutoff}|{office_tz}|{sub_utc.isoformat()}|{grace}"
    extra = {"program_key": key, "version": "v2"}
    surface = surface_of(kind)
    if qv == "noul":
        return SynthItem(family=FAMILY, scenario="deadline_boolean", state=state, qtype="noul",
                         instructions=f"Was the {filing_kind} in file {ref} received on time?",
                         criteria={"true": f"It reached the portal by {fmt_time(cutoff)} {office_city} time on the deadline day.", "false": "It was received after the deadline."},
                         expected=truth, trace=tr, signature=key, distractor=kind, surface_answer=surface if surface != truth else "", extra=extra)
    if qv == "tier":
        return SynthItem(family=FAMILY, scenario="deadline_boolean", state=state, qtype="choice",
                         instructions=f"How is the {filing_kind} in file {ref} to be treated under Rules 3 to 5?",
                         criteria={"on_time": "Received by the deadline; accepted without a fee.",
                                   "late_with_fee": f"Received after the deadline but within {grace} days of it; accepted with the late fee.",
                                   "rejected": f"Received more than {grace} days after the deadline; rejected."},
                         expected=tier, trace=tr, signature=key, distractor=kind, surface_answer=surface if surface != tier else "", extra=extra)
    cands = [w[0] for w in wrongs.values()] + [deadline + timedelta(days=1), deadline - timedelta(days=1), raw_end]
    labels = pick_labels(rng, snake_date(deadline), [snake_date(d) for d in cands])
    return SynthItem(family=FAMILY, scenario="deadline_boolean", state=state, qtype="choice",
                     instructions=f"What is the deadline date for the {filing_kind} in file {ref}?",
                     criteria={l: f"The deadline is {l.replace('_', '-')} ({fmt_time(cutoff)} {office_city} time)." for l in labels},
                     expected=snake_date(deadline), trace=tr, signature=key, distractor=kind,
                     surface_answer=surface if surface != snake_date(deadline) and surface in labels else "", extra=extra)


# ----------------------------------------------------------------------------- scenario 5: AND / OR over several derived numbers


def multi_condition(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    """Tier rules combining derived numbers (12-month spend from quarters, tenure in months, late payments in a
    window, referrals) with AND / OR and an overriding exclusion."""
    f = facts or {}
    tr = Trace()
    flavour = rng.choice(["customer", "supplier"])
    org = names.org(rng.choice(["Holdings", "Services", "Partners", "Industries"]))
    subject = names.org(rng.choice(["Foods", "Logistics", "Textiles", "Engineering"])) if flavour == "supplier" else names.person()
    acct = names.ident("ACC" if flavour == "customer" else "SUP", 6)
    cur = rng.choice(["EUR", "USD", "GBP"])
    assess = f.get("assess", rand_date(rng, 2026, 2027))
    opened = f.get("opened", assess - timedelta(days=rng.randint(200, 1500)))
    tenure = (assess.year - opened.year) * 12 + (assess.month - opened.month) - (1 if assess.day < opened.day else 0)
    tr.given("opened", f"Account opened {fmt_date(opened)}; review date {fmt_date(assess)}.")
    tr.derive("tenure", ["opened"], f"Tenure: {tenure} full months.")
    # five quarters ending before the review date; the oldest is outside the 12-month window
    q_end = [assess - timedelta(days=91 * i) for i in range(1, 6)]
    quarters = f.get("quarters", [(qe, Decimal(rng.randint(3, 60)) * 100) for qe in q_end])
    window_start, _ = add_months_same_day(assess, -12)
    in_win = [(qe, v) for qe, v in quarters if qe > window_start]
    spend = sum(v for _, v in in_win)
    spend_all = sum(v for _, v in quarters)
    for i, (qe, v) in enumerate(quarters):
        tr.given(f"q{i}", f"Quarter ending {fmt_date(qe)}: {cur} {v:,}.")
    tr.derive("spend", [f"q{i}" for i, (qe, v) in enumerate(quarters) if qe > window_start],
              f"Spend in the 12 months to {fmt_date(assess)} (quarters ending after {fmt_date(window_start)}): {cur} {spend:,}.")
    payments = f.get("payments", [])
    if not payments:
        for i in range(rng.randint(5, 7)):
            due = assess - timedelta(days=rng.randint(5, 500))
            paid = due + timedelta(days=rng.choice([-3, -1, 0, 0, 0, 0, 0, 0, 2, 6, 15]))
            payments.append((due, paid))
    late_in_win = [(d, p) for d, p in payments if d > window_start and p > d]
    late_all = [(d, p) for d, p in payments if p > d]
    tr.given("payments", f"{len(payments)} invoices with due and payment dates.")
    tr.derive("late", ["payments"], f"Paid after the due date within the window: {len(late_in_win)}.")
    referrals = f.get("referrals", rng.randint(0, 5))
    tr.given("referrals", f"Referrals: {referrals}.")
    suspended = f.get("suspended", rng.random() < 0.25)
    susp_date = assess - timedelta(days=rng.randint(20, 170)) if suspended else None
    if suspended:
        tr.given("suspension", f"Account suspended on {fmt_date(susp_date)} (within 6 months of the review).")
    A = f.get("A", max(Decimal(2000), (spend * Decimal(str(rng.uniform(0.7, 1.3))) / 500).quantize(Decimal("1")) * 500))
    B = f.get("B", rng.choice([b for b in (6, 12, 18, 24, 36) if abs(b - tenure) <= 12] or [12]))
    C = f.get("C", rng.choice([2, 3, 4]))
    structure = f.get("structure", rng.choice(["or_of_ands", "and_with_or"]))
    c_spend, c_tenure, c_ref, c_late = spend >= A, tenure >= B, referrals >= C, len(late_in_win) == 0
    if structure == "or_of_ands":
        gold_core = (c_spend and c_tenure) or (c_ref and c_late)
        rule_gold = (f"Gold: (spend in the last 12 months of at least {cur} {A:,} AND tenure of at least {B} months) OR (at least {C} referrals AND no invoice paid late in the last 12 months).")
        naive_core = (c_spend and c_tenure and c_ref and c_late)  # reads OR as AND
        naive_desc = "all four conditions must hold"
    else:
        gold_core = c_spend and (c_tenure or c_ref) and c_late
        rule_gold = (f"Gold: spend in the last 12 months of at least {cur} {A:,} AND (tenure of at least {B} months OR at least {C} referrals) AND no invoice paid late in the last 12 months.")
        naive_core = c_spend or (c_tenure and c_ref) or c_late  # swaps AND / OR
        naive_desc = "any of the conditions suffices"
    gold = gold_core and not suspended
    A2, B2 = A // 2, max(6, B // 2)
    silver = (spend >= A2 or tenure >= B2) and not suspended
    tier = "gold" if gold else ("silver" if silver else "standard")
    n_met = sum([c_spend, c_tenure, c_ref, c_late])
    tr.derive("gold_core", ["spend", "tenure", "referrals", "late"], f"Gold conditions: spend {'>=' if c_spend else '<'} {cur} {A:,}, tenure {'>=' if c_tenure else '<'} {B}, referrals {'>=' if c_ref else '<'} {C}, late {len(late_in_win)}: {'met' if gold_core else 'not met'}.")
    tr.derive("tier", ["gold_core"] + (["suspension"] if suspended else []), f"{'Suspension within 6 months blocks gold. ' if suspended and gold_core else ''}Tier: {tier}.")

    def tier_of(g_core: bool, sp: Decimal, susp: bool) -> str:
        g = g_core and not susp
        s = (sp >= A2 or tenure >= B2) and not susp
        return "gold" if g else ("silver" if s else "standard")

    wrongs = {
        "and_or_swapped": tier_of(naive_core, spend, suspended),
        "window_ignored": tier_of(((spend_all >= A and c_tenure) or (c_ref and len(late_all) == 0)) if structure == "or_of_ands"
                                  else (spend_all >= A and (c_tenure or c_ref) and len(late_all) == 0), spend_all, suspended),
    }
    if suspended:
        wrongs["suspension_ignored"] = tier_of(gold_core, spend, False)
    variant = f.get("variant", rng.random())
    qv = "tier" if variant < 0.45 else ("noul" if variant < 0.75 else "score")
    truth = tier if qv == "tier" else (("yes" if gold else "no") if qv == "noul" else str(n_met))
    n_naive = {"and_or_swapped": n_met, "window_ignored": sum([spend_all >= A, c_tenure, c_ref, len(late_all) == 0]), "suspension_ignored": n_met}

    def surface_of(k: str) -> str:
        if qv == "tier":
            return wrongs[k]
        if qv == "noul":
            return "yes" if wrongs[k] == "gold" else "no"
        return str(n_naive[k])

    kinds = [k for k in wrongs if surface_of(k) != truth]
    if not kinds and f.get("_retry", 0) < 8:
        return multi_condition(rng, names, {**f, "_retry": f.get("_retry", 0) + 1, "variant": variant})
    kind = f.get("kind", rng.choice(kinds) if kinds else rng.choice(list(wrongs)))
    who = names.person()
    texts = {
        "and_or_swapped": f"Review note ({who}): reading the gold rule as '{naive_desc}', with spend {cur} {spend:,}, tenure {tenure} months, {referrals} referrals and {len(late_in_win)} late payment(s), the account is {wrongs[kind]}.",
        "window_ignored": f"Review note ({who}): total spend across the five quarters on file is {cur} {spend_all:,} and there are {len(late_all)} late payment(s) in the history, so the account is {wrongs[kind]}.",
        "suspension_ignored": f"Review note ({who}): the suspension was lifted and is no longer relevant; on the numbers the account is {wrongs[kind]}.",
    }
    note = texts[kind]
    inv_lines = [f"- Invoice {names.ident('IN', 6)}: due {fmt_date(d, 1)}, paid {fmt_date(p, 1)}" for d, p in sorted(payments)]
    state = "\n".join([
        f"{org.upper()} — {'CUSTOMER TIER' if flavour == 'customer' else 'PREFERRED SUPPLIER'} REVIEW, {acct}",
        "",
        *fillers(rng, names),
        "Tier rules (extract)",
        rule_gold,
        f"Silver: spend in the last 12 months of at least {cur} {A2:,} OR tenure of at least {B2} months (and not gold).",
        "Standard: otherwise.",
        f"Exclusion: an account that was suspended at any time in the 6 months before the review date cannot be gold or silver, whatever the numbers say.",
        "Definitions: 'the last 12 months' means the 12 months ending on the review date; a quarter counts if it ends inside that window. Tenure is counted in full months from the opening date. An invoice is paid late if the payment date is after the due date.",
        "",
        f"Account {acct} ({subject}): opened {fmt_date(opened, rng.randint(0, 2))}; review date {fmt_date(assess, rng.randint(0, 2))}; referrals credited: {referrals}; "
        f"{'suspended on ' + fmt_date(susp_date, rng.randint(0, 2)) + ' for a documentation gap, lifted after ' + str(rng.randint(5, 30)) + ' days' if suspended else 'no suspensions on record'}.",
        "Quarterly spend on file:",
        *[f"- Quarter ending {fmt_date(qe, 1)}: {cur} {v:,}" for qe, v in sorted(quarters)],
        "Invoices:",
        *inv_lines,
        "",
        note,
    ])
    key = f"{assess}|{opened}|{[(qe.isoformat(), str(v)) for qe, v in quarters]}|{[(d.isoformat(), p.isoformat()) for d, p in payments]}|{referrals}|{susp_date}|{A}|{B}|{C}|{structure}"
    extra = {"program_key": key, "version": "v2"}
    surface = surface_of(kind)
    if qv == "tier":
        return SynthItem(family=FAMILY, scenario="multi_condition", state=state, qtype="choice",
                         instructions=f"Which tier applies to account {acct} at this review?",
                         criteria={"gold": "Gold under the gold rule (and not excluded).", "silver": "Silver (the silver rule is met, gold is not, and not excluded).",
                                   "standard": "Standard (neither rule is met, or the exclusion applies)."},
                         expected=tier, trace=tr, signature=key, distractor=kind, surface_answer=surface if surface != tier else "", extra=extra)
    if qv == "noul":
        return SynthItem(family=FAMILY, scenario="multi_condition", state=state, qtype="noul",
                         instructions=f"Is account {acct} eligible for gold at this review?",
                         criteria={"true": "The gold rule is met and the exclusion does not apply.", "false": "The gold rule is not met, or the exclusion applies."},
                         expected=truth, trace=tr, signature=key, distractor=kind, surface_answer=surface if surface != truth else "", extra=extra)
    tr.derive("n_met", ["gold_core"], f"Number of the four gold conditions met: {n_met}.")
    return SynthItem(family=FAMILY, scenario="multi_condition", state=state, qtype="score",
                     instructions=f"How many of the four gold conditions (spend, tenure, referrals, no late payment) does account {acct} meet?",
                     criteria=[f"{i} of the four conditions" for i in range(5)],
                     expected=str(n_met), trace=tr, signature=key, distractor=kind, surface_answer=surface if surface != str(n_met) else "", extra=extra)


SCENARIOS = [term_vs_cap, fx_lines_cap, effective_expiry, deadline_boolean, multi_condition]
WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]


def make_item(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    fn = f.get("fn", rng.choices(SCENARIOS, weights=WEIGHTS)[0])
    return fn(rng, names)
