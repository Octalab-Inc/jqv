"""temporal_numeric: month-end and leap-year rules, time zones, business days, pro-rating, unit thresholds.

Each scenario builds the facts, solves them with the standard library (datetime / zoneinfo / decimal), records a
Trace, and places a distractor note in the state that argues for a wrong but plausible answer.
"""

from __future__ import annotations

import calendar
import random
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from jqv.synth.common import Names, SynthItem, Trace, money

# ----------------------------------------------------------------------------- calendar helpers

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def fmt_date(d: date, style: int = 0) -> str:
    if style == 0:
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == 1:
        return d.isoformat()
    return f"{MONTHS[d.month - 1][:3]} {d.day}, {d.year}"


def fmt_time(t: time) -> str:
    return t.strftime("%H:%M")


def add_months_same_day(d: date, n: int) -> tuple[date, bool]:
    """Same day number n months later; if that month is shorter, the last day of it. Returns (date, adjusted)."""
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    last = calendar.monthrange(y, m)[1]
    if d.day > last:
        return date(y, m, last), True
    return date(y, m, d.day), False


def utc_offset_str(dt: datetime) -> str:
    off = dt.utcoffset()
    total = int(off.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    h, mi = divmod(total, 60)
    return f"UTC{sign}{h}" + (f":{mi:02d}" if mi else "")


def business_days_after(start: date, n: int, holidays: set[date]) -> tuple[date, list[str]]:
    """The n-th business day after `start` (start itself not counted). Returns the date and the skipped days."""
    d, count, skipped = start, 0, []
    while count < n:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            skipped.append(f"{fmt_date(d)} (weekend)")
            continue
        if d in holidays:
            skipped.append(f"{fmt_date(d)} (holiday)")
            continue
        count += 1
    return d, skipped


def rand_date(rng: random.Random, y0: int = 2025, y1: int = 2028, prefer_month_end: float = 0.0) -> date:
    y = rng.randint(y0, y1)
    m = rng.randint(1, 12)
    if rng.random() < prefer_month_end:
        d = rng.choice([29, 30, 31])
        d = min(d, calendar.monthrange(y, m)[1])
        if d < 29:  # February: use its last day
            d = calendar.monthrange(y, m)[1]
    else:
        d = rng.randint(1, calendar.monthrange(y, m)[1])
    return date(y, m, d)


def snake_date(d: date) -> str:
    return d.strftime("%Y_%m_%d")


def filler_lines(rng: random.Random, names: Names, n: int | None = None) -> list[str]:
    """Realistic but irrelevant record details, so the state is not only the facts that matter."""
    n = rng.randint(3, 6) if n is None else n
    pool = [
        f"Internal routing code: {rng.choice('ABCDEFGH')}{rng.randint(10, 99)}-{rng.randint(100, 999)}.",
        f"Previous contact: {names.person()} on {fmt_date(rand_date(rng, 2024, 2026))} ({rng.choice(['phone', 'email', 'branch visit'])}), no action required.",
        f"File owner: {names.person()}, {rng.choice(['claims team B', 'operations desk 3', 'customer care', 'compliance unit'])}.",
        f"Document control: version {rng.randint(1, 6)}.{rng.randint(0, 9)}, printed {fmt_date(rand_date(rng, 2025, 2027))}.",
        f"Note: attachments {rng.randint(1, 4)} of {rng.randint(4, 9)} were scanned in {rng.choice(['colour', 'greyscale'])}; pages {rng.randint(2, 6)}-{rng.randint(7, 14)} omitted here.",
        f"Customer preference: correspondence in {rng.choice(['English', 'Spanish', 'German', 'Portuguese'])}, {rng.choice(['email', 'post'])}.",
        f"Reference numbers quoted by the customer: {names.ident('REF', 6)}, {names.ident('INV', 5)}.",
        f"Payment method on file: {rng.choice(['direct debit', 'card ending ' + str(rng.randint(1000, 9999)), 'invoice, net 30'])}.",
        f"Escalation flag: {rng.choice(['none', 'none', 'watch list (no impact on handling)', 'VIP account'])}.",
        f"Related items: {names.ident('CASE', 5)} (closed), {names.ident('CASE', 5)} (closed).",
        f"Site contact: {names.person()}, {rng.choice(['ext. ' + str(rng.randint(100, 999)), 'mobile on file'])}.",
        f"Audit trail: record created {fmt_date(rand_date(rng, 2025, 2027))} {rng.randint(8, 18):02d}:{rng.choice(['05', '20', '40', '55'])}, last edited by {names.person()}.",
    ]
    return rng.sample(pool, min(n, len(pool)))


# ----------------------------------------------------------------------------- scenario A: warranty end + tz


def warranty_month_end_tz(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org("Home Appliances") if rng.random() < 0.5 else names.org("Electronics"))
    product = rng.choice(["fridge-freezer", "heat pump", "dishwasher", "induction hob", "washer-dryer", "boiler",
                          "solar inverter", "wall oven", "air conditioner", "water heater"])
    model = f.get("model", f"{rng.choice(['Model', 'Series', 'Type'])} {rng.choice('ABCDEFGHKLMNPRST')}{rng.randint(20, 990)}")
    serial = f.get("serial", names.ident(rng.choice(["SN", "UNIT", "EQ"]), 7))
    cert = f.get("cert", names.ident(rng.choice(["EW", "WC", "GT"]), 7))
    claim_id = f.get("claim_id", names.ident("CL", 6))
    customer = names.person()
    home_city, home_tz = f.get("home_city", names.city())
    delivery = f.get("delivery", rand_date(rng, prefer_month_end=0.55))
    n_months = f.get("n_months", rng.choice([6, 12, 18, 24, 30, 36]))
    tr.given("delivery", f"Delivery date {fmt_date(delivery)}.")
    tr.given("term", f"Term {n_months} months.")

    tentative_y, tentative_m = divmod(delivery.month - 1 + n_months, 12)
    tentative_y, tentative_m = delivery.year + tentative_y, tentative_m + 1
    end_date, adjusted = add_months_same_day(delivery, n_months)
    tr.derive("end_month", ["delivery", "term"], f"{n_months} months after {fmt_date(delivery)} falls in "
              f"{MONTHS[tentative_m - 1]} {tentative_y}.")
    if adjusted:
        leap_note = ""
        if tentative_m == 2:
            leap_note = f" {tentative_y} is {'a leap' if calendar.isleap(tentative_y) else 'not a leap'} year, so"
        tr.derive("end_date", ["end_month"], f"That month has no day {delivery.day}, so the period ends on its last day;"
                  f"{leap_note} that is {fmt_date(end_date)}.")
    else:
        tr.derive("end_date", ["end_month"], f"Day {delivery.day} exists in that month, so the period ends on {fmt_date(end_date)}.")

    # end instant: 24:00 home time on end_date == 00:00 next day
    home = ZoneInfo(home_tz)
    end_instant = datetime.combine(end_date + timedelta(days=1), time(0, 0), tzinfo=home)
    home_off = utc_offset_str(datetime.combine(end_date, time(12, 0), tzinfo=home))

    # claim location and time
    same_tz = f.get("same_tz", rng.random() < 0.3)
    if same_tz:
        claim_city, claim_tz = home_city, home_tz
    else:
        claim_city, claim_tz = names.city()
        while claim_tz == home_tz:
            claim_city, claim_tz = names.city()
    ctz = ZoneInfo(claim_tz)
    # place the claim within +-36h of the end instant, concentrated near it
    delta_minutes = f.get("delta_minutes", int(rng.gauss(0, 9 * 60)))
    delta_minutes = max(-36 * 60, min(36 * 60, delta_minutes))
    if abs(delta_minutes) < 20:  # keep a clear margin
        delta_minutes = 20 if delta_minutes >= 0 else -20
    claim_utc = end_instant.astimezone(ZoneInfo("UTC")) + timedelta(minutes=delta_minutes)
    claim_local = claim_utc.astimezone(ctz)
    claim_local = claim_local.replace(second=0, microsecond=0)
    claim_utc = claim_local.astimezone(ZoneInfo("UTC"))
    claim_off = utc_offset_str(claim_local)
    in_time = claim_utc < end_instant.astimezone(ZoneInfo("UTC"))

    tr.given("claim_time", f"Claim submitted {fmt_date(claim_local.date())} {fmt_time(claim_local.time())} {claim_city} time ({claim_off}).")
    if not same_tz:
        home_local = claim_utc.astimezone(home)
        tr.derive("claim_home_time", ["claim_time"], f"In {home_city} time ({home_off}) that is {fmt_date(home_local.date())} "
                  f"{fmt_time(home_local.time())}.")
        last = "claim_home_time"
    else:
        last = "claim_time"
    tr.derive("verdict", [last, "end_date"], f"The period ends at 24:00 {home_city} time on {fmt_date(end_date)}, so the claim is "
              f"{'within' if in_time else 'outside'} the period.")

    # distractor: three kinds of wrong reasoning
    kind = f.get("kind", rng.choice(["thirty_days", "no_month_end_rule", "no_tz"] if not same_tz else ["thirty_days", "no_month_end_rule"]))
    if kind == "thirty_days":
        wrong_end = delivery + timedelta(days=30 * n_months)
        wrong_in = claim_utc < datetime.combine(wrong_end + timedelta(days=1), time(0, 0), tzinfo=home).astimezone(ZoneInfo("UTC"))
        note = (f"Claims desk note ({names.person()}): counted {n_months} x 30 = {30 * n_months} days from delivery, giving an end "
                f"date of {fmt_date(wrong_end)}; on that basis the claim is {'in time' if wrong_in else 'late'}.")
    elif kind == "no_month_end_rule":
        if adjusted:  # carries the missing days into the next month, a common wrong reading
            wrong_end = end_date + timedelta(days=delivery.day - end_date.day)
        else:
            wrong_end = end_date - timedelta(days=1)  # off-by-one reading of "ends N months later"
        wrong_in = claim_utc < datetime.combine(wrong_end + timedelta(days=1), time(0, 0), tzinfo=home).astimezone(ZoneInfo("UTC"))
        note = (f"Claims desk note ({names.person()}): the period ends on {fmt_date(wrong_end)} by my reckoning, so the claim "
                f"is {'in time' if wrong_in else 'late'}.")
    else:
        wrong_in = claim_local.replace(tzinfo=home) < end_instant  # compares wall-clock times without converting
        note = (f"Claims desk note ({names.person()}): the portal timestamp {fmt_time(claim_local.time())} on "
                f"{fmt_date(claim_local.date())} is {'before' if wrong_in else 'after'} the end of {fmt_date(end_date)}, so the "
                f"claim is {'in time' if wrong_in else 'late'}. (No conversion applied.)")
    surface_in = wrong_in

    doc_kind = rng.choice(["EXTENDED WARRANTY CERTIFICATE", "SERVICE PLAN CERTIFICATE", "PROTECTION PLAN RECORD"])
    rule = rng.choice([
        f"Term. The plan lasts {n_months} months, counted from the delivery date. It expires on the day of the final month whose number matches "
        f"the delivery day; if the final month is too short to contain that day, it expires on that month's final day. Expiry is at 24:00 in the "
        f"local time of the registered address ({home_city}); a report received after that moment is out of time.",
        f"Coverage window: {n_months} calendar months counted from delivery, expiring on the matching day-of-month in the last month (or, if that "
        f"month has fewer days, on its final day), at midnight {home_city} local time. Reports are in time only if received before expiry.",
    ])
    filler = rng.choice([
        f"Product: {org.split()[0]} {model} {product}, serial {serial}. Purchased with the {rng.choice(['standard', 'plus', 'premium'])} plan.",
        f"Product: {model} {product} (serial {serial}). Colour {rng.choice(['graphite', 'white', 'steel', 'ivory'])}.",
    ])
    state = "\n".join([
        f"{org.upper()} — {doc_kind} {cert} (EXTRACT) AND CLAIM RECORD {claim_id}",
        "",
        *filler_lines(rng, names),
        filler,
        f"Registered address: {names.address()}, {home_city}. {home_city} local time on the relevant dates is {home_off}.",
        f"Delivery date: {fmt_date(delivery, rng.randint(0, 2))}.",
        rule,
        "",
        f"Claim {claim_id}: submitted by {customer} through the online portal at {fmt_time(claim_local.time())} on "
        f"{fmt_date(claim_local.date(), rng.randint(0, 2))}, {claim_city} local time ({claim_off}). Reported fault: "
        f"{rng.choice(['no cooling', 'error code E', 'water leak', 'will not start', 'intermittent power loss'])}"
        f"{rng.randint(1, 9) if rng.random() < 0.3 else ''}.",
        "",
        note,
    ])

    variant = f.get("variant", rng.random())
    if variant < 0.5:
        item = SynthItem(
            family="temporal_numeric", scenario="warranty_month_end_tz", state=state, qtype="noul",
            instructions=f"Was claim {claim_id} reported within the {rng.choice(['warranty', 'plan', 'coverage'])} period of certificate {cert}?",
            criteria={"true": f"The claim was received before the end of the period (24:00 {home_city} time on the final day).",
                      "false": "The claim was received after the period had ended."},
            expected="yes" if in_time else "no", trace=tr, signature=f"{delivery}|{n_months}|{home_tz}|{claim_utc.isoformat()}",
            distractor=kind, surface_answer="yes" if surface_in else "no")
    elif variant < 0.8:
        cands = {snake_date(end_date)}
        cands.add(snake_date(delivery + timedelta(days=30 * n_months)))
        cands.add(snake_date(end_date - timedelta(days=1)))
        cands.add(snake_date(end_date + timedelta(days=1)))
        cands.add(snake_date(date(tentative_y, tentative_m, 1) - timedelta(days=1)))
        labels = sorted(cands)[: rng.randint(4, 5)] if len(cands) > 5 else sorted(cands)
        if snake_date(end_date) not in labels:
            labels[-1] = snake_date(end_date)
        rng.shuffle(labels)
        crit = {l: f"The period ends on {l.replace('_', '-')} ({home_city} time)." for l in labels}
        wrong_end_label = snake_date(delivery + timedelta(days=30 * n_months)) if kind == "thirty_days" else None
        item = SynthItem(
            family="temporal_numeric", scenario="warranty_month_end_tz", state=state, qtype="choice",
            instructions=f"On which date does the coverage period of certificate {cert} end?",
            criteria=crit, expected=snake_date(end_date), trace=tr,
            signature=f"{delivery}|{n_months}|{home_tz}|enddate", distractor=kind,
            surface_answer=wrong_end_label if wrong_end_label in crit else "")
    else:
        late_by = (claim_utc - end_instant.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600
        if in_time:
            expected = "in_time"
        elif late_by < 24:
            expected = "late_under_24h"
        else:
            expected = "late_24h_or_more"
        tr.derive("lateness", ["verdict"], "Lateness band: " + expected.replace("_", " ") + ".")
        item = SynthItem(
            family="temporal_numeric", scenario="warranty_month_end_tz", state=state, qtype="choice",
            instructions=f"Classify claim {claim_id} against the end of the coverage period.",
            criteria={"in_time": "Received before the period ended.",
                      "late_under_24h": "Received after the period ended, by less than 24 hours.",
                      "late_24h_or_more": "Received 24 hours or more after the period ended."},
            expected=expected, trace=tr, signature=f"{delivery}|{n_months}|{home_tz}|{claim_utc.isoformat()}|band",
            distractor=kind, surface_answer="in_time" if surface_in else "")
    return item


# ----------------------------------------------------------------------------- scenario B: business-day deadline


def business_day_deadline(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Assurance", "Logistics", "Services", "Partners"])))
    ref = f.get("ref", names.ident(rng.choice(["NT", "REQ", "CASE"]), 6))
    received = f.get("received", rand_date(rng, 2025, 2027))
    while received.weekday() >= 5:
        received += timedelta(days=1)
    cutoff = f.get("cutoff", time(rng.choice([16, 17, 18]), 0))
    rec_time = f.get("rec_time", time(rng.randint(7, 20), rng.choice([0, 15, 30, 45])))
    n_days = f.get("n_days", rng.choice([3, 4, 5, 7, 10]))
    # holidays: 1-3 near the window, some outside
    holidays = set()
    for _ in range(rng.randint(1, 3)):
        h = received + timedelta(days=rng.randint(1, n_days + 6))
        if h.weekday() < 5:
            holidays.add(h)
    far = f.get("far", received + timedelta(days=rng.randint(30, 60)))
    holidays.add(far)
    holidays = set(f.get("holidays", holidays))
    hol_names = ["Founders' Day", "Harvest Holiday", "Civic Day", "Unity Day", "Remembrance Day", "Spring Bank Holiday",
                 "Constitution Day", "Labour Day (observed)"]
    hol_list = sorted(holidays)
    hol_text = "; ".join(f"{fmt_date(h)} ({hol_names[i % len(hol_names)]})" for i, h in enumerate(hol_list))

    after_hours = rec_time > cutoff
    tr.given("received", f"Received {fmt_date(received)} at {fmt_time(rec_time)}.")
    tr.given("rule", f"Respond within {n_days} business days; cutoff {fmt_time(cutoff)}.")
    if after_hours:
        eff = received + timedelta(days=1)
        while eff.weekday() >= 5 or eff in holidays:
            eff += timedelta(days=1)
        tr.derive("effective", ["received", "rule"], f"Received after the {fmt_time(cutoff)} cutoff, so it counts as received on the next "
                  f"business day, {fmt_date(eff)}.")
    else:
        eff = received
        tr.derive("effective", ["received", "rule"], f"Received before the cutoff, so the receipt day is {fmt_date(eff)}.")
    due, skipped = business_days_after(eff, n_days, holidays)
    skipped_h = [s for s in skipped if "holiday" in s]
    if skipped_h:
        tr.derive("holidays_in_window", ["effective"], "Skipped: " + ", ".join(skipped) + ".")
        tr.derive("due", ["holidays_in_window"], f"The {n_days}th business day after {fmt_date(eff)} is {fmt_date(due)}.")
    else:
        tr.derive("due", ["effective"], f"Counting business days (weekends skipped: {sum('weekend' in s for s in skipped)}), the due date is {fmt_date(due)}.")

    # wrong candidates
    cal_due = eff + timedelta(days=n_days)
    no_hol_due, _ = business_days_after(eff, n_days, set())
    no_ah_due, _ = business_days_after(received, n_days, holidays)
    kind = f.get("kind", rng.choice(["calendar_days", "ignored_holiday", "ignored_cutoff"]))
    if kind == "calendar_days":
        wrong = cal_due
        note = f"Team note ({names.person()}): {n_days} days from {fmt_date(received)} is {fmt_date(cal_due)}; I have put that in the tracker."
    elif kind == "ignored_holiday":
        wrong = no_hol_due
        note = f"Team note ({names.person()}): counting weekdays from {fmt_date(eff)}, the deadline is {fmt_date(no_hol_due)}."
    else:
        wrong = no_ah_due
        note = f"Team note ({names.person()}): the request came in on {fmt_date(received)}, so the clock started that day and the deadline is {fmt_date(no_ah_due)}."

    doc = rng.choice(["SERVICE DESK TICKET", "COMPLAINT HANDLING FILE", "REGULATORY NOTICE LOG", "VENDOR QUERY RECORD"])
    state = "\n".join([
        f"{org.upper()} — {doc} {ref}",
        "",
        *filler_lines(rng, names),
        f"Handling standard (section {rng.randint(2, 9)}.{rng.randint(1, 6)}): a written response must be sent within {n_days} business days of receipt. "
        f"The day of receipt is not counted. Business days are Monday to Friday excluding the public holidays listed below. "
        f"Items received after {fmt_time(cutoff)} are treated as received on the next business day.",
        f"Public holidays on file: {hol_text}.",
        "",
        f"Item {ref}: received by {rng.choice(['email', 'secure portal', 'fax', 'registered post scan'])} on {fmt_date(received, rng.randint(0, 2))} "
        f"at {fmt_time(rec_time)}. Subject: {rng.choice(['premium refund query', 'delivery shortfall', 'data access request', 'invoice dispute', 'policy wording question'])}. "
        f"Assigned to {names.person()}.",
        "",
        note,
    ])

    if f.get("variant", rng.random()) < 0.5:
        sent = f.get("sent", due + timedelta(days=rng.choice([-2, -1, 0, 1, 2, 3])))
        on_time = sent <= due
        tr.given("sent", f"Response sent {fmt_date(sent)}.")
        tr.derive("verdict", ["due", "sent"], f"Sent on {fmt_date(sent)}, {'on or before' if on_time else 'after'} the due date.")
        state += f"\n\nResponse: sent on {fmt_date(sent, rng.randint(0, 2))}."
        return SynthItem(
            family="temporal_numeric", scenario="business_day_deadline", state=state, qtype="noul",
            instructions=f"Was the written response to item {ref} sent within the handling standard?",
            criteria={"true": "The response was sent on or before the due date.", "false": "The response was sent after the due date."},
            expected="yes" if on_time else "no", trace=tr, signature=f"{received}|{rec_time}|{n_days}|{sorted(holidays)}|{sent}",
            distractor=kind, surface_answer="yes" if sent <= wrong else "no")
    cands = {snake_date(due), snake_date(cal_due), snake_date(no_hol_due), snake_date(no_ah_due), snake_date(due + timedelta(days=1))}
    labels = sorted(cands)
    rng.shuffle(labels)
    labels = labels[:5]
    if snake_date(due) not in labels:
        labels[0] = snake_date(due)
    crit = {l: f"Due on {l.replace('_', '-')}." for l in labels}
    return SynthItem(
        family="temporal_numeric", scenario="business_day_deadline", state=state, qtype="choice",
        instructions=f"What is the due date for the written response to item {ref}?",
        criteria=crit, expected=snake_date(due), trace=tr, signature=f"{received}|{rec_time}|{n_days}|{sorted(holidays)}|due",
        distractor=kind, surface_answer=snake_date(wrong) if snake_date(wrong) in crit else "")


# ----------------------------------------------------------------------------- scenario C: continuous service months


def service_months_band(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Industries", "Logistics", "Technologies", "Foods", "Medical"])))
    emp = names.person()
    emp_id = f.get("emp_id", names.ident("EMP", 5))
    hire = f.get("hire", rand_date(rng, 2019, 2025, prefer_month_end=0.4))
    ref = f.get("ref", hire + timedelta(days=rng.randint(200, 2200)))
    leave_days = f.get("leave_days", rng.choice([0, 0, 12, 21, 28, 31, 35, 45, 60, 75, 92]))
    limit = f.get("limit", rng.choice([30, 31, 45, 60]))
    tr.given("hire", f"Hired {fmt_date(hire)}.")
    tr.given("ref", f"Reference date {fmt_date(ref)}.")
    start = hire
    leave_text = ""
    if leave_days:
        leave_start = hire + timedelta(days=rng.randint(60, max(61, (ref - hire).days - leave_days - 60)))
        leave_end = leave_start + timedelta(days=leave_days - 1)
        ret = leave_end + timedelta(days=1)
        leave_text = (f"Unpaid leave: {fmt_date(leave_start)} to {fmt_date(leave_end)} inclusive ({leave_days} consecutive days), "
                      f"returned to work on {fmt_date(ret)}.")
        tr.given("leave", f"Unpaid leave of {leave_days} days, return {fmt_date(ret)}.")
        if leave_days > limit:
            start = ret
            tr.derive("start", ["hire", "leave"], f"The leave exceeds {limit} days, so continuous service restarts on the return date {fmt_date(ret)}.")
        else:
            tr.derive("start", ["hire", "leave"], f"The leave does not exceed {limit} days, so continuous service still runs from {fmt_date(hire)}.")
    else:
        tr.derive("start", ["hire"], f"No breaks; service runs from {fmt_date(hire)}.")

    # complete months from start to ref: count month anniversaries <= ref (same-day rule; last day if missing)
    months = 0
    while True:
        anniv, _ = add_months_same_day(start, months + 1)
        if anniv <= ref:
            months += 1
        else:
            break
    tr.derive("months", ["start", "ref"], f"Complete months of service from {fmt_date(start)} to {fmt_date(ref)}: {months} "
              f"(the {months + 1}th month would complete on {fmt_date(add_months_same_day(start, months + 1)[0])}).")
    naive = (ref.year - start.year) * 12 + (ref.month - start.month)  # ignores day of month
    naive_hire = (ref.year - hire.year) * 12 + (ref.month - hire.month)

    a, b, c = sorted(rng.sample([6, 12, 18, 24, 36, 48, 60], 3))
    bands = [f"fewer than {a} complete months", f"{a} to {b - 1} complete months", f"{b} to {c - 1} complete months",
             f"{c} complete months or more"]
    band = 0 if months < a else 1 if months < b else 2 if months < c else 3
    tr.derive("band", ["months"], f"{months} months falls in band {band} ({bands[band]}).")
    naive_band = 0 if naive_hire < a else 1 if naive_hire < b else 2 if naive_hire < c else 3
    kind = "hr_note_calendar_months" if leave_days > limit or naive_hire != months else "hr_note_off_by_one"
    if kind == "hr_note_calendar_months":
        note = (f"HR note ({names.person()}): {hire.strftime('%B %Y')} to {ref.strftime('%B %Y')} is {naive_hire} months, so {emp.split()[0]} "
                f"is in band {naive_band} ({bands[naive_band]}).")
        surface = str(naive_band)
    else:
        nb = min(3, band + 1)
        note = f"HR note ({names.person()}): by my count {emp.split()[0]} has just crossed into band {nb} ({bands[nb]})."
        surface = str(nb)

    state = "\n".join([
        f"{org.upper()} — SENIORITY REVIEW, EMPLOYEE {emp_id}",
        "",
        *filler_lines(rng, names),
        f"Policy {rng.choice(['HR-4', 'SEN-2', 'POL-7'])}.{rng.randint(1, 9)}: Seniority bands are assigned from complete months of continuous service as of the reference date. "
        f"A month completes on the same day-of-month as the service start date (or on the last day of the month if it has no such day). "
        f"Unpaid leave longer than {limit} consecutive days resets the continuous-service start date to the day of return; shorter unpaid leave counts as service.",
        f"Bands: 0 = {bands[0]}; 1 = {bands[1]}; 2 = {bands[2]}; 3 = {bands[3]}.",
        "",
        f"Employee: {emp} ({emp_id}), {rng.choice(['warehouse operative', 'field technician', 'analyst', 'nurse', 'line supervisor', 'buyer'])}. "
        f"Hire date: {fmt_date(hire, rng.randint(0, 2))}. Reference date for this review: {fmt_date(ref, rng.randint(0, 2))}.",
        leave_text,
        "",
        note,
    ])
    if rng.random() < 0.55:
        return SynthItem(
            family="temporal_numeric", scenario="service_months_band", state=state, qtype="score",
            instructions=f"Which seniority band applies to {emp} ({emp_id}) on the reference date?",
            criteria=bands, expected=str(band), trace=tr, signature=f"{hire}|{ref}|{leave_days}|{limit}|{a}{b}{c}",
            distractor=kind, surface_answer=surface)
    thr = f.get("thr", rng.choice([a, b, c]))
    ok = months >= thr
    tr.derive("eligible", ["months"], f"{months} months is {'at least' if ok else 'fewer than'} {thr}, so {'eligible' if ok else 'not eligible'}.")
    naive_ok = naive_hire >= thr
    return SynthItem(
        family="temporal_numeric", scenario="service_months_band", state=state, qtype="noul",
        instructions=f"Does {emp} have at least {thr} complete months of continuous service on the reference date?",
        criteria={"true": f"At least {thr} complete months of continuous service.", "false": f"Fewer than {thr} complete months."},
        expected="yes" if ok else "no", trace=tr, signature=f"{hire}|{ref}|{leave_days}|{limit}|thr{thr}",
        distractor=kind, surface_answer="yes" if naive_ok else "no")


# ----------------------------------------------------------------------------- scenario D: pro-rated invoice


def prorated_invoice(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Systems", "Services", "Technologies", "Energy"])))
    acct = f.get("acct", names.ident("ACC", 6))
    fee = f.get("fee", Decimal(rng.choice([480, 600, 900, 1200, 1500, 1800, 2400, 3600, 4800])))
    start = f.get("start", rand_date(rng, 2025, 2027))
    end = f.get("end", start + timedelta(days=rng.randint(20, 300)))
    inclusive = f.get("inclusive", rng.random() < 0.5)
    days = (end - start).days + (1 if inclusive else 0)
    base_days = 366 if calendar.isleap(start.year) else 365
    discount = f.get("discount", Decimal(rng.choice([5, 10, 12, 15, 20])))
    admin = f.get("admin", Decimal(rng.choice([25, 35, 40, 50, 75])))
    tr.given("fee", f"Annual fee {money(fee)}.")
    tr.given("period", f"Service {fmt_date(start)} to {fmt_date(end)}.")
    tr.derive("days", ["period"], f"Billable days ({'both dates inclusive' if inclusive else 'end date exclusive'}): {days}.")
    pro = (fee * days / base_days)
    tr.derive("prorated", ["days", "fee"], f"Pro-rated fee: {money(fee)} x {days}/{base_days} = {money(pro.quantize(Decimal('0.0001')))} (unrounded).")
    after_disc = pro * (100 - discount) / 100
    tr.derive("discount", ["prorated"], f"Less {discount}% loyalty discount: {money(after_disc.quantize(Decimal('0.0001')))}.")
    total = (after_disc + admin).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    tr.derive("total", ["discount"], f"Plus the {money(admin)} administration fee (not discounted), rounded to the cent: {money(total)}.")

    wrongs = {
        "discount_on_fee": ((pro + admin) * (100 - discount) / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        "day_count": ((fee * (days + (-1 if inclusive else 1)) / base_days) * (100 - discount) / 100 + admin).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        "year_basis": ((fee * days / (365 if base_days == 366 else 366)) * (100 - discount) / 100 + admin).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        "no_prorate": (fee * (100 - discount) / 100 + admin).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
    }
    kind = f.get("kind", rng.choice(["discount_on_fee", "day_count", "no_prorate"]))
    wrong = wrongs[kind]
    if kind == "discount_on_fee":
        note = f"Billing note ({names.person()}): applied the {discount}% discount to the whole invoice including the admin fee: {money(wrong)}."
    elif kind == "day_count":
        note = f"Billing note ({names.person()}): counted {days + (-1 if inclusive else 1)} billable days; total {money(wrong)}."
    else:
        note = f"Billing note ({names.person()}): early termination, so the full annual fee applies less discount plus admin: {money(wrong)}."

    state = "\n".join([
        f"{org.upper()} — TERMINATION INVOICE WORKSHEET, ACCOUNT {acct}",
        "",
        *filler_lines(rng, names),
        f"Billing rule {rng.choice(['B-3', 'INV-7', 'T-2'])}: on early termination the annual fee of {money(fee)} is pro-rated by the number of billable days "
        f"({'the start date and the termination date both count' if inclusive else 'the start date counts; the termination date does not'}) "
        f"divided by the number of days in the calendar year of the service start date. The loyalty discount of {discount}% is then applied to the "
        f"pro-rated fee. A fixed administration fee of {money(admin)} is added after the discount and is never discounted. Round once, to the cent, at the end.",
        "",
        f"Account {acct}: service start {fmt_date(start, rng.randint(0, 2))}; termination effective {fmt_date(end, rng.randint(0, 2))}. "
        f"Plan: {rng.choice(['Standard', 'Business', 'Fleet', 'Campus'])}. Contact: {names.person()}.",
        "",
        note,
    ])
    labels = {f"amount_{str(v).replace('.', '_')}": v for v in {total, *wrongs.values()}}
    keys = list(labels)
    rng.shuffle(keys)
    keys = keys[:5]
    exp_key = f"amount_{str(total).replace('.', '_')}"
    if exp_key not in keys:
        keys[0] = exp_key
    crit = {k: f"{money(labels[k])}" for k in keys}
    wrong_key = f"amount_{str(wrong).replace('.', '_')}"
    return SynthItem(
        family="temporal_numeric", scenario="prorated_invoice", state=state, qtype="choice",
        instructions=f"What is the correct termination invoice total for account {acct}?",
        criteria=crit, expected=exp_key, trace=tr, signature=f"{fee}|{start}|{end}|{inclusive}|{discount}|{admin}",
        distractor=kind, surface_answer=wrong_key if wrong_key in crit else "")


# ----------------------------------------------------------------------------- scenario E: DST cutoff


def dst_cutoff(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Logistics", "Components", "Foods", "Textiles"])))
    home_city, home_tz = f.get("home_city", names.city())
    other_city, other_tz = f.get("other_city", names.city())
    while other_tz == home_tz:
        other_city, other_tz = names.city()
    home, other = ZoneInfo(home_tz), ZoneInfo(other_tz)
    cutoff = f.get("cutoff", time(rng.choice([15, 16, 17]), rng.choice([0, 30])))
    # pick a date near a DST transition of either zone when possible
    year = f.get("year", rng.randint(2025, 2027))
    candidates = []
    for tz in (home, other):
        prev = None
        for day in range(1, 366):
            d = date(year, 1, 1) + timedelta(days=day - 1)
            if d.year != year:
                break
            off = datetime.combine(d, time(12), tzinfo=tz).utcoffset()
            if prev is not None and off != prev:
                candidates.append(d)
            prev = off
    if candidates and rng.random() < 0.7:
        d0 = rng.choice(candidates) + timedelta(days=rng.randint(-3, 3))
    else:
        d0 = date(year, rng.randint(1, 12), rng.randint(1, 28))
    while d0.weekday() >= 5:
        d0 += timedelta(days=1)
    d0 = f.get("d0", d0)
    # order time in other city, within +-4h of the cutoff instant
    cutoff_instant = datetime.combine(d0, cutoff, tzinfo=home)
    delta = f.get("delta", timedelta(minutes=rng.choice([-240, -180, -120, -90, -60, -45, -30, -20, 20, 30, 45, 60, 90, 120, 180, 240])))
    order_instant = (cutoff_instant + delta).astimezone(other).replace(second=0, microsecond=0)
    same_day = order_instant.astimezone(home) <= cutoff_instant and order_instant.astimezone(home).date() == d0
    home_off = utc_offset_str(cutoff_instant)
    other_off = utc_offset_str(order_instant)
    # offsets table (with the seasonal note)
    def season_table(city, tz):
        jan = utc_offset_str(datetime(year, 1, 15, 12, tzinfo=tz))
        jul = utc_offset_str(datetime(year, 7, 15, 12, tzinfo=tz))
        if jan == jul:
            return f"{city}: {jan} all year."
        # find transition dates
        trans = []
        prev = None
        for day in range(1, 366):
            d = date(year, 1, 1) + timedelta(days=day - 1)
            if d.year != year:
                break
            off = utc_offset_str(datetime.combine(d, time(12), tzinfo=tz))
            if prev is not None and off != prev:
                trans.append((d, off))
            prev = off
        parts = [f"{city}: {jan} at the start of {year}"] + [f"{o} from {fmt_date(d)}" for d, o in trans]
        return "; ".join(parts) + "."
    table = "Time zone reference (from the ops handbook): " + season_table(home_city, home) + " " + season_table(other_city, other)
    tr.given("cutoff", f"Cutoff {fmt_time(cutoff)} {home_city} time.")
    tr.given("order", f"Order placed {fmt_date(order_instant.date())} {fmt_time(order_instant.time())} {other_city} time.")
    tr.derive("offsets", ["order"], f"On that date {other_city} is {other_off} and {home_city} is {home_off}.")
    hl = order_instant.astimezone(home)
    tr.derive("home_time", ["offsets"], f"The order time in {home_city} is {fmt_date(hl.date())} {fmt_time(hl.time())}.")
    tr.derive("verdict", ["home_time", "cutoff"], f"That is {'on or before' if same_day else 'after'} the {fmt_time(cutoff)} cutoff on {fmt_date(d0)}, "
              f"so it is {'processed the same day' if same_day else 'not processed that day'}.")
    # distractor: fixed-offset assumption (uses the January offsets) or wall-clock comparison
    kind = f.get("kind", rng.choice(["winter_offset", "wall_clock"]))
    if kind == "winter_offset":
        jan_home = datetime(year, 1, 15, 12, tzinfo=home).utcoffset()
        jan_other = datetime(year, 1, 15, 12, tzinfo=other).utcoffset()
        naive_home = (order_instant.replace(tzinfo=None) - jan_other + jan_home)
        wrong = naive_home <= datetime.combine(d0, cutoff) and naive_home.date() == d0
        note = (f"Dispatch note ({names.person()}): using the usual {utc_offset_str(datetime(year, 1, 15, 12, tzinfo=other))}/"
                f"{utc_offset_str(datetime(year, 1, 15, 12, tzinfo=home))} difference, that is {fmt_time(naive_home.time())} our time, "
                f"so the order {'makes' if wrong else 'misses'} the cutoff.")
    else:
        wrong = order_instant.time() <= cutoff
        note = (f"Dispatch note ({names.person()}): the order timestamp {fmt_time(order_instant.time())} is {'before' if wrong else 'after'} "
                f"{fmt_time(cutoff)}, so it {'makes' if wrong else 'misses'} the cutoff.")
    order_id = f.get("order_id", names.ident("ORD", 7))
    state = "\n".join([
        f"{org.upper()} — SAME-DAY DISPATCH LOG, ORDER {order_id}",
        "",
        *filler_lines(rng, names),
        f"Dispatch rule {rng.choice(['D-1', 'OPS-5', 'SD-2'])}: orders received by {fmt_time(cutoff)} {home_city} local time on a business day are dispatched the same day. "
        f"Orders received later are dispatched on the next business day. The cutoff is always read in {home_city} local time.",
        table,
        "",
        f"Order {order_id}: placed by {names.org(rng.choice(['Foods', 'Medical', 'Industries']))} ({other_city} office) at {fmt_time(order_instant.time())} "
        f"{other_city} local time on {fmt_date(order_instant.date(), rng.randint(0, 2))}. {rng.choice(['12 pallets', '3 crates', '40 cartons', '1 container'])} of "
        f"{rng.choice(['fasteners', 'chilled produce', 'cable assemblies', 'packaging film', 'valves'])}.",
        "",
        note,
    ])
    if rng.random() < 0.6:
        return SynthItem(
            family="temporal_numeric", scenario="dst_cutoff", state=state, qtype="noul",
            instructions=f"Is order {order_id} dispatched on {fmt_date(d0)} under the same-day rule?",
            criteria={"true": f"The order was received by {fmt_time(cutoff)} {home_city} time on {fmt_date(d0)}.",
                      "false": "The order was received after the cutoff (or on another day) and is dispatched later."},
            expected="yes" if same_day else "no", trace=tr, signature=f"{home_tz}|{other_tz}|{order_instant.isoformat()}|{cutoff}",
            distractor=kind, surface_answer="yes" if wrong else "no")
    # choice: the order time expressed in home time
    correct = f"{hl.strftime('%H_%M')}"
    cands = {correct}
    for dm in (-60, 60, -120, 120):
        cands.add((hl + timedelta(minutes=dm)).strftime("%H_%M"))
    labels = sorted(cands)
    rng.shuffle(labels)
    labels = labels[:4] if correct in labels[:4] else [correct] + labels[:3]
    crit = {f"t_{l}": f"{l.replace('_', ':')} {home_city} time" for l in labels}
    return SynthItem(
        family="temporal_numeric", scenario="dst_cutoff", state=state, qtype="choice",
        instructions=f"What time was order {order_id} placed, expressed in {home_city} local time?",
        criteria=crit, expected=f"t_{correct}", trace=tr, signature=f"{home_tz}|{other_tz}|{order_instant.isoformat()}|hometime",
        distractor=kind, surface_answer="")


# ----------------------------------------------------------------------------- scenario F: unit threshold


def unit_threshold(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    org = f.get("org", names.org(rng.choice(["Energy", "Manufacturing", "Foods", "Industries"])))
    site = f.get("site", names.ident("SITE", 4))
    kind_u = f.get("kind_u", rng.choice(["energy", "mass", "volume"]))
    if kind_u == "energy":
        big, small, factor, cap_unit = "MWh", "kWh", Decimal(1000), "MWh"
        cap = Decimal(rng.choice([12, 15, 18, 20, 24, 30]))
    elif kind_u == "mass":
        big, small, factor, cap_unit = "t", "kg", Decimal(1000), "t"
        cap = Decimal(rng.choice([40, 50, 60, 75, 90, 120]))
    else:
        big, small, factor, cap_unit = "m3", "L", Decimal(1000), "m3"
        cap = Decimal(rng.choice([80, 100, 120, 150, 200]))
    months = rng.sample(MONTHS, 3)
    readings = []
    total = Decimal(0)
    for m in months:
        in_small = rng.random() < 0.5
        base = cap / 3 * Decimal(rng.uniform(0.7, 1.35)).quantize(Decimal("0.01"))
        if in_small:
            val = (base * factor).quantize(Decimal("1"))
            readings.append((m, f"{val:,} {small}"))
            total += val / factor
            tr.given(f"r_{m}", f"{m}: {val:,} {small}.")
            tr.derive(f"c_{m}", [f"r_{m}"], f"{m}: {val:,} {small} = {(val / factor).quantize(Decimal('0.001'))} {big}.")
        else:
            val = base.quantize(Decimal("0.01"))
            readings.append((m, f"{val} {big}"))
            total += val
            tr.given(f"r_{m}", f"{m}: {val} {big}.")
    tr.derive("sum", [f"c_{m}" if any(s.endswith(small) for mm, s in readings if mm == m) else f"r_{m}" for m in months],
              f"Quarter total: {total.quantize(Decimal('0.001'))} {big}.")
    exceeded = total > cap
    tr.derive("verdict", ["sum"], f"Cap {cap} {cap_unit}: {'exceeded' if exceeded else 'not exceeded'}.")
    naive_total = sum(Decimal(s.split()[0].replace(',', '')) for _, s in readings)
    note = (f"Site note ({names.person()}): adding the three figures as reported gives {naive_total:,}, "
            f"{'well over' if naive_total > cap else 'under'} the cap of {cap}.")
    state = "\n".join([
        f"{org.upper()} — QUARTERLY {'CONSUMPTION' if kind_u == 'energy' else 'THROUGHPUT'} REPORT, {site}",
        "",
        *filler_lines(rng, names),
        f"Permit condition {rng.randint(3, 14)}: total quarterly {kind_u} {'consumption' if kind_u == 'energy' else 'handled'} must not exceed {cap} {cap_unit}. "
        f"(1 {big} = {int(factor):,} {small}.) Meter readings are reported in whichever unit the meter displays.",
        "",
        "Readings for the quarter:",
        *[f"- {m}: {s}" for m, s in readings],
        "",
        note,
    ])
    if rng.random() < 0.6:
        return SynthItem(
            family="temporal_numeric", scenario="unit_threshold", state=state, qtype="noul",
            instructions=f"Did site {site} exceed the quarterly cap in permit condition?",
            criteria={"true": f"The quarter total exceeds {cap} {cap_unit}.", "false": f"The quarter total is at or below {cap} {cap_unit}."},
            expected="yes" if exceeded else "no", trace=tr, signature=f"{kind_u}|{readings}|{cap}",
            distractor="no_unit_conversion", surface_answer="yes" if naive_total > cap else "no")
    ratio = total / cap
    band = 0 if ratio <= Decimal("0.9") else 1 if ratio <= 1 else 2 if ratio <= Decimal("1.1") else 3
    tr.derive("band", ["verdict"], f"Total / cap = {ratio.quantize(Decimal('0.001'))}, band {band}.")
    bands = ["at or below 90% of the cap", "above 90% of the cap but not above it", "above the cap by up to 10%", "above the cap by more than 10%"]
    naive_ratio = naive_total / cap
    nb = 0 if naive_ratio <= Decimal("0.9") else 1 if naive_ratio <= 1 else 2 if naive_ratio <= Decimal("1.1") else 3
    return SynthItem(
        family="temporal_numeric", scenario="unit_threshold", state=state, qtype="score",
        instructions=f"How does the quarter total for site {site} compare with the cap?",
        criteria=bands, expected=str(band), trace=tr, signature=f"{kind_u}|{readings}|{cap}|band",
        distractor="no_unit_conversion", surface_answer=str(nb))


SCENARIOS = [warranty_month_end_tz, business_day_deadline, service_months_band, prorated_invoice, dst_cutoff, unit_threshold]
WEIGHTS = [0.24, 0.2, 0.16, 0.14, 0.14, 0.12]


def make_item(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    fn = f.get("fn", rng.choices(SCENARIOS, weights=WEIGHTS)[0])
    return fn(rng, names)
