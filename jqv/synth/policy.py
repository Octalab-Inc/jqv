"""long_policy: 2-4k-token policy documents (definitions, numbered exclusions with exceptions, dated endorsements,
sublimits) plus a claim file whose decision follows from applying the rules, and a clerk/trainee note that argues
for the surface answer. Five domains share the machinery; each has its own rule engine.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from jqv.synth.common import Names, SynthItem, Trace, money, pct
from jqv.synth.temporal import fmt_date, rand_date

# ----------------------------------------------------------------------------- generic clauses (reading burden)


def generic_clauses(rng: random.Random, names: Names, kind: str) -> list[str]:
    n_days = rng.choice([30, 60, 90])
    n_days2 = rng.choice([10, 14, 21, 30])
    pctv = rng.choice([5, 10, 15])
    yrs = rng.choice([1, 2, 3])
    amt = rng.choice([250, 500, 1000, 2500])
    pool = [
        f"Duties after loss. The {kind} must give notice as soon as practicable, protect the property from further damage, keep an accurate record of "
        f"repair expenses, and produce receipts, inventories and other records that we reasonably request. A sworn proof of loss must be signed and "
        f"delivered within {n_days} days after our request. These duties are conditions precedent to payment.",
        f"Concealment or fraud. We do not provide coverage if, whether before or after a loss, the {kind} has intentionally concealed or misrepresented "
        f"any material fact or circumstance, engaged in fraudulent conduct, or made false statements relating to this contract.",
        f"Other insurance. If a loss covered by this contract is also covered by other insurance, we will pay only the proportion of the loss that the "
        f"limit of liability that applies under this contract bears to the total amount of insurance covering the loss.",
        f"Appraisal. If the parties fail to agree on the amount of loss, either may demand an appraisal. Each party selects a competent appraiser within "
        f"{n_days2} days; the appraisers select an umpire. A decision agreed to by any two is binding. Each party pays its own appraiser and shares the cost "
        f"of the umpire equally.",
        f"Subrogation. Any party covered under this contract may be required to assign to us rights of recovery against another party, up to the amount we "
        f"have paid. The covered party must do nothing after loss to impair those rights.",
        f"Loss payment. We will adjust all losses with the {kind} and pay within {n_days} days after we reach agreement, after entry of a final judgment, "
        f"or after the filing of an appraisal award with us, whichever comes first.",
        f"Cancellation. The {kind} may cancel at any time by returning the contract to us or by notifying us in writing of the date cancellation is to take "
        f"effect. We may cancel by mailing notice at least {n_days2} days before the effective date, or {n_days} days where the cancellation is for reasons "
        f"other than non-payment. Unearned premium is refunded pro rata.",
        f"Assignment. Assignment of this contract is not valid unless we give our written consent.",
        f"Territorial limits. Coverage applies to losses occurring within the territory shown in the declarations. Property temporarily removed from the "
        f"premises is covered for up to {pctv}% of the applicable limit, but not for more than {rng.choice([30, 45, 60])} consecutive days.",
        f"Salvage. When we pay for a loss we may take all or part of the damaged property at an agreed or appraised value. There is no abandonment of "
        f"property to us.",
        f"Inflation guard. The limits shown in the declarations increase by {rng.choice([2, 3, 4])}% at each anniversary of the inception date unless "
        f"this endorsement is deleted by the {kind}.",
        f"Mortgagee clause. If a mortgagee is named in the declarations, any loss payable under Coverage A is paid to the mortgagee and the {kind} as "
        f"interests appear. Denial of the {kind}'s claim does not apply to a valid claim of the mortgagee if the mortgagee notifies us of any change in "
        f"ownership, occupancy or substantial change in risk of which it is aware.",
        f"Suit against us. No action can be brought against us unless there has been full compliance with all of the terms of this contract and the action "
        f"is started within {yrs} year{'s' if yrs > 1 else ''} after the date of loss.",
        f"Waiver or change of provisions. A waiver or change of a provision of this contract must be in writing by us to be valid. Our request for an "
        f"appraisal or examination does not waive any of our rights.",
        f"Recovered property. If the {kind} or we recover any property for which we have made payment, the {kind} must notify us. The property may be "
        f"returned to the {kind}, in which case the loss payment is adjusted accordingly.",
        f"Volunteer payments. We will not pay for any expense voluntarily incurred by the {kind} beyond {money(amt)} without our prior agreement, except "
        f"for emergency measures reasonably necessary to protect the property from further loss.",
        f"Electronic data. Loss of or damage to electronic data is covered only for the cost of blank media and the cost of copying from backups, subject "
        f"to a limit of {money(rng.choice([1000, 2500, 5000]))} per occurrence.",
        f"Ordinance or law. Extra cost caused by having to comply with building codes or other regulations when repairing, rebuilding or demolishing "
        f"is covered up to {pctv}% of the Coverage A limit.",
        f"Nuclear hazard. We do not cover loss caused directly or indirectly by nuclear reaction, radiation or radioactive contamination, whether "
        f"controlled or uncontrolled, and whether or not caused by a peril otherwise covered.",
        f"War. We do not cover loss caused by war, including undeclared war, civil war, insurrection, rebellion or revolution, or by warlike acts of a "
        f"military force or personnel.",
        f"Intentional loss. We do not cover loss caused by an act that the {kind} committed, or directed someone else to commit, intending the loss to happen.",
        f"Neglect. We do not cover loss that results from the {kind} failing to take reasonable steps to protect and preserve the property once a loss "
        f"has occurred.",
        f"Governmental action. We do not cover the destruction, confiscation or seizure of property by order of any governmental or public authority, "
        f"except as ordered to prevent the spread of fire.",
        f"Definitions of general application. In this contract 'we' and 'us' mean the insurer that issued it, and 'you' means the {kind} shown on the "
        f"declarations page. 'Business day' means a day other than Saturday, Sunday or a public holiday at the place shown in the declarations.",
    ]
    rng.shuffle(pool)
    return pool


def coverage_clauses(rng: random.Random, kind: str) -> list[str]:
    """Insuring agreements and decoy provisions that add reading burden without changing the decision."""
    lim = rng.choice([1000, 2000, 5000])
    pool = [
        f"Insuring agreement. We will pay for direct physical loss to covered property caused by a peril insured against, unless the loss is excluded or limited in this contract. "
        f"The most we will pay for any one occurrence is the applicable limit shown in the declarations, less the deductible.",
        f"Debris removal. We will pay the reasonable expense of removing debris of covered property after a covered loss, up to {rng.choice([5, 10])}% of the limit that applies to the damaged property, "
        f"as an additional amount of insurance.",
        f"Reasonable repairs. We will pay the reasonable cost incurred by the {kind} for necessary measures taken solely to protect covered property from further damage after a covered loss, "
        f"up to {money(lim)} without prior approval. This does not increase the limit of liability.",
        f"Loss settlement. Losses are settled at the cost to repair or replace with materials of like kind and quality, without deduction for depreciation, provided repair or replacement is completed "
        f"within {rng.choice([180, 365])} days of the loss; otherwise on an actual cash value basis until completion.",
        f"Emergency mitigation. Where water has escaped, the {kind} must arrange drying and dehumidification within 72 hours of discovery; the cost is covered as part of the loss up to {money(lim * 2)}.",
        f"Pair or set. In case of loss to a pair or set we may elect to repair or replace any part to restore the pair or set to its value before the loss, or to pay the difference between actual "
        f"cash value before and after the loss.",
        f"Glass replacement. Loss for damage to glass caused by a covered peril includes the cost of replacing with safety glazing material when required by ordinance or law.",
        f"Fungi and rot. We pay up to {money(rng.choice([2500, 5000, 10000]))} for the removal of fungi, wet or dry rot resulting from a covered loss, including the cost of testing the air and property.",
    ]
    rng.shuffle(pool)
    return pool[: rng.randint(4, 6)]


def correspondence(rng: random.Random, names: Names, ref: str) -> list[str]:
    """A short exchange of messages on the file: dates, people and requests that do not affect the decision."""
    a, b = names.person(), names.person()
    d = rand_date(rng, 2025, 2027)
    return [
        f"[File correspondence, extracted]",
        f"From {a} to {b}, {fmt_date(d)}: Please find attached the contractor's invoice and the photographs for {ref}. The customer asks whether a further claim can be added later; "
        f"I have told them a separate form is needed. Can you confirm the figures shown on the declarations are the current ones?",
        f"From {b} to {a}, {fmt_date(d + timedelta(days=rng.randint(1, 4)))}: Confirmed. Note that the customer's broker has asked to be copied on the outcome letter. "
        f"Please also check whether any endorsement or amendment on the file changes the figures before we draft the letter; the base text alone is not the whole story.",
        f"From {a} to {b}, {fmt_date(d + timedelta(days=rng.randint(5, 9)))}: Draft outcome letter to follow once the position is settled. The trainee has added a note with a recommendation; "
        f"it has not been reviewed.",
    ]


def process_clauses(rng: random.Random, names: Names, kind: str) -> list[str]:
    """Administrative process text for the HR policy and the services agreement (reading burden only)."""
    d1, d2 = rng.choice([5, 10, 15]), rng.choice([20, 30, 45])
    owner = names.person()
    if kind == "hr":
        pool = [
            f"Approvals. A relocation must be approved in writing by the receiving manager and by HR ({owner}'s team) before the move date. Approval is recorded on form RL-{rng.randint(10, 99)} and attached to the employee's file.",
            f"Receipts. Only itemised receipts in the employee's name are accepted. Credit-card statements alone are not sufficient. Receipts in a foreign currency are converted at the company's monthly rate for the month of the expense.",
            f"Advances. An advance of up to 50% of the applicable cap may be requested before the move; it is deducted from the final reimbursement. Unused advances must be repaid within {d2} days.",
            f"Tax treatment. Reimbursements are reported in accordance with local tax rules. Where a reimbursement is taxable, the amount paid to the employee is not grossed up unless the offer letter says otherwise.",
            f"Data protection. Documents submitted with a claim are kept for {rng.choice([3, 5, 7])} years after the claim is closed and are accessible only to HR and to the auditors.",
            f"Appeals. An employee may appeal a decision under this policy within {d1} business days by writing to the HR director. The appeal is decided within {d2} days; the decision on appeal is final.",
            f"Audit. Internal audit samples {rng.choice([5, 10])}% of relocation claims each year. A claim found to be inaccurate may be recovered from salary in accordance with the deductions policy.",
            f"Temporary accommodation. Where temporary accommodation is reimbursed, the employee must move to permanent accommodation within 30 nights; extensions require the approval of the HR director.",
            f"Definitions. 'Start date' means the first day the employee is required to attend the new workplace, as stated in the transfer letter. 'Former home' means the address on file on the date of the transfer letter.",
            f"Interaction with other policies. Travel on the move day is reimbursed under this policy, not under the travel policy. Costs reimbursed under this policy are not claimable again under any other policy.",
        ]
    else:
        pool = [
            f"Monitoring. Availability is measured by the provider's monitoring system from three regions at one-minute intervals. A minute is unavailable if the service fails the health check from at least two regions.",
            f"Incident classification. Each incident is classified by the provider within {d1} business days after it closes; the customer may dispute a classification within {d2} days by giving written reasons. Disputed classifications are reviewed by the service manager ({owner}).",
            f"Maintenance notices. Scheduled maintenance is announced at least 72 hours in advance through the status page. Maintenance that starts inside the window but runs beyond it counts as downtime for the minutes outside the window.",
            f"Reporting. A monthly service report is issued within {rng.choice([5, 7, 10])} business days after the month end and contains the incident log, the uptime calculation and any credit due.",
            f"Credit application. Credits are applied to the invoice for the month after the report and are not paid in cash. Credits do not accumulate beyond the monthly fee.",
            f"Force majeure. Neither party is liable for delay or failure caused by events beyond its reasonable control, including natural disasters, war, and failures of public networks, provided the affected party notifies the other promptly.",
            f"Data handling. The provider processes customer data only as instructed and in accordance with the data processing schedule. Backups are retained for {rng.choice([30, 60, 90])} days.",
            f"Termination for chronic failure. If the uptime falls into the 25% credit tier in any three months of a rolling six-month period, the customer may terminate the affected service on {d2} days' notice without penalty.",
            f"Definitions. 'Available minutes' means minutes in the month in which the service passed the health check. 'Published maintenance window' means the window shown in the service description at the start of the month.",
            f"Dispute resolution. Disputes about a service report are escalated to the account managers, then to the executives named in the agreement, before either party starts proceedings.",
        ]
    rng.shuffle(pool)
    return pool[: rng.randint(7, 10)]


def render_doc(title: str, sub: str, parts: list[tuple[str, list[str]]]) -> str:
    out = [title, sub, "Extract assembled for the coverage decision. Pagination removed. Notes in square brackets were added by the file clerk.", ""]
    for heading, paras in parts:
        out.append("=" * 70)
        out.append(heading)
        out.append("=" * 70)
        out.extend(paras)
        out.append("")
    return "\n".join(out)


def numbered(prefix: str, clauses: list[str], start: int = 1) -> list[str]:
    return [f"{prefix}{start + i}. {c}" for i, c in enumerate(clauses)]


def sub_label(amount) -> str:
    return f"pay_within_{int(amount)}_sublimit"


# ----------------------------------------------------------------------------- domain 1: home water damage


def home_water(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    insurer = f.get("insurer", names.org(rng.choice(["Mutual", "Assurance", "Insurance Group", "Indemnity"])))
    insured = names.person()
    city, _ = f.get("city", names.city())
    policy_no = f.get("policy_no", names.ident("HP", 8))
    claim_no = f.get("claim_no", names.ident("CLM", 7))
    form = f.get("form", f"HF-{rng.randint(2, 9)}")
    R = f.get("R", rng.choice([10, 14, 21]))
    V = f.get("V", rng.choice([30, 45, 60, 90]))
    S1 = f.get("S1", rng.choice([5000, 7500, 10000]))
    S2 = f.get("S2", S1 + rng.choice([2500, 5000, 10000]))
    ded = f.get("ded", rng.choice([500, 1000, 1500, 2500]))
    end_no = f.get("end_no", names.ident("HE", 4))
    endorsement_date = f.get("endorsement_date", rand_date(rng, 2025, 2026))
    renewal = f.get("renewal", endorsement_date + timedelta(days=rng.choice([-120, -60, -20, -5, 5, 20, 60, 120])))
    term_end = renewal + timedelta(days=365)
    # case facts near the thresholds
    duration = f.get("duration", rng.choice([R - 3, R - 1, R, R + 2, R + 7, R * 3, R * 2 + 5, 2]))
    duration = max(1, duration)
    concealed = f.get("concealed", rng.random() < 0.6)
    known = f.get("known", rng.random() < 0.3)
    furnished = f.get("furnished", rng.random() < 0.6)
    absence = f.get("absence", rng.choice([V - 5, V - 1, V + 1, V + 15, V + 30, V * 2, 7, 12]))
    absence = max(1, absence)
    permit = f.get("permit", rng.random() < 0.15)
    loss = f.get("loss", rng.choice([S1 - 1500, S1 + 2400, S2 - 800, S2 + 3000, S2 + 9000, S1 // 2]))
    loss = max(1200, loss)
    loss_date = f.get("loss_date", renewal + timedelta(days=rng.randint(30, 300)))
    tr.given("duration", f"Leak ran about {duration} days.")
    tr.given("concealment", f"Concealed: {'yes' if concealed else 'no'}; known to the insured before discovery: {'yes' if known else 'no'}.")
    tr.given("occupancy", f"Furnished: {'yes' if furnished else 'no'}; absence {absence} consecutive days; renovation permit: {'yes' if permit else 'no'}.")
    tr.given("dates", f"Renewed {fmt_date(renewal)}; endorsement {end_no} applies to terms renewed on or after {fmt_date(endorsement_date)}.")
    tr.given("amounts", f"Loss {money(loss)}, deductible {money(ded)}.")

    vacant = (not furnished) and absence > V
    unoccupied = furnished and absence > V
    occ_text = "vacant" if vacant else "unoccupied" if unoccupied else "neither vacant nor unoccupied"
    tr.derive("occupancy_class", ["occupancy"], f"The dwelling was {'without furnishings' if not furnished else 'furnished'} and empty for {absence} days "
              f"({'more' if absence > V else 'not more'} than {V}), so it was {occ_text} under the definitions.")
    vacancy_excl = vacant and not permit
    if vacant and permit:
        tr.derive("vacancy_exclusion", ["occupancy_class"], "Exception to the vacancy exclusion: renovation under a building permit was in progress, so the exclusion does not apply.")
    else:
        tr.derive("vacancy_exclusion", ["occupancy_class"], f"The vacancy exclusion {'applies' if vacancy_excl else 'does not apply'}.")
    seepage = duration >= R
    tr.derive("seepage", ["duration"], f"{duration} days is {'at least' if seepage else 'less than'} {R} days, so the water {'is' if seepage else 'is not'} 'repeated seepage or leakage'.")
    exception = seepage and concealed and not known
    if seepage:
        tr.derive("concealed_exception", ["seepage", "concealment"], "The concealed-water exception " +
                  ("applies: the water was hidden and unknown to the insured, so cover is restored subject to the sublimit." if exception else
                   ("does not apply: the leak was not concealed." if not concealed else "does not apply: the insured knew of the leak.")))
    endorsed = renewal >= endorsement_date
    S = S2 if endorsed else S1
    if seepage and exception and not vacancy_excl:
        tr.derive("sublimit", ["dates"], f"The term was renewed {'on or after' if endorsed else 'before'} {fmt_date(endorsement_date)}, so the concealed-water sublimit is {money(S)}.")
    if vacancy_excl:
        decision = "deny_vacant_dwelling"
        tr.derive("decision", ["vacancy_exclusion"], "Decision: deny under the vacancy exclusion.")
        payable = 0
    elif seepage and not exception:
        decision = "deny_repeated_leakage"
        tr.derive("decision", ["vacancy_exclusion", "concealed_exception"], "Decision: deny as repeated seepage with no exception.")
        payable = 0
    elif seepage:
        decision = sub_label(S)
        payable = min(loss - ded, S)
        tr.derive("decision", ["vacancy_exclusion", "concealed_exception", "sublimit"], f"Decision: pay, subject to the {money(S)} sublimit "
                  f"({money(loss)} - {money(ded)} = {money(loss - ded)}, capped at {money(S)} = {money(payable)}).")
    else:
        decision = "pay_estimate_less_deductible"
        payable = loss - ded
        tr.derive("decision", ["vacancy_exclusion", "seepage"], f"Decision: sudden discharge, covered in full less the deductible: {money(payable)}.")

    # distractor
    kinds = ["vacancy_note", "old_sublimit", "seepage_note", "full_pay_note"]
    kind = f.get("kind", rng.choice(kinds))
    if kind == "vacancy_note":
        note = (f"[Trainee note, {names.person()}: insured was away {absence} days, over the {V}-day threshold — vacancy exclusion 4.{7}, recommend DENY.]")
        surface = "deny_vacant_dwelling"
    elif kind == "old_sublimit":
        note = f"[Trainee note, {names.person()}: Part E shows the concealed-water sublimit as {money(S1)}, so any payment is capped at {money(S1)}.]"
        surface = sub_label(S1)
    elif kind == "seepage_note":
        note = f"[Trainee note, {names.person()}: leak ran {duration} days — that is repeated seepage under 4.3, recommend DENY.]"
        surface = "deny_repeated_leakage"
    else:
        note = f"[Trainee note, {names.person()}: burst pipe, covered peril, pay the estimate {money(loss)} less {money(ded)} deductible.]"
        surface = "pay_estimate_less_deductible"

    gen = generic_clauses(rng, names, "insured")
    defs = [
        f"'Vacant' means that the dwelling contains no furnishings or contents sufficient for normal residential use and no one has lived in it for more than {V} consecutive days.",
        f"'Unoccupied' means that the dwelling contains furnishings sufficient for normal residential use but no one has lived in it for more than {V} consecutive days. An unoccupied dwelling is not vacant.",
        f"'Repeated seepage or leakage' means water or steam escaping continuously or repeatedly for {R} days or longer.",
        "'Concealed water' means water that escapes within walls, floors, ceilings or under a slab, that is not visible from any living space, and whose escape is unknown to the insured until discovery.",
        "'Sudden and accidental discharge' means an escape of water that begins abruptly and is not repeated seepage or leakage.",
        "'Residence premises' means the one- or two-family dwelling where the insured resides, shown in the declarations, including structures attached to it.",
    ]
    exclusions = [
        "Earth movement, including earthquake, landslide, mudflow, sinkhole, subsidence, or earth sinking, rising or shifting, whether caused by or resulting from human or animal forces or any act of nature.",
        "Water from flood, surface water, waves, tidal water, or overflow of a body of water, or spray from any of these, whether or not driven by wind; water that backs up through sewers or drains except as provided by endorsement.",
        f"Water or steam that seeps or leaks repeatedly out of plumbing, heating or cooling equipment, a fire sprinkler installation, or a domestic appliance. "
        f"Exception 4.3.1: this exclusion does not apply to concealed water, in which case we will pay for the resulting damage to the building subject to the concealed-water sublimit "
        f"shown in Part D, less the deductible. Exception 4.3.1 does not apply if the insured knew of the seepage or leakage before the damage was discovered.",
        "Wear and tear, marring, deterioration, mechanical breakdown, rust or other corrosion, mould, wet or dry rot, or smog, except that resulting ensuing loss from a covered peril is covered.",
        "Freezing of plumbing, heating or cooling equipment, a fire sprinkler installation, or a domestic appliance, unless the insured took reasonable care to keep the building heated, or turned off the water and drained the pipes.",
        "Theft in or to a dwelling under construction, or of materials and supplies for use in the construction, until the dwelling is finished and occupied.",
        f"Vacancy: loss to the building caused by water, freezing, vandalism, glass breakage or theft while the dwelling is vacant. "
        f"Exception 4.7.1: this exclusion does not apply while the dwelling is undergoing renovation under a building permit issued by the local authority. "
        f"This exclusion does not apply to an unoccupied dwelling.",
        "Power failure that originates off the residence premises, unless a covered peril ensues on the premises and then only for the ensuing loss.",
        "Intentional loss, neglect, war, nuclear hazard and governmental action as set out in Part E.",
    ]
    end_text = (f"ENDORSEMENT {end_no} — CONCEALED WATER SUBLIMIT INCREASE. For policy terms renewed or issued on or after {fmt_date(endorsement_date)}, the concealed-water sublimit in Part D is "
                f"{money(S2)} instead of {money(S1)}. All other terms unchanged. [clerk: attached to the file at renewal]")
    claim = [
        f"Claim {claim_no}. Insured: {insured}. Residence premises: {names.address()}, {city}. Date of loss (discovery): {fmt_date(loss_date)}.",
        f"Current term: renewed effective {fmt_date(renewal)}, expiring {fmt_date(term_end)}. Deductible {money(ded)}. Coverage A limit {money(rng.choice([320000, 410000, 550000, 700000]))}.",
        f"Adjuster's narrative ({names.person()}): The insured returned from {rng.choice(['an extended work assignment abroad', 'a stay with family', 'a hospital stay and convalescence', 'a sailing trip'])} "
        f"after {absence} consecutive days away. {'The house remained fully furnished with the insured\'s contents in place' if furnished else 'The house had been cleared of furniture and contents before the absence, pending a planned sale'}"
        f"{' and a kitchen renovation was under way under building permit ' + names.ident('BP', 6) if permit else ''}. On return the insured found water damage to the "
        f"{rng.choice(['kitchen floor and cabinets', 'ground-floor ceiling', 'hallway subfloor', 'bathroom walls'])}. The plumber traced the source to a "
        f"{rng.choice(['pinhole in a copper supply line', 'failed compression joint', 'split flexible hose', 'corroded elbow'])} "
        f"{'inside the wall cavity, not visible from any room' if concealed else 'under the kitchen sink, in plain view'}. From the extent of staining and the plumber's report the leak "
        f"had been running for about {duration} days before discovery."
        + (" The insured states that a damp smell had been noticed and reported to the letting agent before the absence." if known else " The insured had no indication of a leak before discovery."),
        f"Repair estimate (building): {money(loss)}. Contents claim: none.",
        note,
    ]
    n1, n2 = rng.randint(10, 13), rng.randint(7, 10)
    parts = [
        ("PART A — POLICY DECLARATIONS (TERM IN FORCE)", [f"Policyholder: {insured}", f"Policy number: {policy_no}", f"Form: {form} (edition {rng.choice(['03/2022', '04/2023', '01/2024'])})",
                                                 f"Term: {fmt_date(renewal)} to {fmt_date(term_end)}", f"Deductible: {money(ded)} per occurrence"]),
        ("PART B — DEFINITIONS", numbered("", defs)),
        ("PART C — COVERAGES", numbered("2.", coverage_clauses(rng, "insured"))),
        ("PART D — GENERAL CONDITIONS", numbered("3.", gen[:n1])),
        ("PART E — EXCLUSIONS (SECTION 4) AND SUBLIMITS", numbered("4.", exclusions) + [f"Sublimits (Part E): concealed water {money(S1)} per occurrence; mould remediation {money(rng.choice([2500, 5000]))}; "
                                                                                     f"trees, shrubs and plants {money(rng.choice([500, 1000]))} per item."]),
        ("PART F — ADDITIONAL CONDITIONS", numbered("5.", gen[n1:n1 + n2])),
        ("PART G — ENDORSEMENTS ON FILE", [end_text]),
        (f"PART H — CLAIM FILE {claim_no}", claim + correspondence(rng, names, claim_no)),
    ]
    state = render_doc(f"{insurer.upper()}", f"HOMEOWNERS POLICY FORM {form} — EXCERPTS, ENDORSEMENTS AND CLAIM FILE {claim_no}", parts)
    sig = f"hw|{R}|{V}|{S1}|{S2}|{ded}|{duration}|{concealed}|{known}|{furnished}|{absence}|{permit}|{loss}|{renewal}|{endorsement_date}"
    labels = {"deny_vacant_dwelling": "Deny: the vacancy exclusion (4.7) bars the building water damage and no exception restores cover.",
              "deny_repeated_leakage": "Deny: the repeated-seepage exclusion (4.3) bars the loss and the concealed-water exception does not restore it.",
              sub_label(S1): f"Pay: cover is restored by the concealed-water exception, capped at {money(S1)} after the deductible.",
              sub_label(S2): f"Pay: cover is restored by the concealed-water exception, capped at {money(S2)} after the deductible.",
              "pay_estimate_less_deductible": "Pay in full: no exclusion or sublimit applies; the repair estimate minus the deductible is payable."}
    r = f.get("r", rng.random())
    if r < 0.6:
        return SynthItem(family="long_policy", scenario="home_water", state=state, qtype="choice",
                         instructions=f"How should claim {claim_no} for the building water damage be decided under the policy as it stands?",
                         criteria=labels, expected=decision, trace=tr, signature=sig, distractor=kind, surface_answer=surface,
                         extra={"payable": payable})
    if r < 0.85:
        covered = payable > 0
        return SynthItem(family="long_policy", scenario="home_water", state=state, qtype="noul",
                         instructions=f"Is any part of the building water damage in claim {claim_no} payable under the policy?",
                         criteria={"true": "Some amount is payable (with or without a sublimit).", "false": "The claim is excluded and nothing is payable."},
                         expected="yes" if covered else "no", trace=tr, signature=sig + "|noul", distractor=kind,
                         surface_answer="yes" if surface.startswith("pay") else "no", extra={"payable": payable})
    n_bar = int(vacancy_excl) + int(seepage and not exception)
    tr.derive("exclusion_count", ["decision"], f"Exclusions barring cover with no exception: {n_bar}.")
    n_surface = 1 if surface.startswith("deny") else 0
    return SynthItem(family="long_policy", scenario="home_water", state=state, qtype="score",
                     instructions=f"How many exclusions bar cover for the building water damage in claim {claim_no}, after applying any exceptions?",
                     criteria=["No exclusion bars cover.", "Exactly one exclusion bars cover.", "Two exclusions bar cover."],
                     expected=str(n_bar), trace=tr, signature=sig + "|score", distractor=kind, surface_answer=str(n_surface), extra={"payable": payable})


# ----------------------------------------------------------------------------- domain 2: equipment breakdown


def equipment_breakdown(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    insurer = f.get("insurer", names.org(rng.choice(["Underwriters", "Assurance", "Indemnity"])))
    insured = f.get("insured", names.org(rng.choice(["Foods", "Manufacturing", "Textiles", "Logistics"])))
    policy_no = f.get("policy_no", names.ident("EB", 8))
    claim_no = f.get("claim_no", names.ident("CLM", 7))
    equip = f.get("equip", rng.choice(["ammonia compressor", "CNC milling centre", "industrial chiller", "extrusion line drive", "bakery tunnel oven", "walk-in freezer plant"]))
    W = f.get("W", rng.choice([3, 6, 9])        )  # months of documented deterioration -> wear and tear
    M = f.get("M", rng.choice([30, 45, 60, 90]) )  # days overdue -> maintenance lapse
    S1 = f.get("S1", rng.choice([15000, 20000, 25000]))
    S2 = f.get("S2", S1 + rng.choice([10000, 15000, 25000]))
    ded = f.get("ded", rng.choice([1000, 2500, 5000]))
    end_no = f.get("end_no", names.ident("EE", 4))
    endorsement_date = f.get("endorsement_date", rand_date(rng, 2025, 2026))
    inception = f.get("inception", endorsement_date + timedelta(days=rng.choice([-150, -40, -10, 10, 40, 150])))
    signs_months = f.get("signs_months", rng.choice([0, 0, W - 1, W, W + 2, W * 2]))
    surge = f.get("surge", rng.random() < 0.5)
    overdue = f.get("overdue", rng.choice([0, M - 5, M + 3, M + 20, M * 2, 10]))
    booked = f.get("booked", rng.random() < 0.4)
    loss = f.get("loss", rng.choice([S1 - 2000, S1 + 4000, S2 - 1500, S2 + 8000, S1 // 2 + 500]))
    fail_date = f.get("fail_date", inception + timedelta(days=rng.randint(40, 330)))
    tr.given("signs", f"Deterioration documented for {signs_months} months before failure.")
    tr.given("surge", f"Documented electrical surge on the failure day: {'yes' if surge else 'no'}.")
    tr.given("service", f"Service interval exceeded by {overdue} days; service visit booked before failure: {'yes' if booked else 'no'}.")
    tr.given("dates", f"Policy incepted {fmt_date(inception)}; endorsement {end_no} applies to policies incepting on or after {fmt_date(endorsement_date)}.")
    tr.given("amounts", f"Repair cost {money(loss)}, deductible {money(ded)}.")
    maint = overdue > M and not booked
    tr.derive("maintenance", ["service"], f"Overdue by {overdue} days ({'more' if overdue > M else 'not more'} than {M})"
              + (f" but a visit was booked before the failure, so the maintenance exclusion does not apply." if (overdue > M and booked) else
                 (", so the maintenance exclusion applies." if maint else ", so the maintenance exclusion does not apply.")))
    wear = signs_months >= W
    tr.derive("wear", ["signs"], f"{signs_months} months of documented deterioration is {'at least' if wear else 'less than'} {W}, so the failure {'is' if wear else 'is not'} wear and tear.")
    exception = wear and surge
    if wear:
        tr.derive("surge_exception", ["wear", "surge"], "The sudden-electrical-event exception " + ("applies: a documented surge triggered the final failure, so cover is restored subject to the sublimit."
                                                                                                   if exception else "does not apply: no documented surge."))
    endorsed = inception >= endorsement_date
    S = S2 if endorsed else S1
    if wear and exception and not maint:
        tr.derive("sublimit", ["dates"], f"Inception {'on or after' if endorsed else 'before'} {fmt_date(endorsement_date)}: the restored-cover sublimit is {money(S)}.")
    if maint:
        decision, payable = "deny_maintenance_lapse", 0
        tr.derive("decision", ["maintenance"], "Decision: deny for maintenance lapse.")
    elif wear and not exception:
        decision, payable = "deny_wear_and_tear", 0
        tr.derive("decision", ["maintenance", "surge_exception"], "Decision: deny as wear and tear.")
    elif wear:
        decision = sub_label(S); payable = min(loss - ded, S)
        tr.derive("decision", ["maintenance", "surge_exception", "sublimit"], f"Decision: pay subject to {money(S)}: min({money(loss - ded)}, {money(S)}) = {money(payable)}.")
    else:
        decision, payable = "pay_full_repair_less_deductible", loss - ded
        tr.derive("decision", ["maintenance", "wear"], f"Decision: sudden breakdown, pay {money(loss)} less {money(ded)} = {money(payable)}.")
    kind = f.get("kind", rng.choice(["overdue_note", "old_sublimit", "wear_note", "full_pay_note"]))
    if kind == "overdue_note":
        note = f"[Trainee note, {names.person()}: service was {overdue} days overdue — maintenance exclusion 4.4 applies, DENY.]"; surface = "deny_maintenance_lapse"
    elif kind == "old_sublimit":
        note = f"[Trainee note, {names.person()}: schedule shows the restored-cover sublimit as {money(S1)}; cap any payment there.]"; surface = sub_label(S1)
    elif kind == "wear_note":
        note = f"[Trainee note, {names.person()}: the service reports mention {max(signs_months, 1)} month(s) of bearing noise — that is wear and tear, DENY.]"; surface = "deny_wear_and_tear"
    else:
        note = f"[Trainee note, {names.person()}: sudden failure of insured equipment; pay {money(loss)} less {money(ded)} deductible.]"; surface = "pay_full_repair_less_deductible"
    gen = generic_clauses(rng, names, "insured")
    defs = [
        f"'Wear and tear' means gradual deterioration evidenced in service or condition reports over a period of {W} months or more before the breakdown.",
        "'Sudden electrical event' means an electrical surge, arcing or lightning strike recorded by the site power-quality monitor or the utility on the day of the breakdown.",
        f"'Service interval' means the manufacturer's scheduled maintenance interval shown in the equipment schedule. A service is 'overdue' by the number of days elapsed after the interval date.",
        "'Breakdown' means sudden and accidental physical damage to covered equipment that requires repair or replacement.",
        "'Covered equipment' means the items listed in the equipment schedule attached to the declarations.",
    ]
    exclusions = [
        "Loss caused by fire, explosion of a boiler, flood, earth movement, windstorm or hail (these perils are insured elsewhere).",
        "Loss to vacuum tubes, gas tubes, brushes, belts, filters, refrigerant, lubricant or other consumable parts, except as part of a covered repair.",
        f"Loss caused by wear and tear. Exception 4.3.1: where a sudden electrical event on the day of the breakdown is documented, we will pay for the breakdown subject to the "
        f"restored-cover sublimit shown in the schedule, less the deductible, even if wear and tear contributed.",
        f"Loss to equipment whose service is overdue by more than {M} days at the time of the breakdown. Exception 4.4.1: this exclusion does not apply if a service visit had been "
        f"booked with a qualified contractor before the breakdown occurred.",
        "Loss caused by testing, including hydrostatic, pneumatic or gas pressure tests, unless we have agreed in writing.",
        "Loss caused by any defect, error or omission in design, plans, specifications or materials, except ensuing breakdown.",
        "Loss to computer software or data, except as provided under the electronic data condition.",
    ]
    end_text = (f"ENDORSEMENT {end_no} — RESTORED-COVER SUBLIMIT. For policies incepting on or after {fmt_date(endorsement_date)}, the restored-cover sublimit under Exception 4.3.1 "
                f"is {money(S2)} rather than {money(S1)}. [clerk: filed with the schedule]")
    claim = [
        f"Claim {claim_no}. Insured: {insured}. Equipment: {equip}, schedule item {rng.randint(2, 14)}. Date of breakdown: {fmt_date(fail_date)}.",
        f"Policy {policy_no}: incepted {fmt_date(inception)}, 12-month term. Deductible {money(ded)}. Restored-cover sublimit per schedule: {money(S1)}.",
        f"Engineer's report ({names.person()}): the {equip} stopped at {rng.randint(1, 23):02d}:{rng.choice(['05', '20', '45'])} on {fmt_date(fail_date)}. "
        + (f"Site power-quality logs record a {rng.choice([380, 520, 640])} V transient on the incoming supply {rng.randint(2, 40)} minutes before the stop; the utility confirmed a switching surge that day. "
           if surge else "Site power-quality logs show no abnormal events that day. ")
        + (f"Service reports over the previous {signs_months} months note progressive bearing noise and rising vibration readings. " if signs_months else "Service reports up to the breakdown show no abnormal readings. ")
        + f"The last scheduled service was due {overdue} days before the breakdown" + (f"; a service visit had been booked on {fmt_date(fail_date - timedelta(days=rng.randint(3, 25)))} for a date after the breakdown."
                                                                                     if booked else "; no visit had been booked.") if overdue else "The service schedule was up to date.",
        f"Repair quotation: {money(loss)} (parts and labour). Business interruption: not claimed.",
        note,
    ]
    parts = [
        ("PART A — POLICY DECLARATIONS", [f"Policyholder: {insured}", f"Policy number: {policy_no}", f"Form: EB-{rng.randint(2, 6)}", f"Inception: {fmt_date(inception)}", f"Deductible: {money(ded)} per breakdown"]),
        ("PART B — DEFINITIONS", numbered("", defs)),
        ("PART C — COVERAGES", numbered("2.", coverage_clauses(rng, "insured"))),
        ("PART D — CONDITIONS", numbered("3.", gen[:rng.randint(10, 13)])),
        ("PART E — EXCLUSIONS (SECTION 4)", numbered("4.", exclusions)),
        ("PART F — ADDITIONAL CONDITIONS", numbered("5.", gen[13:13 + rng.randint(6, 9)])),
        ("PART G — ENDORSEMENTS ON FILE", [end_text]),
        (f"PART H — CLAIM FILE {claim_no}", claim + correspondence(rng, names, claim_no)),
    ]
    state = render_doc(insurer.upper(), f"EQUIPMENT BREAKDOWN POLICY — EXCERPTS, ENDORSEMENTS AND CLAIM FILE {claim_no}", parts)
    sig = f"eb|{W}|{M}|{S1}|{S2}|{ded}|{signs_months}|{surge}|{overdue}|{booked}|{loss}|{inception}|{endorsement_date}"
    labels = {"deny_maintenance_lapse": f"Deny: service overdue by more than {M} days with no booked visit (Exclusion 4.4).",
              "deny_wear_and_tear": "Deny: wear and tear (Exclusion 4.3) with no exception applying.",
              sub_label(S1): f"Pay: cover restored by Exception 4.3.1, capped at {money(S1)} after the deductible.",
              sub_label(S2): f"Pay: cover restored by Exception 4.3.1, capped at {money(S2)} after the deductible.",
              "pay_full_repair_less_deductible": "Pay in full: no exclusion or sublimit applies; the quotation minus the deductible is payable."}
    r = f.get("r", rng.random())
    if r < 0.65:
        return SynthItem(family="long_policy", scenario="equipment_breakdown", state=state, qtype="choice",
                         instructions=f"How should claim {claim_no} be decided under the policy as it stands?", criteria=labels, expected=decision,
                         trace=tr, signature=sig, distractor=kind, surface_answer=surface, extra={"payable": payable})
    covered = payable > 0
    return SynthItem(family="long_policy", scenario="equipment_breakdown", state=state, qtype="noul",
                     instructions=f"Is any amount payable for claim {claim_no} under the policy?",
                     criteria={"true": "Some amount is payable (with or without a sublimit).", "false": "The claim is excluded and nothing is payable."},
                     expected="yes" if covered else "no", trace=tr, signature=sig + "|noul", distractor=kind,
                     surface_answer="yes" if surface.startswith("pay") else "no", extra={"payable": payable})


# ----------------------------------------------------------------------------- domain 3: trip cancellation


def trip_cancellation(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    insurer = f.get("insurer", names.org(rng.choice(["Assurance", "Travel Cover", "Insurance Group"])))
    insured = names.person()
    policy_no = f.get("policy_no", names.ident("TR", 8))
    claim_no = f.get("claim_no", names.ident("CLM", 7))
    L = f.get("L", rng.choice([60, 90, 120, 180]) )  # lookback days for pre-existing conditions
    K = f.get("K", rng.choice([30, 60, 90])        )  # stability period
    D = f.get("D", rng.choice([10, 14, 21])        )  # CFAR purchase window after deposit
    S = f.get("S", rng.choice([2500, 5000, 7500])  )  # sublimit for stable pre-existing conditions
    cost = f.get("cost", rng.choice([3200, 4800, 6100, 7400, 9900, 12500]))
    deposit = f.get("deposit", rand_date(rng, 2025, 2027))
    purchase = f.get("purchase", deposit + timedelta(days=rng.choice([2, 5, D - 1, D, D + 1, D + 10, 40])))
    cfar = f.get("cfar", rng.random() < 0.6)
    departure = f.get("departure", purchase + timedelta(days=rng.randint(45, 200)))
    reason = f.get("reason", rng.choice(["medical", "medical", "advisory"]))
    tr.given("policy", f"Lookback {L} days; stability {K} days; CFAR window {D} days after deposit; sublimit {money(S)}; non-refundable cost {money(cost)}.")
    tr.given("dates", f"Deposit {fmt_date(deposit)}; policy purchased {fmt_date(purchase)}; departure {fmt_date(departure)}.")
    cfar_valid = cfar and (purchase - deposit).days <= D
    tr.derive("cfar", ["dates", "policy"], ("CFAR rider bought " + (f"{(purchase - deposit).days} days after the deposit, within {D} days, so it is valid." if cfar_valid
                                                                  else f"{(purchase - deposit).days} days after the deposit, outside the {D}-day window, so it is void.")) if cfar else "No CFAR rider was purchased.")
    if reason == "medical":
        last_treat_days_before = f.get("last_treat_days_before", rng.choice([L + 30, L + 5, L - 2, L - 20, 15, 5]))
        stable_days = f.get("stable_days", rng.choice([K + 20, K + 1, K - 3, K - 30, 5]))
        treat_date = purchase - timedelta(days=last_treat_days_before)
        tr.given("medical", f"Last treatment change {last_treat_days_before} days before purchase; condition stable {stable_days} days before purchase.")
        pre = last_treat_days_before <= L
        tr.derive("pre_existing", ["medical", "policy"], f"Treatment {last_treat_days_before} days before purchase is {'within' if pre else 'outside'} the {L}-day lookback, "
                  f"so the condition {'is' if pre else 'is not'} pre-existing.")
        stable = stable_days >= K
        if pre:
            tr.derive("stable", ["pre_existing", "medical"], f"Stable for {stable_days} days, {'at least' if stable else 'less than'} {K}: the stability exception {'applies' if stable else 'does not apply'}.")
        if not pre:
            decision, payable = "pay_full_nonrefundable_costs", cost
            tr.derive("decision", ["pre_existing"], f"Decision: covered medical cancellation, pay {money(cost)}.")
        elif stable:
            decision, payable = sub_label(S), min(cost, S)
            tr.derive("decision", ["stable"], f"Decision: pay subject to the {money(S)} sublimit: {money(payable)}.")
        elif cfar_valid:
            decision, payable = "pay_cfar_75_percent", int(cost * 0.75)
            tr.derive("decision", ["stable", "cfar"], f"Decision: excluded as pre-existing, but the valid CFAR rider pays 75%: {money(payable)}.")
        else:
            decision, payable = "deny_pre_existing_condition", 0
            tr.derive("decision", ["stable", "cfar"], "Decision: deny as a pre-existing condition; no valid CFAR rider.")
        narrative = (f"The insured cancelled on {fmt_date(departure - timedelta(days=rng.randint(3, 20)))} on the advice of their physician because of "
                     f"{rng.choice(['a cardiac condition', 'a back condition', 'a respiratory condition', 'an autoimmune condition'])}. The physician's statement records the last change "
                     f"in treatment on {fmt_date(treat_date)} and no change in medication or symptoms in the {stable_days} days before the policy purchase date.")
        advisory_text = ""
    else:
        adv_offset = f.get("adv_offset", rng.choice([-20, -5, -1, 2, 10]))
        advisory = purchase + timedelta(days=adv_offset)
        tr.given("advisory", f"Government advisory issued {fmt_date(advisory)}.")
        known = advisory < purchase
        tr.derive("known_event", ["advisory", "dates"], f"The advisory was issued {'before' if known else 'after'} the purchase date, so it {'is' if known else 'is not'} a known event.")
        if not known:
            decision, payable = "pay_full_nonrefundable_costs", cost
            tr.derive("decision", ["known_event"], f"Decision: covered cancellation for a post-purchase advisory, pay {money(cost)}.")
        elif cfar_valid:
            decision, payable = "pay_cfar_75_percent", int(cost * 0.75)
            tr.derive("decision", ["known_event", "cfar"], f"Decision: excluded as a known event, but the valid CFAR rider pays 75%: {money(payable)}.")
        else:
            decision, payable = "deny_known_event", 0
            tr.derive("decision", ["known_event", "cfar"], "Decision: deny as a known event; no valid CFAR rider.")
        narrative = (f"The insured cancelled on {fmt_date(departure - timedelta(days=rng.randint(3, 20)))} after the foreign ministry issued a 'do not travel' advisory for the destination on "
                     f"{fmt_date(advisory)} because of {rng.choice(['civil unrest', 'a volcanic eruption', 'an outbreak', 'severe flooding'])}.")
        advisory_text = ""
    kind = f.get("kind", rng.choice(["deny_note", "full_note", "cfar_note"]))
    if kind == "deny_note":
        note = f"[Trainee note, {names.person()}: " + ("condition existed before the trip — pre-existing, DENY.]" if reason == "medical" else "the situation was in the news before they bought, DENY as known event.]")
        surface = "deny_pre_existing_condition" if reason == "medical" else "deny_known_event"
    elif kind == "full_note":
        note = f"[Trainee note, {names.person()}: cancellation for a covered reason, pay the full non-refundable amount {money(cost)}.]"; surface = "pay_full_nonrefundable_costs"
    else:
        note = f"[Trainee note, {names.person()}: CFAR rider is on the file, so 75% of {money(cost)} is payable whatever the reason.]"; surface = "pay_cfar_75_percent"
    gen = generic_clauses(rng, names, "insured")
    defs = [
        f"'Pre-existing condition' means any injury, sickness or condition for which the insured received treatment, advice, a diagnosis, or a change in prescribed medication within the {L} days before the policy purchase date.",
        f"'Stable' means that, in the {K} days before the policy purchase date, the condition had no new symptoms, no change in treatment or medication, and no hospitalisation.",
        "'Known event' means a government travel advisory, strike, natural disaster or outbreak that was announced or publicly reported before the policy purchase date.",
        "'Non-refundable trip cost' means prepaid amounts that the travel supplier will not refund, as shown in the booking statement.",
        "'Policy purchase date' means the date shown in the declarations on which the premium was paid.",
    ]
    exclusions = [
        "Cancellation because of a change of mind, financial circumstances, or business obligations, except as provided by a Cancel-for-Any-Reason rider.",
        "Cancellation caused by a pre-existing condition. Exception 6.2.1: this exclusion does not apply if the condition was stable, in which case we pay the non-refundable trip cost subject to the "
        "pre-existing-condition sublimit shown in the schedule.",
        "Cancellation caused by a known event.",
        "Cancellation because a travel supplier ceases operations, unless the Supplier Default endorsement is on file.",
        "Cancellation caused by war, nuclear hazard, or the insured's intentional act.",
        "Expenses that are refundable, recoverable from any other source, or paid with frequent-traveller points.",
    ]
    end_text = (f"ENDORSEMENT CFAR-{rng.randint(10, 99)} — CANCEL FOR ANY REASON. If bought no later than {D} days after the first trip payment, this rider pays 75% of the non-refundable trip cost when the insured "
                f"cancels for a reason not otherwise covered, provided the cancellation is made at least 48 hours before departure. A rider purchased outside that window is void and the premium is refunded. "
                + (f"[clerk: rider purchased with the policy on {fmt_date(purchase)}]" if cfar else "[clerk: rider offered, not purchased]"))
    claim = [
        f"Claim {claim_no}. Insured: {insured}. Policy {policy_no} purchased {fmt_date(purchase)}. Initial trip deposit paid {fmt_date(deposit)}. Departure {fmt_date(departure)}.",
        f"Non-refundable trip cost per booking statement: {money(cost)}. Pre-existing-condition sublimit per schedule: {money(S)}.",
        f"Claim narrative ({names.person()}): {narrative}",
        note,
    ]
    parts = [
        ("PART A — POLICY DECLARATIONS", [f"Policyholder: {insured}", f"Policy number: {policy_no}", f"Plan: {rng.choice(['Voyager', 'Explorer', 'Compass'])} {rng.choice(['Standard', 'Plus'])}",
                                   f"Policy purchase date: {fmt_date(purchase)}", f"Trip cost insured: {money(cost)}"]),
        ("PART B — DEFINITIONS", numbered("", defs)),
        ("PART C — COVERAGES", numbered("2.", coverage_clauses(rng, "insured"))),
        ("PART D — CONDITIONS", numbered("3.", gen[:rng.randint(10, 13)])),
        ("PART E — TRIP CANCELLATION EXCLUSIONS (SECTION 6)", numbered("6.", exclusions)),
        ("PART F — GENERAL EXCLUSIONS AND CONDITIONS", numbered("7.", gen[13:13 + rng.randint(6, 9)])),
        ("PART G — ENDORSEMENTS", [end_text]),
        (f"PART H — CLAIM FILE {claim_no}", claim + correspondence(rng, names, claim_no)),
    ]
    state = render_doc(insurer.upper(), f"TRAVEL PROTECTION POLICY — EXCERPTS, RIDERS AND CLAIM FILE {claim_no}", parts)
    sig = f"tc|{L}|{K}|{D}|{S}|{cost}|{deposit}|{purchase}|{cfar}|{reason}|{tr.rationale()[:80]}"
    labels = {"deny_pre_existing_condition": "Excluded as a pre-existing condition with no exception or rider applying.",
              "deny_known_event": "Excluded as a known event with no rider applying.",
              "pay_cfar_75_percent": "Not otherwise covered; the Cancel-for-Any-Reason rider pays 75% of the non-refundable trip cost.",
              sub_label(S): f"Pay: the stability exception applies, capped at {money(S)}.",
              "pay_full_nonrefundable_costs": "Covered cancellation; pay the full non-refundable trip cost."}
    if f.get("r", rng.random()) < 0.7:
        return SynthItem(family="long_policy", scenario="trip_cancellation", state=state, qtype="choice",
                         instructions=f"How should claim {claim_no} be decided under the policy and riders on file?", criteria=labels, expected=decision,
                         trace=tr, signature=sig, distractor=kind, surface_answer=surface, extra={"payable": payable})
    return SynthItem(family="long_policy", scenario="trip_cancellation", state=state, qtype="noul",
                     instructions=f"Is any amount payable for claim {claim_no}?",
                     criteria={"true": "Some amount is payable under the policy or a rider.", "false": "Nothing is payable."},
                     expected="yes" if payable > 0 else "no", trace=tr, signature=sig + "|noul", distractor=kind,
                     surface_answer="yes" if surface.startswith("pay") else "no", extra={"payable": payable})


# ----------------------------------------------------------------------------- domain 4: relocation reimbursement (HR policy)


def relocation_reimbursement(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    employer = f.get("employer", names.org(rng.choice(["Technologies", "Industries", "Systems", "Energy"])))
    emp = names.person()
    emp_id = f.get("emp_id", names.ident("EMP", 5))
    case_no = f.get("case_no", names.ident("REL", 6))
    Dkm = f.get("Dkm", rng.choice([40, 50, 60, 80]))
    T = f.get("T", rng.choice([60, 90, 120]))
    caps_old = f.get("caps_old", {"G1": rng.choice([3000, 4000]), "G2": rng.choice([6000, 7500]), "G3": rng.choice([10000, 12000])})
    bump = f.get("bump", rng.choice([1000, 1500, 2500]))
    caps_new = {g: c + bump for g, c in caps_old.items()}
    supplement = f.get("supplement", rng.choice([1000, 1500, 2000]))
    amend_date = f.get("amend_date", rand_date(rng, 2025, 2026))
    grade = f.get("grade", rng.choice(["G1", "G2", "G3"]))
    start = f.get("start", amend_date + timedelta(days=rng.choice([-90, -30, -3, 3, 30, 90])))
    old_commute = f.get("old_commute", rng.randint(5, 40))
    new_commute = f.get("new_commute", old_commute + rng.choice([Dkm - 8, Dkm - 1, Dkm, Dkm + 5, Dkm + 30, 12]))
    dependants = f.get("dependants", rng.random() < 0.5)
    leave_days = f.get("leave_days", rng.choice([0, 0, 0, 14, 25, 40]))
    claim_delay = f.get("claim_delay", rng.choice([T - 10, T - 1, T + 2, T + 10, T + 30, 20]))
    claim_date = start + timedelta(days=claim_delay)
    expenses = f.get("expenses", rng.choice([caps_old[grade] - 700, caps_old[grade] + 400, caps_new[grade] - 200, caps_new[grade] + 900, caps_new[grade] + supplement + 300, 2200]))
    tr.given("rule", f"Distance test {Dkm} km; claim within {T} days of start (extended by approved leave days); caps by grade; supplement {money(supplement)} with dependants.")
    tr.given("facts", f"Old commute {old_commute} km, new {new_commute} km; grade {grade}; start {fmt_date(start)}; claim {fmt_date(claim_date)}; leave {leave_days} days; dependants {'yes' if dependants else 'no'}; expenses {money(expenses)}.")
    diff = new_commute - old_commute
    dist_ok = diff >= Dkm
    tr.derive("distance", ["facts", "rule"], f"The commute increases by {diff} km, {'at least' if dist_ok else 'less than'} {Dkm} km: distance test {'met' if dist_ok else 'failed'}.")
    deadline = start + timedelta(days=T + leave_days)
    on_time = claim_date <= deadline
    tr.derive("deadline", ["facts", "rule"], f"Deadline: {T} days after {fmt_date(start)}" + (f" plus {leave_days} leave days" if leave_days else "") + f" = {fmt_date(deadline)}; the claim on {fmt_date(claim_date)} is {'on time' if on_time else 'late'}.")
    amended = start >= amend_date
    cap = (caps_new if amended else caps_old)[grade]
    cap_total = cap + (supplement if dependants else 0)
    if dist_ok and on_time:
        tr.derive("cap", ["facts", "rule"], f"Start date {'on or after' if amended else 'before'} the amendment date {fmt_date(amend_date)}: grade {grade} cap {money(cap)}"
                  + (f" plus the {money(supplement)} dependants supplement = {money(cap + supplement)}." if dependants else "."))
    if not dist_ok:
        decision, payable = "deny_distance_test", 0
        tr.derive("decision", ["distance"], "Decision: not eligible, distance test failed.")
    elif not on_time:
        decision, payable = "deny_late_claim", 0
        tr.derive("decision", ["distance", "deadline"], "Decision: claim submitted after the deadline, not payable.")
    elif expenses <= cap_total:
        decision, payable = "reimburse_in_full", expenses
        tr.derive("decision", ["distance", "deadline", "cap"], f"Decision: {money(expenses)} is within the cap of {money(cap_total)}, reimburse in full.")
    else:
        decision, payable = f"reimburse_capped_at_{cap_total}", cap_total
        tr.derive("decision", ["distance", "deadline", "cap"], f"Decision: {money(expenses)} exceeds the cap; reimburse {money(cap_total)}.")
    kind = f.get("kind", rng.choice(["old_cap", "late_note", "full_note", "distance_note"]))
    old_total = caps_old[grade] + (supplement if dependants else 0)
    if kind == "old_cap":
        note = f"[HR assistant note, {names.person()}: policy table gives {grade} a cap of {money(caps_old[grade])}{' plus ' + money(supplement) + ' supplement' if dependants else ''}; reimburse up to {money(old_total)}.]"
        surface = "reimburse_in_full" if expenses <= old_total else f"reimburse_capped_at_{old_total}"
    elif kind == "late_note":
        note = f"[HR assistant note, {names.person()}: claim came {claim_delay} days after the start date, past the {T}-day limit — decline.]"; surface = "deny_late_claim"
    elif kind == "full_note":
        note = f"[HR assistant note, {names.person()}: receipts are in order, reimburse {money(expenses)} in full.]"; surface = "reimburse_in_full"
    else:
        note = f"[HR assistant note, {names.person()}: new commute is {new_commute} km, well above {Dkm} km, so the distance test is met.]"
        surface = "reimburse_in_full" if expenses <= cap_total else f"reimburse_capped_at_{cap_total}"
    gen = generic_clauses(rng, names, "employee")
    rules = [
        f"Eligibility (distance test). A relocation is eligible only if the one-way distance from the employee's former home to the new workplace exceeds the distance from the former home to the former workplace by at least {Dkm} km.",
        f"Time limit. Claims must be submitted with receipts within {T} days of the employee's start date at the new workplace. Where the employee is on approved leave during that period, the limit is extended by the number of leave days.",
        f"Caps. Reimbursement is capped by grade: G1 {money(caps_old['G1'])}; G2 {money(caps_old['G2'])}; G3 {money(caps_old['G3'])}. Amounts above the cap are not reimbursed.",
        f"Dependants supplement. Where a spouse, partner or dependent child moves with the employee, the cap is increased by {money(supplement)}.",
        "Excluded costs. Costs of buying or selling a home, mortgage penalties, and temporary accommodation beyond 30 nights are not reimbursable under this policy.",
        "Repayment. An employee who leaves the company within 12 months of the start date must repay the reimbursement pro rata.",
    ]
    amendment = (f"AMENDMENT A-{rng.randint(10, 99)} (effective {fmt_date(amend_date)}): for relocations where the employee's start date at the new workplace is on or after the effective date, the caps in "
                 f"rule 3 are increased to G1 {money(caps_new['G1'])}; G2 {money(caps_new['G2'])}; G3 {money(caps_new['G3'])}. Relocations with earlier start dates remain on the previous caps.")
    file_ = [
        f"Case {case_no}. Employee: {emp} ({emp_id}), grade {grade}. Former workplace: {names.city()[0]} office. New workplace: {names.city()[0]} plant. Start date at new workplace: {fmt_date(start)}.",
        f"Distances (from the former home, one way, per the routing tool): to former workplace {old_commute} km; to new workplace {new_commute} km.",
        f"Household: {'spouse and one child relocated with the employee' if dependants else 'the employee relocated alone'}."
        + (f" Approved unpaid leave: {leave_days} days during the claim period." if leave_days else ""),
        f"Claim submitted {fmt_date(claim_date)} with receipts totalling {money(expenses)} (removal contractor, travel, {rng.choice(['storage', 'utility connections', 'temporary accommodation 12 nights'])}).",
        note,
    ]
    parts = [
        ("SECTION 1 — PURPOSE AND SCOPE", [f"This policy sets out when {employer} reimburses relocation costs for employees who move at the company's request. It applies to all grades and locations."]),
        ("SECTION 2 — RULES", numbered("", rules)),
        ("SECTION 3 — PROCESS", numbered("3.", process_clauses(rng, names, "hr"))),
        ("SECTION 4 — GENERAL PROVISIONS", numbered("4.", gen[:rng.randint(8, 12)])),
        ("SECTION 5 — AMENDMENTS", [amendment]),
        (f"SECTION 6 — CASE FILE {case_no}", file_),
    ]
    state = render_doc(employer.upper(), f"RELOCATION REIMBURSEMENT POLICY (HR-{rng.randint(10, 40)}) — TEXT, AMENDMENTS AND CASE FILE {case_no}", parts)
    sig = f"rr|{Dkm}|{T}|{caps_old}|{bump}|{supplement}|{amend_date}|{grade}|{start}|{old_commute}|{new_commute}|{dependants}|{leave_days}|{claim_delay}|{expenses}"
    labels = {"deny_distance_test": "Not eligible: the distance test is not met.", "deny_late_claim": "Not payable: the claim was submitted after the time limit.",
              "reimburse_in_full": "Eligible and within the applicable cap: reimburse the full receipts.",
              f"reimburse_capped_at_{old_total}": f"Eligible; reimburse {money(old_total)} (cap reached).",
              f"reimburse_capped_at_{cap_total}": f"Eligible; reimburse {money(cap_total)} (cap reached)."}
    if old_total == cap_total:
        labels[f"reimburse_capped_at_{cap_total + bump}"] = f"Eligible; reimburse {money(cap_total + bump)} (cap reached)."
    return SynthItem(family="long_policy", scenario="relocation_reimbursement", state=state, qtype="choice",
                     instructions=f"How should relocation case {case_no} be decided under the policy and its amendments?", criteria=labels, expected=decision,
                     trace=tr, signature=sig, distractor=kind, surface_answer=surface if surface in labels else "", extra={"payable": payable})


# ----------------------------------------------------------------------------- domain 5: SLA service credits


def sla_credits(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    tr = Trace()
    provider = f.get("provider", names.org(rng.choice(["Systems", "Technologies", "Services"])))
    customer = f.get("customer", names.org(rng.choice(["Foods", "Logistics", "Medical", "Partners"])))
    contract = f.get("contract", names.ident("MSA", 6))
    month_minutes = f.get("month_minutes", rng.choice([43200, 44640, 40320]))  # 30, 31, 28 days
    tiers_old = [(Decimal("99.9"), 0), (Decimal("99.5"), 5), (Decimal("99.0"), 10), (Decimal("0"), 25)]
    tiers_new = [(Decimal("99.95"), 0), (Decimal("99.7"), 5), (Decimal("99.2"), 10), (Decimal("0"), 25)]
    amend_date = f.get("amend_date", rand_date(rng, 2025, 2026))
    renewal = f.get("renewal", amend_date + timedelta(days=rng.choice([-100, -30, -2, 2, 30, 100])))
    incidents = []
    n_inc = f.get("n_inc", rng.randint(3, 5))
    for i in range(n_inc):
        cls = rng.choice(["unplanned", "unplanned", "scheduled_in_window", "scheduled_out_of_window", "customer_caused", "force_majeure"])
        dur = rng.choice([8, 12, 20, 35, 48, 65, 90, 120, 150, 240])
        incidents.append((names.ident("INC", 5), cls, dur))
    incidents = f.get("incidents", incidents)
    tr.given("contract", f"Renewed {fmt_date(renewal)}; amendment applies to contracts renewed on or after {fmt_date(amend_date)}.")
    tr.given("incidents", "; ".join(f"{i}: {c.replace('_', ' ')}, {d} min" for i, c, d in incidents))
    excluded = sum(d for _, c, d in incidents if c in ("scheduled_in_window", "customer_caused", "force_majeure"))
    counted = sum(d for _, c, d in incidents if c in ("unplanned", "scheduled_out_of_window"))
    tr.derive("excluded", ["incidents"], f"Excluded from the calculation (scheduled within the window, customer-caused, force majeure): {excluded} min; counted downtime: {counted} min.")
    denom = month_minutes - sum(d for _, c, d in incidents if c in ("scheduled_in_window", "force_majeure"))
    uptime = (Decimal(denom - counted) / Decimal(denom) * 100).quantize(Decimal("0.001"))
    tr.derive("uptime", ["excluded"], f"Uptime = ({denom} - {counted}) / {denom} = {uptime}% (scheduled-window and force-majeure minutes are removed from the denominator; customer-caused minutes stay in it).")
    amended = renewal >= amend_date
    tiers = tiers_new if amended else tiers_old
    tr.derive("tiers", ["contract"], f"Contract renewed {'on or after' if amended else 'before'} {fmt_date(amend_date)}: {'amended' if amended else 'original'} credit tiers apply.")
    credit = next(c for thr, c in tiers if uptime >= thr)
    tr.derive("credit", ["uptime", "tiers"], f"{uptime}% falls in the {credit}% credit tier.")
    naive_uptime = (Decimal(month_minutes - sum(d for _, _, d in incidents)) / Decimal(month_minutes) * 100).quantize(Decimal("0.001"))
    naive_credit = next(c for thr, c in tiers_old if naive_uptime >= thr)
    kind = f.get("kind", rng.choice(["count_everything", "old_tiers"]))
    if kind == "count_everything":
        note = f"[Account manager note, {names.person()}: total outage {sum(d for _, _, d in incidents)} min this month, uptime {naive_uptime}%, so a {naive_credit}% credit is due.]"
        surface = f"credit_{naive_credit}_percent" if naive_credit else "no_credit"
    else:
        oc = next(c for thr, c in tiers_old if uptime >= thr)
        note = f"[Account manager note, {names.person()}: uptime {uptime}% against the MSA schedule ({tiers_old[0][0]}% / {tiers_old[1][0]}% / {tiers_old[2][0]}%) gives a {oc}% credit.]"
        surface = f"credit_{oc}_percent" if oc else "no_credit"
    gen = generic_clauses(rng, names, "customer")
    sched = (f"Service credit schedule (original): monthly uptime at or above {tiers_old[0][0]}%: no credit; at or above {tiers_old[1][0]}% but below {tiers_old[0][0]}%: 5% of the monthly fee; "
             f"at or above {tiers_old[2][0]}% but below {tiers_old[1][0]}%: 10%; below {tiers_old[2][0]}%: 25%.")
    rules = [
        "Uptime calculation. Monthly uptime percentage = (available minutes) / (total minutes in the month minus excluded minutes) x 100, rounded to three decimals.",
        "Excluded minutes. Minutes of scheduled maintenance performed within the published maintenance window and minutes of force majeure are excluded from both the numerator and the denominator.",
        "Customer-caused outages. Minutes of unavailability caused by the customer's own equipment, credentials or configuration are not counted as downtime but remain in the denominator.",
        "Scheduled maintenance outside the window. Maintenance performed outside the published window counts as downtime in full.",
        sched,
        "Claims. Credits must be claimed in writing within 30 days after the end of the month and are applied to the next invoice.",
    ]
    amendment = (f"AMENDMENT {rng.randint(2, 6)} (effective {fmt_date(amend_date)}): for contracts renewed on or after the effective date, the schedule thresholds are {tiers_new[0][0]}% (no credit), "
                 f"{tiers_new[1][0]}% (5%), {tiers_new[2][0]}% (10%), below {tiers_new[2][0]}% (25%). Contracts renewed earlier remain on the original schedule until their next renewal.")
    inc_lines = [f"- {i}: {d} minutes, classified '{c.replace('_', ' ')}'" for i, c, d in incidents]
    file_ = [
        f"Contract {contract} between {provider} (provider) and {customer} (customer). Renewed effective {fmt_date(renewal)}. Month under review: {rng.choice(['March', 'May', 'August', 'October', 'January'])} "
        f"({month_minutes // 1440} days, {month_minutes:,} minutes). Published maintenance window: Sundays 01:00-05:00.",
        "Incident log (durations already reconciled with the monitoring system):",
        *inc_lines,
        note,
    ]
    parts = [
        ("SECTION 1 — SERVICE LEVELS", numbered("1.", rules)),
        ("SECTION 2 — OPERATING PROCEDURES", numbered("2.", process_clauses(rng, names, "sla"))),
        ("SECTION 3 — GENERAL TERMS", numbered("3.", gen[:rng.randint(8, 12)])),
        ("SECTION 4 — AMENDMENTS", [amendment]),
        (f"SECTION 5 — MONTHLY REVIEW, CONTRACT {contract}", file_),
    ]
    state = render_doc(provider.upper(), f"MASTER SERVICES AGREEMENT {contract} — SERVICE LEVEL SCHEDULE, AMENDMENTS AND MONTHLY REVIEW", parts)
    sig = f"sla|{month_minutes}|{renewal}|{amend_date}|{incidents}"
    exp_label = f"credit_{credit}_percent" if credit else "no_credit"
    if f.get("r", rng.random()) < 0.6:
        labels = {"no_credit": "No service credit is due.", "credit_5_percent": "A 5% service credit is due.", "credit_10_percent": "A 10% service credit is due.",
                  "credit_25_percent": "A 25% service credit is due."}
        return SynthItem(family="long_policy", scenario="sla_credits", state=state, qtype="choice",
                         instructions=f"What service credit, if any, is due to the customer under contract {contract} for the month under review?",
                         criteria=labels, expected=exp_label, trace=tr, signature=sig, distractor=kind, surface_answer=surface, extra={"uptime": str(uptime)})
    idx = {0: 0, 5: 1, 10: 2, 25: 3}[credit]
    sidx = {"no_credit": 0, "credit_5_percent": 1, "credit_10_percent": 2, "credit_25_percent": 3}[surface]
    return SynthItem(family="long_policy", scenario="sla_credits", state=state, qtype="score",
                     instructions=f"Which service credit tier applies to contract {contract} for the month under review?",
                     criteria=["No credit.", "5% credit.", "10% credit.", "25% credit."], expected=str(idx), trace=tr, signature=sig + "|score",
                     distractor=kind, surface_answer=str(sidx), extra={"uptime": str(uptime)})


SCENARIOS = [home_water, equipment_breakdown, trip_cancellation, relocation_reimbursement, sla_credits]
WEIGHTS = [0.26, 0.2, 0.2, 0.17, 0.17]


def make_item(rng: random.Random, names: Names, facts: dict | None = None) -> SynthItem:
    f = facts or {}
    fn = f.get("fn", rng.choices(SCENARIOS, weights=WEIGHTS)[0])
    return fn(rng, names)
