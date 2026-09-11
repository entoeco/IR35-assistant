# Using the ESQ Reviewer Assistant — a guide for the IR35 team

## What this is

When a manager fills in an Employment Status Questionnaire, they tick boxes
and they write explanations. Sometimes the two don't quite agree — a manager
ticks "we can send anyone as a substitute" and then writes "but honestly it
has to be Dave, he's the only one who knows the system." That kind of mismatch
matters for a "reasonable care" determination, and with a long form and a lot
of submissions, it's easy for one to slip past a read-through.

This tool reads every tick-box answer next to its explanation and flags the
pairs that seem to pull in different directions, so you know where to look
closely instead of reading every line of every form with the same amount of
attention. Each flag comes with a plain-English reason: which question, what
was ticked, what was written, and why they don't sit well together.

## What this is not

**It does not decide employment status, and it never will.** It has no
opinion on whether an engagement is inside or outside IR35 — that stays
entirely your call, made the way it's always been made. Nothing on the
screen should be read as a recommendation either way. If it ever looks like
it's nudging towards an answer, that's a bug — please report it (see
"If something looks wrong" below).

Think of it as a second pair of eyes that only ever says "these two things
you wrote don't match, you might want to double check" — never "so the answer
must be X."

## How to use it

1. Open the reviewer screen (`streamlit run src/ui/app.py`, or however your
   team has it running for you) and type your name in the sidebar. Your name
   is recorded against any decision you make, so there's a record of who
   looked at what.
2. Load the submission — either pick it from a list, paste it in, or upload
   the file.
3. The screen checks the submission for personal data and for missing
   answers first, before anything else. You'll see a short note about
   either.
4. Next you'll see a short banner — **"How much attention does this
   submission need?"** — with a five-square meter and a label (*Nothing
   flagged*, *Light review*, *Moderate review*, *Close review*, or *Urgent
   review*). See "How much attention does this need?" below for what this
   is and, just as importantly, what it deliberately is not.
5. Below that, you'll see the flags — the tick-box/explanation pairs worth a
   look, grouped by which part of the test they relate to (substitution,
   control, financial risk, and so on), strongest evidence first.
6. For each flag, read the question, the tick-box answer, the explanation,
   and the reason given. Decide: is this actually a concern, or is it fine
   once you read it properly?
7. Click **Accept — needs follow-up** if it's a genuine inconsistency worth
   raising with the manager, or **Dismiss — not a concern** if, having read
   it, it's fine. You can add a short note either way — useful if you want
   to remind yourself later why you made the call, or want someone else who
   opens this record to see your reasoning.
8. That's it for that flag. Move to the next one. Nothing you click here sets
   the questionnaire's outcome — that's still decided the way it always has
   been, using this tool's flags as one more thing to look at.
9. Below the flags, there's a further section: **"Do the tick-box answers
   agree with each other?"** These aren't a tick-box against its own written
   explanation — they're two tick-boxes checked against each other (for
   example, "the work has already started" and "not applicable, work hasn't
   started yet," both ticked on the same form). Same idea as the flags
   above, same accept/dismiss controls, just a different kind of clash.

## "How much attention does this need?" — a summary banner, not a verdict

Near the top of the screen, before the individual flags, there's a small
banner with a meter of up to five filled squares and a label such as
*Moderate review* or *Urgent review*. This is a summary of everything in
the flags and consistency-checks sections below it — how many there are,
how strong the evidence is for each, and how much that kind of question
usually matters — squeezed into one glance for when you're working through
a stack of submissions and want to know at a look which ones need the most
time.

**It is not a verdict on the submission, and it is not a scale from "not in
IR35" to "in IR35."** We looked at building exactly that — a scale for how
likely each form is to fall inside or outside IR35 — and decided against
it, for the same reason this tool has never shown a status: nothing behind
this screen has been checked against real HMRC outcomes, so a number that
*looks* precise would actually be a guess wearing a lab coat. What the
meter shows instead is squarely about *how much is here to look at*, never
*which way it points*. Two submissions with completely opposite-looking
answers, but the same number and strength of inconsistencies, will show
the same meter reading — because the meter genuinely cannot tell the
difference between them; nothing that goes into it does.

A full five-square "Urgent review" reading means: at least one of the
inconsistencies below touches a fact that case law treats as potentially
decisive by itself, so it's worth checking that one first — not that the
tool thinks the answer is "inside." A "Nothing flagged" reading means
nothing here disagreed with itself — not that the engagement is outside
IR35. Both readings are about the form's own internal consistency, not
about the underlying question.

## "Why this matters" — a new line on some flags

Some flags and findings now carry a short grey line underneath them, headed
**"Why this matters."** For example:

> **Why this matters:** A major factor. Financial risk is weighted heavily
> in most determinations, though it is not treated as part of the essential
> minimum alongside personal service and control.

This tells you how much *that kind* of question typically matters to a
determination in general — it does **not** tell you anything about this
particular submission, and it is not a lean either way. It's the same kind
of information as knowing "substitution and control are the two tests case
law treats as foundational" — background you'd bring to reading any
submission — just placed next to the relevant flag instead of something you
have to remember. It comes from the same fixed rulebook for every
submission; it never looks at what this particular manager wrote.

It's shown in grey, on purpose, and never in the same colours as the
Worth-a-look / Priority / High-priority bands above it — those two things
are answering different questions (how strong is the evidence that these
two answers disagree, versus how much would it matter if they do), and
keeping them visually separate is meant to stop the two getting read as one
combined score. If a "Why this matters" line ever reads like it's telling
you the outcome rather than the topic's general importance, that's a bug —
report it the same way as the status/nudging issue below.

## What the confidence words mean

Each flag is labelled **Worth a look**, **Priority**, or **High priority**
instead of a percentage. That's deliberate — we tested giving a percentage
and it was actively misleading: a flag the tool called "46% confident" turned
out to be a real problem only about 1 time in 9, not anywhere near half the
time. A label like "Priority" makes an honest, more modest claim: in past
testing, flags at this level turned out to be genuine more often than the
ones below it. It is not a promise about this specific case — it's a
"start here" signal, not a verdict.

## What it will get wrong, and that's expected

No automated check catches everything, and this one is no exception:

- **It will miss some real inconsistencies**, especially subtly worded ones
  — a manager who hedges rather than states things outright ("I'd imagine
  it's probably down to them, generally speaking") is genuinely hard for
  the current version to catch reliably. Testing showed it caught this kind
  of wording only about 1 time in 5.
- **It will flag some things that turn out to be fine** once you read the
  full context — that's normal, and part of why every flag needs a human
  to actually look at it rather than being actioned automatically.
- **It reads differently-worded submissions differently.** Two managers
  saying essentially the same thing in a more formal versus a more clipped
  style can get flagged differently even though the underlying facts are
  the same. This is a known limitation, being tracked, and is exactly the
  kind of thing your accept/dismiss decisions help measure over time.

None of this makes the tool useless — it means treat every flag as "worth a
look," not as "definitely a problem," and treat a clean-looking submission as
"nothing obviously jumped out," not as "definitely fine." Same judgement you'd
apply to any other piece of supporting information.

## Privacy

Every submission is checked for personal data — names, phone numbers, email
addresses, postcodes, national insurance numbers — and anything found is
replaced with a placeholder before anything else happens to it. You'll see a
short summary of what was found (a count, never the actual value). This
happens automatically, every time, on every submission.

## If something looks wrong

- **The tool seems to be suggesting a status, or nudging one way or the
  other:** stop, don't act on it, and report it — this should never happen
  and is treated as a bug, not a quirk.
- **A flag makes no sense given what's actually written:** dismiss it with a
  note saying why. That feedback is valuable even if nobody reads the note
  immediately — it's part of the record of how well the tool is actually
  performing.
- **The screen won't load, or throws an error:** that's a technical issue —
  raise it with whoever supports the tool for your team, not with the IR35
  policy team.
- **A question about how a determination should be made, or what a test
  means:** that's a policy question, unrelated to this tool — same process
  and same people as before it existed.

## A few likely questions

**Do I have to action every flag before I can finish a submission?**
No. The flags are there to help you look in the right place; they don't gate
anything. You decide what needs following up.

**What happens to my accept/dismiss decisions?**
They're saved with your name, the time, and the flag they were on, as a
record of what was reviewed. They don't change the questionnaire's status and
they're not visible to the manager who submitted the form.

**Can I change my mind about a decision?**
Yes — reopen it from the same screen and record a new decision. The old one
stays in the record; nothing is deleted.

**Does this replace reading the submission properly?**
No. It points you at specific pairs of answers worth a closer look. The full
submission, and your own judgement, still matter for everything the tool
doesn't flag as well as everything it does.
