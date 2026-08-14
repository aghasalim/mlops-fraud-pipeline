# Incident log

Written as if these were real pages, because a monitoring system is only worth
anything if someone knows what to do when it fires. Each entry records what the
signal was, what it turned out to be, and what the response would be.

Reproduce with `make simulate`.

---

## INC-001 — Prediction distribution collapsed upward

**Signal.** Prediction PSI **0.494** against a threshold of 0.25. Mean predicted
fraud probability moved 0.0429 → 0.0613 (+43%). `TransactionAmt` PSI **11.85**,
the largest single-feature reading ever recorded by this system.

**Detected in.** The first batch after injection — this one is not subtle.

**Cause.** An upstream service began sending amounts in cents rather than
dollars. Every value remained a valid positive float, so nothing raised, nothing
failed validation, and the model scored all of it happily.

**Why it matters.** A 100× shift in the single most interpretable feature
produced *no error anywhere in the stack*. The service returned 200s throughout.
Without the distribution check this runs until somebody notices the review queue
is full.

**Response.**
1. Stop scoring, fall back to the rules engine. A model fed the wrong units is
   worse than no model, because its outputs still look plausible.
2. Confirm against the upstream schema/changelog rather than guessing — a 100×
   factor is characteristic but a genuine pricing change would look similar.
3. Fix at the producer. Do **not** add a "divide by 100 if amounts look large"
   patch in featurisation: that silently corrupts the day the producer is fixed.
4. Backfill re-scores for the affected window.

**Prevention.** A range assertion on `TransactionAmt` at the API boundary would
have rejected this at ingestion rather than detecting it statistically after the
fact. Cheaper, earlier, and it fails loudly. Filed.

---

## INC-002 — Identity provider returned nulls

**Signal.** `id_31` missing rate **0% → 100%** (+100pp against a 35pp threshold).
Prediction PSI only **0.033** — comfortably below the alert line.

**Cause.** Simulated outage of the third-party identity enrichment feed. All
`id_*` columns arrive null; `has_identity` collapses to 0.

**Why this one nearly escaped.** The distribution tests were **blind** to it. A
KS test compares two samples of values; drop every value in a column and there is
nothing left to compare, so the test passes on no data and PSI scores 0. The
model kept serving because LightGBM treats NaN as a routable value, so
predictions stayed in a plausible range.

This was **missed entirely** on the first run and only caught after adding a
missing-rate rule that runs independently of the distribution path. The most
conspicuous failure in production — an entire feed going down — was the one the
statistical machinery could not see.

**Response.**
1. Confirm the outage with the provider before touching the model.
2. Keep serving. Identity is present for only 24.4% of traffic normally, so the
   model is designed to work without it — but expect degraded ranking on the
   affected slice, and say so to downstream consumers rather than letting them
   assume the score means what it usually means.
3. Suppress downstream alerts driven by `has_identity`.

**Prevention.** Per-feature null-rate monitoring is now a first-class rule rather
than a byproduct of distribution testing.

---

## INC-003 — The one that did not fire, and should have

**Signal.** None. Zero alerts.

**What was happening.** Out-of-sample AUC decaying from 0.9376 to **0.8611**
across eight consecutive windows — a loss of up to **0.137** — while the detector
reported every window healthy. Prediction PSI correlates **−0.692** with the AUC
drop, so the signal is not merely uninformative, it points the wrong way.

**Cause.** The input distribution stayed recognisable while the relationship
between inputs and fraud changed. Feature monitoring cannot observe this by
construction: the features are not what moved.

**Why it is in this log.** It is the most important entry. Two injected faults
were caught, and if the project stopped there it would claim a working monitoring
system. It does not have one for the failure mode that was actually present in
this dataset.

**Response.** There is no threshold change that fixes this, and reaching for one
would be the wrong instinct. What it needs:
1. **Label-based monitoring on delayed feedback.** Chargebacks arrive weeks late,
   so a rolling AUC lags — but a lagging true signal beats a real-time one that
   anti-correlates with the truth.
2. **Scheduled retraining** rather than drift-triggered retraining, since the
   drift trigger demonstrably does not fire when retraining is needed.
3. **Drop `day_index`** (see README) — a feature whose production values always
   lie outside the training range can only degrade with time.

**Status.** Open, and honestly the reason this repo exists. The dashboard is
green and the model is getting worse.
