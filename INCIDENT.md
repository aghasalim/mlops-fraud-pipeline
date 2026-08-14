# Incident log

What I broke on purpose, what the monitor did about it, and — the part I found
more useful — what it failed to notice until I fixed it.

Ground rule, same as the other repos: **no number here I did not personally run.**
Reproduce with `make validate && make simulate`.

---

## The setup

The shipped fraud model from [ieee-fraud-ml](https://github.com/aghasalim/ieee-fraud-ml)
serves behind FastAPI. The first 40% of the transaction period (236,216 rows)
defines "normal" and becomes the baseline profile: a 20k-row reservoir sample of
each monitored feature, plus its missing rate, plus the prediction distribution.
Live traffic is replayed from the period after it.

40 features are monitored, chosen by the model's own gain importance rather than
all 443. Monitoring everything sounds thorough and is worse: features carrying
no weight contribute noise to the batch verdict, and every extra feature is
another chance to fire on healthy data.

---

## Incident 0 — the detector was wrong before any failure was injected

Three bugs, all found by testing rather than by reading the code. Every one of
them would have shipped.

### 0a. A feature that drifts by definition

`day_index` scored **PSI 12.4 in every single scenario, including the healthy
control**. It is position in the calendar, so every batch is later than the
baseline and it drifts by construction. It can never indicate a problem, it
occupied one of 40 alert slots permanently, and its only effect was to teach
whoever is on call that the dashboard cries wolf.

Fixed by excluding monotonic time features from monitoring.

### 0b. The most detectable failure in production was invisible

I simulated the identity provider going down — every `id_*` column arrives null.
The monitor said **healthy**.

The KS test drops non-finite values before comparing. A column that is 100% null
has nothing left to compare, so the test passes on an empty sample and PSI
scores 0. A total feed outage produced a *cleaner* report than normal traffic.

Fixed by tracking missing rate per feature as a signal in its own right,
independent of the distribution test. There is a test for it now, because this
is the kind of bug that comes back.

### 0c. My thresholds were guesses, and both were wrong — in opposite directions

`BATCH_DRIFT_SHARE = 0.15` — required 6 of 40 features to fire. Measured
feature-level false-alarm rate on the null is **0 of 40**, so demanding six was
throwing away nearly all sensitivity.

`NULL_JUMP = 0.10` — missing rates move on their own over time. On healthy
forward traffic `D4` shifts **14.2%** with nothing broken, so a 10% trigger
fired on *every* batch including both controls. I went from a detector that
caught 1 of 3 failures to one that flagged 5 of 5 scenarios, controls included.
A 100% alarm rate is exactly as useless as a 0% detection rate.

Both are now set against measured behaviour rather than intuition.

---

## The healthy control — how often does it fire on nothing?

Two disjoint random halves of the *same* period, 20 trials. No drift exists by
construction, so anything flagged is a false alarm.

| correction | features flagged (mean) | KS alone | batch false-alarm rate |
|---|---|---|---|
| none | 0.0 | **3.35** | 0.0% |
| Bonferroni | 0.0 | 0.0 | 0.0% |
| Benjamini-Hochberg | 0.0 | 0.0 | 0.0% |

**KS testing alone flags 3.35 of 40 features on identical data.** That is the
5% you asked for, arriving as noise, every batch, forever — the thing most
drift dashboards display as "3 features drifted".

The correction I built for it is not what fixes it. The **PSI effect-size gate**
is: requiring a shift to be *large*, not merely detectable, removes all of them,
and BH and Bonferroni then have nothing left to do. I built the correction
believing it was the answer and it wasn't. Worth stating plainly, because
claiming credit for the wrong mechanism is its own kind of wrong.

---

## The injected failures

| scenario | caught? | how |
|---|---|---|
| healthy (control) | **no** ✓ | correct — 1/40 features, prediction PSI 0.020 |
| currency units bug | **yes** | prediction PSI **0.494**, `TransactionAmt` PSI 11.9 |
| new customer segment | **yes** | 3/40 features, `card1_freq` PSI 0.564 |
| identity feed outage | **yes** | `id_31` missing rate **+100%** |
| label shift only | **no** ✓ | correct by construction — see below |

**3 of 3 real failures caught, 0 false alarms on 2 controls.**

The outage is caught *only* by the missing-rate rule — 1/40 features and
prediction PSI 0.033, both below their thresholds. Without incident 0b's fix it
sails straight through.

The currency bug is caught most loudly by the **prediction** distribution, not
the inputs: amounts ×100 push predicted fraud from 4.29% to 6.13% mean. That is
the argument for monitoring outputs alongside inputs — one number, no feature
selection needed, and it moved first.

### The scenario that is *supposed* to fail

`label shift only` leaves every input untouched and changes only which
transactions are fraudulent. **No input-distribution monitor can detect this**,
and this one doesn't. It is in the list precisely because a scenario set where
everything gets caught tells you nothing about where the system is blind.

Catching it needs labels, or a proxy: delayed chargeback outcomes, analyst
dispositions on reviewed alerts, or a canary slice with ground truth. None of
that is in this repo, and pretending otherwise would be the whole failure mode
this project is arguing against.

---

## The finding I did not plan for

The eight windows of real forward traffic, scored with a probe model trained
only on the baseline period so the evaluation is genuinely out-of-sample:

| window | share flagged | prediction PSI | flagged? | AUC | AUC drop |
|---|---|---|---|---|---|
| 1 | 0.025 | 0.088 | no | 0.9376 | 0.060 |
| 2 | 0.125 | 0.080 | **yes** | 0.9013 | 0.096 |
| 3 | 0.125 | 0.072 | **yes** | 0.8900 | 0.108 |
| 4 | 0.075 | 0.078 | **yes** | 0.9104 | 0.087 |
| 5 | 0.100 | 0.044 | **yes** | 0.8958 | 0.102 |
| 6 | 0.100 | 0.044 | **yes** | 0.8611 | **0.137** |
| 7 | 0.050 | 0.057 | **yes** | 0.8691 | 0.129 |
| 8 | 0.050 | 0.075 | **yes** | 0.8802 | 0.117 |

The model loses **0.060 to 0.137 AUC** on real traffic with nothing injected and
nothing broken. No bug — just time passing.

Two things matter here.

**First, an in-sample evaluation would have hidden all of it.** My initial run
scored the shipped model on these windows and got 0.97–0.99 with drops near
zero, because that model was trained on this exact data. Real degradation was
invisible until the probe model made the evaluation honest. Same lesson as the
fraud repo, in a new costume: the measurement was broken before the thing being
measured was.

**Second, prediction PSI correlates −0.709 with AUC loss.** It moves the *wrong
way*: the output distribution looks most stable in the windows where the model
is doing worst. With n=8 that is suggestive rather than conclusive, but the
direction alone kills the reasoning "predictions look normal, so we're fine."
Feature-share correlates +0.282, which is weak but at least signed correctly.

At the calibrated threshold the monitor flags **7 of 8** of these windows. That
looks like a high alarm rate until you notice every one of them is genuinely
degraded. I briefly raised the threshold to silence them, then reverted it:
**silence is not the same as health**, and a monitor tuned until it stops
complaining has been tuned into decoration.

---

## What I would actually do about it

Ordered by what the evidence supports, not by what sounds most decisive.

**Currency units bug — page immediately, roll back the upstream change.** This
is a broken input contract, not model decay. Retraining on corrupted data would
bake the bug in. The fix belongs in the feed, and serving should reject rather
than score: a schema/range assertion on `TransactionAmt` at the API boundary
catches this before a prediction is ever made, which is strictly better than
detecting it in aggregate afterwards.

**Identity feed outage — degrade deliberately, do not silently score.** The
model still returns numbers with every `id_*` null, and they are worse numbers.
Better behaviour is an explicit low-confidence path: flag affected predictions,
route them to manual review, and alert the provider. The error-analysis in the
fraud repo already showed AUC is 0.7066 on rows without identity data versus
0.88 overall, so the cost of that outage is quantified rather than guessed.

**New customer segment — do not roll back, retrain.** Nothing is broken; the
population genuinely changed. A rollback restores a model that knows even less
about the new cohort. This is the case for scheduled retraining on recent data,
with the caveat the fraud repo measured: expanding-window CV *underestimates* a
model trained on everything, so the retrain should be evaluated on a fresh
forward window rather than by CV score.

**Gradual temporal decay — schedule retraining, and stop treating drift alerts
as the trigger.** 0.06–0.14 AUC lost over the period with no incident at all.
The honest conclusion from the −0.709 correlation is that input drift is a poor
proxy for the thing I actually care about. The real fix is measuring performance
directly on delayed labels; drift monitoring is what you run while waiting for
them, not a substitute.

---

## What this system still cannot do

- **No labels, so no direct performance monitoring.** The most important signal
  is absent, and everything above is a proxy for it.
- **Concept drift is invisible.** Demonstrated, not assumed — see the label-shift
  control.
- **Batch, not streaming.** Detection latency is one batch.
- **Training/serving skew is unguarded.** `featurize.py` re-implements
  transformations that live in another repo. A shared library or feature store
  is the real fix; a test comparing serving features against training ones on
  known rows is the cheap mitigation, and it is not written yet.
