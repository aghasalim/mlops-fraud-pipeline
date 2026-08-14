# Production ML pipeline — with the monitoring actually tested

[![ci](https://github.com/aghasalim/mlops-fraud-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/mlops-fraud-pipeline/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The fraud model from [ieee-fraud-ml](https://github.com/aghasalim/ieee-fraud-ml)
put behind a FastAPI service, a CI gate, a model registry and drift monitoring —
by a third-year Applied Computer Science (AI) student.

A monitoring dashboard that has never been tested against a real failure is a
nice-looking chart. So the deliverable here isn't "I set up monitoring", it's a
measured answer to three questions, in this order:

1. How often does it alarm when **nothing** is wrong?
2. Does what it flags **predict the model actually getting worse**?
3. Only then — does it catch failures I inject on purpose?

Question 3 is the one everybody demos. Questions 1 and 2 are the ones that decide
whether the answer to 3 means anything.

---

## The headline: it catches what I inject, and misses what actually happened

**Injected failures** (`make simulate`), after calibration:

| scenario | caught? | what fired |
|---|---|---|
| healthy (control) | correctly silent | — |
| currency units bug (amounts ×100) | **CAUGHT** | prediction PSI 0.494 |
| identity feed outage (columns all null) | **CAUGHT** | `id_31` null rate +100% |
| new customer segment (35% of rows) | **missed** | 0.075 share, below threshold |
| label shift, inputs untouched | correctly invisible | nothing to see |

**0 false alarms across 8 fault-free windows.** 2 of 3 real faults caught. So far
this reads like a success.

**Then I checked whether any of it predicts degradation** (`make validate`). A
model trained on the baseline period only, scored out-of-sample on the eight
later windows:

| window | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AUC | 0.9376 | 0.9013 | 0.8900 | 0.9104 | 0.8958 | **0.8611** | 0.8691 | 0.8802 |
| detector says drifted? | no | no | no | no | no | **no** | no | no |

**The model loses up to 0.137 AUC and the monitor never fires.** Worse, the
prediction-drift signal correlates **−0.692** with the actual AUC drop — the
windows where predictions look *most* like the baseline are the ones where the
model performs *worst*.

That is not a bug in the implementation. It is the thing input-distribution
monitoring cannot do: here the inputs stay recognisable while the relationship
between inputs and fraud changes. No KS test on features can see that, because
the features are not what moved. The only honest fixes are labels, delayed
feedback, or a proxy — none of which a feature dashboard provides.

**If I had only run the injected-failure demo, I would have shipped a monitor
with a 100% catch rate on the failures I imagined and a 0% catch rate on the
failure that was actually occurring in the data.**

---

## The calibration, which is where the real work was

My first thresholds were guesses, and the guesses were wrong in both directions.

**Attempt 1** — KS test per feature at α=0.05. On healthy data this flags
**2.6 of 40 features** (6.5%, which is just α doing what α does). A drift
dashboard built this way fires forever and trains whoever is on call to ignore
it. I expected the multiple-testing correction to be the fix. It wasn't — adding
a **PSI effect-size gate** took it to 0, and Bonferroni/BH then changed nothing.
Statistical significance was never the binding constraint; effect size was.

**Attempt 2** — after adding a missing-rate rule to catch feed outages, I dropped
the batch threshold to 0.05 to make it more sensitive. Result: **7 of 8 healthy
windows alarmed**, including a scenario where I had changed nothing at all.

**Attempt 3** — measure the null first (`make calibrate`). Eight consecutive
fault-free later windows:

| | healthy max | threshold set | injected faults produce |
|---|---|---|---|
| share of features flagged | 0.125 | **0.20** | 0.25–0.30 |
| missing-rate jump | 0.190 | **0.35** | 1.00 |
| prediction PSI | 0.021 | **0.25** | 0.49 |

0/8 false alarms, and every threshold sits in a measured gap rather than on a
round number I liked the look of.

**The subtlety that caused attempt 2**: my first healthy control split one period
at *random*, which holds the calendar constant. Production traffic is never a
random sample of the past — it is always *later*, and things like D-column
missing rates move with time on their own. Calibrating against the random-split
null sets thresholds far too tight. **Your healthy baseline has to be a future
window, or you calibrate against a null that never occurs.**

The new-customer-segment miss is a consequence of this, and I'm keeping it:
catching it needs the 0.05 threshold that alarmed on 7 of 8 clean windows. That's
the precision/recall trade-off of monitoring, measured rather than asserted.

---

## A bug this project found in the last one

`day_index` — the absolute day number of a transaction — scored **PSI 12.4 in
every scenario including the healthy control**, permanently consuming an alert
slot. It drifts by construction: every batch is later than the baseline, forever.

Chasing that turned up something worse. In the shipped fraud model, `day_index`
is the **8th most important feature of 443** by gain (2.29%). The model is using
*when* a transaction happened as a predictor — and every production day lies
outside the training range, so that feature can only mislead. It plausibly
contributes to the 0.137 AUC decay above.

Monitoring now excludes monotonic-by-construction features from the alert budget
([`baseline.py`](src/pipeline/baseline.py)), but the real fix belongs upstream:
drop `day_index` from the model. Filed against the fraud repo rather than patched
here, because the model is that repo's artifact.

---

## Architecture

```
FastAPI (/predict, /health, /metrics)  ->  JSONL request log
        |                                        |
   model.pkl (443 features)              drift job: KS + PSI + null-rate
        |                                        |
   MLflow registry  <- CI gate            Streamlit dashboard
```

- **Serving** — [`serving.py`](src/pipeline/serving.py). Logs the features the
  model *actually saw*, not a re-derivation, so monitoring can catch the serving
  and training pipelines disagreeing.
- **Drift** — [`drift.py`](src/pipeline/drift.py). Per-feature KS (significance)
  **and** PSI (effect size), plus a separate missing-rate rule, because an
  all-null column has no distribution left to test and scores PSI 0.
- **Registry** — MLflow, logging the honest CV number (0.8513), the flattering
  leaky one (0.9557) and the true leaderboard score (0.9086) side by side.
- **CI gate** — [`ci.yml`](.github/workflows/ci.yml) fails the build if tests
  fail *or* the model card comes back non-deployable. It gates; it doesn't just
  run tests and print a badge.
- **Container** — non-root, healthchecked.

## Running it

```bash
make setup && make test
```

```bash
make calibrate && make simulate && make validate
```

Needs the IEEE-CIS data (`IEEE_DATA=/path/to/raw`, see the
[fraud repo](https://github.com/aghasalim/ieee-fraud-ml)). `make serve` and
`make dashboard` need only the committed `model.pkl`.

## What I'd do next

In priority order, which is not the order that looks most impressive:

1. **Get labels into the loop.** Everything above shows feature monitoring cannot
   see the failure that actually occurred. Fraud labels arrive on a chargeback
   delay of weeks — late, but not useless, and a delayed AUC is worth more than
   a real-time PSI that anti-correlates with it.
2. **Drop `day_index`** and re-measure the decay.
3. **Automated retraining** on a schedule rather than on a drift trigger, since
   the drift trigger demonstrably does not fire when retraining is needed.

## License

MIT. The model is reused from [ieee-fraud-ml](https://github.com/aghasalim/ieee-fraud-ml);
competition data is not redistributed.
