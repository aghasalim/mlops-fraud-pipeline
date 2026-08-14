# Production ML pipeline — with the monitoring actually tested

[![ci](https://github.com/aghasalim/mlops-fraud-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/mlops-fraud-pipeline/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The fraud model from [ieee-fraud-ml](https://github.com/aghasalim/ieee-fraud-ml)
served behind FastAPI, versioned, gated by CI, and monitored — built by a
third-year Applied Computer Science (AI) student.

A monitoring dashboard nobody has broken on purpose is a decorative chart. So
the deliverable here is **[INCIDENT.md](INCIDENT.md)**: what I broke, what the
monitor caught, and what it missed until I fixed it.

---

## The short version

**3 of 3 injected failures caught, 0 false alarms on 2 controls.** That is the
result the brief asks for, and it is the least interesting thing I found.

The three findings I would actually want to be asked about:

**1. My detector was broken before I injected anything.** Three bugs, all found
by testing rather than reading code. The worst: I simulated the identity
provider going down — every `id_*` column arriving null — and the monitor
reported **healthy**. A KS test drops non-finite values, so a 100%-null column
has nothing left to compare and scores PSI 0. The single most conspicuous
failure in production produced a *cleaner* report than normal traffic.

**2. The multiple-testing correction I built isn't what fixes false alarms.**
On two random halves of identical data, KS testing alone flags **3.35 of 40
features** — the 5% you asked for, arriving as noise, forever. I added
Benjamini-Hochberg for it. But the thing that actually removes them is the
**PSI effect-size gate**: requiring a shift to be *large*, not merely
detectable. With that in place BH and Bonferroni have nothing left to do. I
built the correction believing it was the answer, and it wasn't.

**3. Prediction drift points the wrong way.** Across eight windows of real
traffic the model loses **0.060–0.137 AUC** with nothing broken — just time
passing. Prediction PSI correlates **−0.709** with that loss: the output
distribution looks most stable exactly where the model is doing worst. n=8, so
suggestive rather than conclusive, but the direction alone kills "predictions
look normal, so we're fine."

---

## What the monitor did

| scenario | caught? | how |
|---|---|---|
| healthy (control) | **no** ✓ | correct — 1/40 features, prediction PSI 0.020 |
| currency units bug | **yes** | prediction PSI **0.494** |
| new customer segment | **yes** | 3/40 features, `card1_freq` PSI 0.564 |
| identity feed outage | **yes** | `id_31` missing rate **+100%** |
| label shift only | **no** ✓ | correct by construction |

The last row is a negative control and is *supposed* to be missed: it changes
only which transactions are fraudulent, leaving every input untouched. **No
input-distribution monitor can see that**, and this one doesn't. It is in the
list because a scenario set where everything gets caught tells you nothing about
where the system is blind.

The outage is caught *only* by the missing-rate rule — both the feature-share
and prediction-PSI signals sit below threshold. Without that fix it sails
through.

---

## Running it

```bash
make setup && make test
```

25 tests. The drift tests encode bugs that actually shipped, so they fail if
those regress.

```bash
make serve
```

Then `curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"TransactionAmt": 120.0}'`

Reproducing the experiments needs the IEEE-CIS data (see
[ieee-fraud-ml](https://github.com/aghasalim/ieee-fraud-ml) for the Kaggle
fetch); point `IEEE_DATA` at it:

```bash
make validate && make simulate && make dashboard
```

---

## How it fits together

| piece | choice | why |
|---|---|---|
| serving | FastAPI + Pydantic | request validation is a monitoring surface: a negative amount is a 422, not a prediction |
| container | Docker, non-root, healthcheck | a process reachable from the network is the last place to run privileged |
| registry | MLflow (SQLite backend) | the file store is in maintenance mode and now raises outright |
| gate | `registry.py` exits non-zero | CI depends on the gate as a *job dependency*, not a check mark someone is trusted to read |
| drift | KS + PSI + missing-rate | significance, effect size, and the thing distribution tests are blind to |
| monitored set | top 40 by gain importance | monitoring all 443 adds noise and alert slots, not coverage |

**The deploy gate blocks, and there are tests proving it.** A gate that has only
ever returned `true` is indistinguishable from no gate — so `test_gate.py`
asserts that a model with AUC 0.60, one trained on 1,000 rows, one whose feature
count doubled, and one that fails to load are each rejected.

---

## What this cannot do

Stated because a monitoring write-up without a limits section is marketing:

- **No labels, so no direct performance monitoring.** The most important signal
  is missing; everything here is a proxy for it, and finding 3 is evidence the
  proxy is weak.
- **Concept drift is invisible** — demonstrated by the negative control, not
  assumed.
- **Batch, not streaming.** Detection latency is one batch.
- **Training/serving skew is unguarded.** `featurize.py` re-implements
  transformations living in another repo; a shared library or feature store is
  the real fix and is not here.

Full detail, including the thresholds I set wrong in both directions before
calibrating them, in **[INCIDENT.md](INCIDENT.md)**.

## License

MIT — see [LICENSE](LICENSE).
