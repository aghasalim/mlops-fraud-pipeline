#!/usr/bin/env bash
# Recompute the numbers this repository publishes, in every language here, and
# require the answers to agree.
#
# Everything in README.md and INCIDENT.md comes out of two pandas scripts under
# experiments/. The figures are drawn from the same frames, the tables are typed
# from the same printouts, and the raw IEEE-CIS data those scripts read is not
# in this repository, so a rerun is not a check. If the aggregation in
# detector_validation.py were wrong, nothing downstream would ever notice,
# because everything downstream reads its output.
#
# So the published numbers are recomputed from the committed reports/ files by
# seven implementations that share no code, and the ones that compute the same
# quantity are required to agree at the end. An error would have to be made
# identically in SQL, C, Go, R, Rust, JavaScript and Java to survive.
#
# Each step is skipped with a clear message if its toolchain is absent, so this
# runs on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

log="$(mktemp "${TMPDIR:-/tmp}/verify.XXXXXX")"
trap 'rm -f "$log"' EXIT

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@" 2>&1 | tee -a "$log"; then
        pass=$((pass + 1)); printf -- '--> %s passed\n' "$name"
    else
        fail=$((fail + 1)); printf -- '--> %s FAILED\n' "$name"
    fi
}

# The SQL has no assertion of its own, so its verdict column is read here.
check_sql () {
    local out status
    out=$(sqlite3 -init verify/tables.sql :memory: "" 2>&1)
    status=$?
    printf '%s\n' "$out"
    [ "$status" -eq 0 ] || return 1
    if printf '%s' "$out" | grep -q '|FAIL|'; then
        return 1
    fi
    printf '%s' "$out" | grep -qc '|ok|' >/dev/null || return 1
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/corrcheck" verify/correlation.c -lm || return 1
    "${TMPDIR:-/tmp}/corrcheck" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/permute && cargo run --release --quiet -- "$root" ); }

run "SQL, published tables"        sqlite3 check_sql
run "C, correlation kernel"        cc      check_c
run "Go, file validation"          go      check_go
run "R, statistical inference"     Rscript Rscript verify/inference.R "$root"
run "Rust, exact permutation"      cargo   check_rust
run "JavaScript, document claims"  node    node verify/docs_claims.mjs "$root"
run "Java, alert rule"             java    java verify/AlertRule.java "$root"

# The differential step. Every implementation prints its answers as
# "RESULT <language> <metric> <value>"; anything computed by more than one has
# to land in the same place. The tolerances say why: the correlation is
# deterministic and agrees to floating point, the bootstrap bounds are Monte
# Carlo estimates from two different generators and can only be required to
# agree within their own sampling error.
printf '\n=== cross-language agreement ===\n'
if [ "$pass" -lt 2 ]; then
    printf 'skipped: fewer than two implementations ran\n'
    skip=$((skip + 1))
else
    if awk '
        /^RESULT / { m = $3; v = $4 + 0
                     n[m]++
                     if (n[m] == 1 || v < lo[m]) lo[m] = v
                     if (n[m] == 1 || v > hi[m]) hi[m] = v
                     who[m] = who[m] " " $2 }
        END {
            tol["corr_pred_psi_auc_drop"] = 1e-12
            tol["perm_extreme"] = 0
            tol["perm_total"] = 0
            tol["boot_ci_lo"] = 0.02      # Rust measures the block to block sd at 0.0001
            tol["boot_ci_hi"] = 0.02      # and 0.0014 on a tenth of its draws
            bad = 0
            for (m in n) {
                if (n[m] < 2) { printf "  %-26s %d implementation, nothing to compare\n", m, n[m]; continue }
                spread = hi[m] - lo[m]
                t = (m in tol) ? tol[m] : 0
                ok = (spread <= t) ? "ok" : "FAIL"
                if (spread > t) bad++
                printf "  %-26s %d implementations%s agree to %.1e (allowed %.1e)  %s\n",
                       m, n[m], who[m], spread, t, ok
            }
            exit bad > 0
        }' "$log"
    then
        pass=$((pass + 1)); printf -- '--> cross-language agreement passed\n'
    else
        fail=$((fail + 1)); printf -- '--> cross-language agreement FAILED\n'
    fi
fi

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
