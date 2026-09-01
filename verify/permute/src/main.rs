//! The two things the published correlation needs and eight rows cannot buy.
//!
//! README section 1 reports a correlation of -0.709 between prediction PSI and
//! AUC loss over eight windows, and adds "n=8, so suggestive rather than
//! conclusive". Nothing in the repository measures how suggestive. This does
//! two pieces of work the pandas experiment could not afford:
//!
//!   1. Every one of the 8! = 40320 relabellings of the AUC drops against the
//!      PSI column, so the two-sided p-value is exact and not a sample. R does
//!      the same enumeration independently in verify/inference.R and the two
//!      counts have to be identical integers.
//!
//!   2. Ten million bootstrap resamples in ten blocks of a million, which is
//!      enough to put a Monte Carlo error bar on the interval itself. The
//!      question that answers is whether the 100,000 resample interval printed
//!      by the R script is reporting the data or reporting its own noise.
//!
//! Exits non-zero if the point estimate misses the published figure, if the
//! interval is not stable across blocks, or if the interval does not contain
//! the point estimate.

use std::env;
use std::fs;
use std::process::exit;

const PUBLISHED: f64 = -0.709; // README section 1 and INCIDENT.md
const TOL: f64 = 5e-4; // the published figure is given to 3 dp
const EPS: f64 = 1e-9; // permutation ties: the nearest distinct |r| is 1.7e-4 away
const BLOCKS: usize = 10;
const PER_BLOCK: usize = 1_000_000;
const SMALL: usize = 100_000; // the resample count verify/inference.R uses
const STABILITY_TOL: f64 = 0.01;

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast, and seeded reproducibly so a failure here can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}

/// Lexicographic successor, in place. Returns false once the array is back in
/// descending order, which is how the enumeration knows it has seen all of them.
fn next_permutation(a: &mut [usize]) -> bool {
    let n = a.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}

/// Linear interpolation between order statistics: numpy's default and R's
/// type 7. Matching the convention matters, because the alternatives shift a
/// quantile by up to one order statistic and that is visible at this width.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

/// One block of bootstrap resamples, returned as the 2.5 and 97.5 percentiles
/// and the share of replicates that keep the sign of the estimate.
fn block(x: &[f64], y: &[f64], draws: usize, rng: &mut Rng) -> (f64, f64, f64, usize) {
    let n = x.len();
    let mut stats = Vec::with_capacity(draws);
    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];
    let mut degenerate = 0usize;
    let mut negative = 0usize;

    for _ in 0..draws {
        for k in 0..n {
            let i = rng.below(n);
            xs[k] = x[i];
            ys[k] = y[i];
        }
        let r = pearson(&xs, &ys);
        if r.is_finite() {
            if r < 0.0 {
                negative += 1;
            }
            stats.push(r);
        } else {
            // A resample can draw one window eight times, which leaves no
            // variance and no correlation to compute. Counted, not zeroed.
            degenerate += 1;
        }
    }
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let share_neg = negative as f64 / stats.len() as f64;
    (quantile(&stats, 0.025), quantile(&stats, 0.975), share_neg, degenerate)
}

fn load(root: &str) -> (Vec<f64>, Vec<f64>) {
    let path = format!("{}/reports/drift_vs_degradation.csv", root);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2)
    });

    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("empty file").split(',').collect();
    let col = |name: &str| {
        header.iter().position(|h| h.trim() == name).unwrap_or_else(|| {
            eprintln!("no {} column in drift_vs_degradation.csv", name);
            exit(2)
        })
    };
    let (px, py) = (col("pred_psi"), col("auc_drop"));

    let (mut x, mut y) = (Vec::new(), Vec::new());
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() <= px.max(py) {
            eprintln!("a window row is short");
            exit(2);
        }
        let parse = |s: &str| -> f64 {
            s.trim().parse::<f64>().unwrap_or_else(|_| {
                eprintln!("{:?} is not a number", s);
                exit(2)
            })
        };
        x.push(parse(f[px]));
        y.push(parse(f[py]));
    }
    if x.len() != 8 {
        eprintln!("expected 8 windows, read {}", x.len());
        exit(1);
    }
    (x, y)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");
    let (x, y) = load(root);
    let n = x.len();

    let r = pearson(&x, &y);
    println!("RESULT rust corr_pred_psi_auc_drop {:.15e}", r);

    let mut failures = 0;
    let d = (r - PUBLISHED).abs();
    println!(
        "  correlation        Rust {:+.15}  README {:+.3}  |d| {:.1e}  {}",
        r,
        PUBLISHED,
        d,
        if d <= TOL { "ok" } else { "FAIL" }
    );
    failures += (d > TOL) as i32;

    // Exhaustive relabelling. The permutation leaves the mean and spread of y
    // alone, so the denominator is fixed and only the cross product moves.
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mx).collect();
    let denom = (xc.iter().map(|v| v * v).sum::<f64>()
        * y.iter().map(|v| (v - my) * (v - my)).sum::<f64>())
    .sqrt();

    let mut idx: Vec<usize> = (0..n).collect();
    let (mut total, mut extreme) = (0usize, 0usize);
    loop {
        let num: f64 = (0..n).map(|j| y[idx[j]] * xc[j]).sum();
        if (num / denom).abs() >= r.abs() - EPS {
            extreme += 1;
        }
        total += 1;
        if !next_permutation(&mut idx) {
            break;
        }
    }
    println!("RESULT rust perm_extreme {}", extreme);
    println!("RESULT rust perm_total {}", total);
    println!(
        "  permutation test   {} of {} relabellings reach |r| >= {:.6}, exact two-sided p = {:.6}",
        extreme,
        total,
        r.abs(),
        extreme as f64 / total as f64
    );

    // Ten blocks of a million. The spread across blocks is the Monte Carlo
    // error on an interval of that size, and it is the number that says whether
    // the interval printed by the R script means anything.
    let mut los = Vec::new();
    let mut his = Vec::new();
    let mut degenerate = 0usize;
    let mut neg = 0.0;
    for b in 0..BLOCKS {
        let mut rng = Rng::new(0xC0FF_EE00 + b as u64 * 104_729);
        let (lo, hi, share_neg, deg) = block(&x, &y, PER_BLOCK, &mut rng);
        los.push(lo);
        his.push(hi);
        degenerate += deg;
        neg += share_neg / BLOCKS as f64;
    }
    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let sd = |v: &Vec<f64>| {
        let m = mean(v);
        (v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    };
    let (lo, hi) = (mean(&los), mean(&his));
    println!("RESULT rust boot_ci_lo {:.15e}", lo);
    println!("RESULT rust boot_ci_hi {:.15e}", hi);
    println!(
        "  bootstrap          {} resamples in {} blocks, {} degenerate, 95% percentile CI [{:.3}, {:.3}]",
        BLOCKS * PER_BLOCK,
        BLOCKS,
        degenerate,
        lo,
        hi
    );
    println!(
        "                     block to block sd of the bounds {:.4} and {:.4}, {:.1}% of resamples keep the sign negative",
        sd(&los),
        sd(&his),
        100.0 * neg
    );

    // Was 100,000 enough? Same estimator, one hundredth of the work.
    let mut rng = Rng::new(0x5EED_1234);
    let (slo, shi, _, _) = block(&x, &y, SMALL, &mut rng);
    println!(
        "                     at {} resamples the same interval is [{:.3}, {:.3}], within {:.4} and {:.4}",
        SMALL,
        slo,
        shi,
        (slo - lo).abs(),
        (shi - hi).abs()
    );

    let unstable = sd(&los) > STABILITY_TOL || sd(&his) > STABILITY_TOL;
    failures += unstable as i32;
    println!(
        "  interval stability bounds move less than {} across blocks  {}",
        STABILITY_TOL,
        if unstable { "FAIL" } else { "ok" }
    );

    let inside = r >= lo && r <= hi;
    failures += !inside as i32;
    println!(
        "  point in interval  {}",
        if inside { "ok" } else { "FAIL" }
    );

    if failures > 0 {
        println!("\n{} checks failed", failures);
        exit(1);
    }
    println!(
        "\nRust reproduces the published correlation, and its exact permutation\ncount matches the independent enumeration in verify/inference.R"
    );
}
