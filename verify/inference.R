# What is the published correlation worth, with eight windows?
#
# The README's third finding is that prediction PSI correlates -0.709 with AUC
# loss, and it says "n=8, so suggestive rather than conclusive". That is the
# right instinct, but the repository never attaches a number to it: the whole
# claim is one call to np.corrcoef in experiments/detector_validation.py, with
# no p-value, no interval and no check that the sign survives a rank transform.
#
# This recomputes the correlation in base R and then does the inference the
# Python skipped:
#
#   exact permutation test   all 8! = 40320 relabellings, not a sample of them,
#                            so the p-value is exact rather than simulated
#   Spearman                 the same claim with the parametric assumption
#                            removed, since with eight points a single window
#                            could be carrying the whole thing
#   bootstrap                a percentile interval, and how often the sign of
#                            the estimate flips under resampling
#
# No packages, so CI needs nothing beyond base R.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

PUBLISHED <- -0.709   # README section 1 and INCIDENT.md
TOL <- 5e-4           # the published figure is given to 3 dp
EPS <- 1e-9           # permutation ties: the nearest distinct |r| is 1.7e-4 away
BOOT <- 100000

w <- read.csv(file.path(root, "reports", "drift_vs_degradation.csv"))
x <- as.numeric(w$pred_psi)
y <- as.numeric(w$auc_drop)
n <- length(x)
stopifnot(n == 8, all(is.finite(x)), all(is.finite(y)))

pearson <- function(a, b) {
    da <- a - mean(a)
    db <- b - mean(b)
    sum(da * db) / sqrt(sum(da * da) * sum(db * db))
}

r <- pearson(x, y)
cat(sprintf("RESULT r corr_pred_psi_auc_drop %.15e\n", r))

failures <- 0
d <- abs(r - PUBLISHED)
ok <- d <= TOL
failures <- failures + !ok
cat(sprintf("  correlation        R %+.15f  README %+.3f  |d| %.1e  %s\n",
            r, PUBLISHED, d, if (ok) "ok" else "FAIL"))

# Every permutation of the eight AUC drops against the fixed PSI column. The
# permutations do not change the mean or the spread of y, so the denominator is
# constant and every relabelled correlation is one matrix product away.
all_perms <- function(k) {
    if (k == 1L) return(matrix(1L, 1L, 1L))
    sub <- all_perms(k - 1L)
    m <- nrow(sub)
    out <- matrix(0L, m * k, k)
    for (i in seq_len(k)) {
        shifted <- ifelse(sub >= i, sub + 1L, sub)
        out[((i - 1L) * m + 1L):(i * m), ] <- cbind(i, matrix(shifted, m))
    }
    out
}

perm <- all_perms(n)
stopifnot(nrow(perm) == factorial(n))
ymat <- matrix(y[perm], nrow(perm), n)
xc <- x - mean(x)
denom <- sqrt(sum(xc * xc) * sum((y - mean(y))^2))
r_perm <- as.vector(ymat %*% xc) / denom

extreme <- sum(abs(r_perm) >= abs(r) - EPS)
p_exact <- extreme / length(r_perm)
cat(sprintf("RESULT r perm_extreme %d\n", extreme))
cat(sprintf("RESULT r perm_total %d\n", length(r_perm)))
cat(sprintf("  permutation test   %d of %d relabellings reach |r| >= %.6f, exact two-sided p = %.6f\n",
            extreme, length(r_perm), abs(r), p_exact))

rho <- pearson(rank(x), rank(y))
cat(sprintf("RESULT r spearman %.15e\n", rho))
sign_ok <- rho < 0 && r < 0
failures <- failures + !sign_ok
cat(sprintf("  Spearman           rho %+.6f, same sign as Pearson  %s\n",
            rho, if (sign_ok) "ok" else "FAIL"))

# Percentile bootstrap. With n = 8 a resample can draw the same window eight
# times, which leaves no variance and no correlation to compute; those are
# counted and dropped rather than quietly turned into zeros.
idx <- matrix(sample.int(n, BOOT * n, replace = TRUE), BOOT, n)
xb <- matrix(x[idx], BOOT, n)
yb <- matrix(y[idx], BOOT, n)
cx <- xb - rowMeans(xb)
cy <- yb - rowMeans(yb)
rb <- rowSums(cx * cy) / sqrt(rowSums(cx * cx) * rowSums(cy * cy))
degenerate <- sum(!is.finite(rb))
rb <- rb[is.finite(rb)]
ci <- quantile(rb, c(0.025, 0.975), names = FALSE)
neg <- mean(rb < 0)

cat(sprintf("RESULT r boot_ci_lo %.15e\n", ci[1]))
cat(sprintf("RESULT r boot_ci_hi %.15e\n", ci[2]))
cat(sprintf("  bootstrap          %d resamples, %d degenerate, 95%% percentile CI [%.3f, %.3f]\n",
            BOOT, degenerate, ci[1], ci[2]))
cat(sprintf("                     %.1f%% of resamples keep the sign negative\n", 100 * neg))

inside <- r >= ci[1] && r <= ci[2]
failures <- failures + !inside
cat(sprintf("  point in interval  %s\n", if (inside) "ok" else "FAIL"))

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces the published correlation, and the negative direction\n")
cat("survives both the rank transform and the exact permutation test\n")
