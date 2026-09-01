/* Recompute the correlation the README's third finding rests on, in C.
 *
 * README section 1: "Prediction PSI correlates -0.709 with that loss". That
 * number is one line of numpy in experiments/detector_validation.py, printed
 * once and copied into two markdown files. Nothing recomputed it, and the
 * figure it argues from is drawn from the same array, so a mistake in the
 * correlation would appear in the prose and in the plot identically.
 *
 * This reads reports/drift_vs_degradation.csv, resolves the columns by name so
 * a column inserted upstream cannot silently shift what is read, and computes
 * the Pearson correlation in two passes. It also checks the two other numbers
 * the same file carries: the range of the AUC loss, and the fact that
 * auc + auc_drop is one baseline figure repeated in every row.
 *
 * Exits non-zero on the first disagreement past the tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXROW 4096
#define LINE 4096

/* From README section 1 and INCIDENT.md. Written out so that a change in the
 * data that moves them shows up here rather than in a reader's memory. */
#define PUB_CORR (-0.709)
#define PUB_DROP_MIN (0.060)
#define PUB_DROP_MAX (0.137)
#define PUB_TOL (5e-4)          /* the published figures are given to 3 dp */
#define BASE_SPREAD_TOL (2e-4)  /* auc and auc_drop are each rounded to 4 dp */

static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    strncpy(buf, header, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';

    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[256];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p)
            return NULL;
        p++;
        col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out)
        n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl)
        *nl = '\0';
    return out;
}

static double number(const char *line, int index, int *ok)
{
    const char *s = field(line, index);
    if (!s || *s == '\0') { *ok = 0; return 0.0; }
    return atof(s);
}

/* Two passes rather than the sum of squares shortcut. The shortcut cancels
 * large numbers against each other and loses digits exactly where the
 * comparison against the other implementations is being made. */
static double pearson(const double *x, const double *y, int n)
{
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;

    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int i = 0; i < n; i++) {
        const double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx == 0.0 || syy == 0.0)
        return NAN;
    return sxy / sqrt(sxx * syy);
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE];

    snprintf(path, sizeof path, "%s/reports/drift_vs_degradation.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }

    const int psi_col = column_of(header, "pred_psi");
    const int drop_col = column_of(header, "auc_drop");
    const int auc_col = column_of(header, "auc");
    if (psi_col < 0 || drop_col < 0 || auc_col < 0) {
        fprintf(stderr, "drift_vs_degradation.csv is missing a needed column\n");
        fclose(f);
        return 2;
    }

    double psi[MAXROW], drop[MAXROW], base[MAXROW];
    int n = 0;
    while (fgets(line, sizeof line, f) && n < MAXROW) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;
        /* field() hands back one static buffer, so each value is consumed
         * before the next call rather than held as a pointer. */
        int ok = 1;
        psi[n] = number(line, psi_col, &ok);
        drop[n] = number(line, drop_col, &ok);
        base[n] = number(line, auc_col, &ok) + drop[n];
        if (!ok) {
            fprintf(stderr, "row %d is short\n", n + 2);
            fclose(f);
            return 2;
        }
        if (!isfinite(psi[n]) || !isfinite(drop[n]) || !isfinite(base[n])) {
            fprintf(stderr, "row %d is not finite\n", n + 2);
            fclose(f);
            return 2;
        }
        n++;
    }
    fclose(f);

    if (n != 8) {
        fprintf(stderr, "expected 8 windows, read %d\n", n);
        return 1;
    }

    const double r = pearson(psi, drop, n);
    printf("RESULT c corr_pred_psi_auc_drop %.15e\n", r);

    int failures = 0;

    const double dcorr = fabs(r - PUB_CORR);
    printf("  correlation           C %+.15f  README %+.3f  |d| %.1e  %s\n",
           r, PUB_CORR, dcorr, dcorr <= PUB_TOL ? "ok" : "FAIL");
    failures += dcorr > PUB_TOL;

    double lo = drop[0], hi = drop[0];
    for (int i = 1; i < n; i++) {
        if (drop[i] < lo) lo = drop[i];
        if (drop[i] > hi) hi = drop[i];
    }
    const int range_bad = fabs(lo - PUB_DROP_MIN) > PUB_TOL
                       || fabs(hi - PUB_DROP_MAX) > PUB_TOL;
    printf("  auc loss range        C %.4f to %.4f     README %.3f to %.3f  %s\n",
           lo, hi, PUB_DROP_MIN, PUB_DROP_MAX, range_bad ? "FAIL" : "ok");
    failures += range_bad;

    double blo = base[0], bhi = base[0], bsum = 0.0;
    for (int i = 0; i < n; i++) {
        if (base[i] < blo) blo = base[i];
        if (base[i] > bhi) bhi = base[i];
        bsum += base[i];
    }
    const double spread = bhi - blo;
    printf("  implied baseline auc  C %.6f  spread over %d windows %.1e  %s\n",
           bsum / n, n, spread, spread <= BASE_SPREAD_TOL ? "ok" : "FAIL");
    failures += spread > BASE_SPREAD_TOL;

    if (failures) {
        printf("\n%d of 3 checks failed\n", failures);
        return 1;
    }
    printf("\nC reproduces the published correlation and AUC range from the "
           "window file\n");
    return 0;
}
