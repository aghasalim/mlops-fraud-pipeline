// Reimplement the batch alert rule and require it to reproduce every published
// verdict.
//
// src/pipeline/drift.py decides that a batch has drifted when any of three
// things is true: a monitored feature's missing rate moved by NULL_JUMP or
// more, the share of flagged features reached BATCH_DRIFT_SHARE, or the
// prediction PSI reached PSI_MAJOR. Those verdicts are published as the
// would_alert column of reports/calibration.csv and the caught column of
// reports/simulated_failures.csv, and the README's headline result is a count
// of them.
//
// The rule itself is nine lines of Python that nothing tests against the
// published output. A wrong comparison would be quiet: two of the eight windows
// sit exactly on the 0.05 boundary, so >= and > give different answers on real
// rows of the real file.
//
// The thresholds are written out below rather than read blindly, and then
// checked against src/pipeline/config.py, so a threshold that moves in the
// source without moving the published tables is a failure here rather than a
// silent re-verification of a stale number.
//
// Run: java verify/AlertRule.java <repo root>

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AlertRule {

    // src/pipeline/config.py, checked against the source below.
    static final double BATCH_DRIFT_SHARE = 0.05;
    static final double PSI_MAJOR = 0.25;
    static final double NULL_JUMP = 0.35;
    static final int N_MONITORED = 40;

    static final double PUBLISHED_CORR = -0.709;   // README section 1
    static final double CORR_TOL = 5e-4;

    static int failures = 0;

    static void check(String name, boolean ok, String detail) {
        System.out.printf("  %-28s %s  %s%n", name, ok ? "ok  " : "FAIL", detail);
        if (!ok) {
            failures++;
        }
    }

    static List<Map<String, String>> readCsv(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path);
        String[] cols = lines.get(0).split(",", -1);
        List<Map<String, String>> rows = new ArrayList<>();
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) {
                continue;
            }
            String[] cells = line.split(",", -1);
            if (cells.length != cols.length) {
                throw new IOException(path.getFileName() + " has a ragged row: " + line);
            }
            Map<String, String> row = new LinkedHashMap<>();
            for (int i = 0; i < cols.length; i++) {
                row.put(cols[i].trim(), cells[i].trim());
            }
            rows.add(row);
        }
        return rows;
    }

    /** The default of a config constant, whether it is a literal or a getenv fallback. */
    static double configValue(String source, String name) {
        Matcher env = Pattern.compile(
                name + "\\s*=\\s*(?:float|int)\\(os\\.getenv\\(\"[^\"]*\",\\s*\"([^\"]*)\"\\)\\)")
                .matcher(source);
        if (env.find()) {
            return Double.parseDouble(env.group(1));
        }
        Matcher literal = Pattern.compile(name + "\\s*=\\s*(-?[0-9.]+)\\s*$", Pattern.MULTILINE)
                .matcher(source);
        if (literal.find()) {
            return Double.parseDouble(literal.group(1));
        }
        return Double.NaN;
    }

    static double num(Map<String, String> row, String col) {
        return Double.parseDouble(row.get(col));
    }

    static boolean bool(Map<String, String> row, String col) {
        return "True".equals(row.get(col));
    }

    /** drift.py: any one of the three reasons is enough. */
    static boolean alerts(double share, int nullShifts, double predPsi, double shareThreshold) {
        return nullShifts > 0 || share >= shareThreshold || predPsi >= PSI_MAJOR;
    }

    static double pearson(double[] x, double[] y) {
        int n = x.length;
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) {
            mx += x[i];
            my += y[i];
        }
        mx /= n;
        my /= n;
        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - mx, dy = y[i] - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        return sxy / Math.sqrt(sxx * syy);
    }

    public static void main(String[] args) throws IOException {
        Path root = Path.of(args.length > 0 ? args[0] : ".");

        String config = Files.readString(root.resolve("src/pipeline/config.py"));
        boolean sameThresholds =
                configValue(config, "BATCH_DRIFT_SHARE") == BATCH_DRIFT_SHARE
                && configValue(config, "PSI_MAJOR") == PSI_MAJOR
                && configValue(config, "NULL_JUMP") == NULL_JUMP
                && configValue(config, "N_MONITORED") == N_MONITORED;
        check("thresholds match config.py", sameThresholds,
                String.format("share %.2f, prediction PSI %.2f, null jump %.2f, %d features",
                        BATCH_DRIFT_SHARE, PSI_MAJOR, NULL_JUMP, N_MONITORED));

        // 1. Every window verdict in reports/calibration.csv.
        List<Map<String, String>> calib = readCsv(root.resolve("reports/calibration.csv"));
        int wrong = 0, boundary = 0;
        for (Map<String, String> row : calib) {
            double share = num(row, "share_flagged");
            int nulls = (int) num(row, "n_null_shift");
            boolean got = alerts(share, nulls, num(row, "pred_psi"), BATCH_DRIFT_SHARE);
            if (got != bool(row, "would_alert")) {
                wrong++;
                System.out.printf("    window %s: rule says %b, file says %s%n",
                        row.get("window"), got, row.get("would_alert"));
            }
            if (share == BATCH_DRIFT_SHARE) {
                boundary++;
            }
        }
        check("calibration would_alert", wrong == 0 && calib.size() == 8,
                String.format("%d of %d windows reproduced, %d sit exactly on the %.2f boundary",
                        calib.size() - wrong, calib.size(), boundary, BATCH_DRIFT_SHARE));

        // 2. The missing-rate column has to be consistent with its own threshold.
        int inconsistent = 0;
        for (Map<String, String> row : calib) {
            boolean counted = num(row, "n_null_shift") > 0;
            boolean implied = num(row, "max_null_delta") >= NULL_JUMP;
            if (counted != implied) {
                inconsistent++;
            }
        }
        double worst = calib.stream().mapToDouble(r -> num(r, "max_null_delta")).max().orElse(-1);
        check("null shift self-consistent", inconsistent == 0,
                String.format("largest missing-rate move %.3f against a %.2f threshold", worst, NULL_JUMP));

        // 3. The injected scenarios. The distribution rules alone cannot explain
        //    the identity feed outage, and the README says so: that row is caught
        //    by the missing-rate rule, whose input is not in this file. Exactly
        //    one row must disagree, and it must be that one.
        List<Map<String, String>> scen = readCsv(root.resolve("reports/simulated_failures.csv"));
        List<String> unexplained = new ArrayList<>();
        for (Map<String, String> row : scen) {
            boolean got = alerts(num(row, "share"), 0, num(row, "pred_psi"), BATCH_DRIFT_SHARE);
            if (got != bool(row, "caught")) {
                unexplained.add(row.get("scenario"));
            }
        }
        check("scenario verdicts",
                unexplained.size() == 1 && unexplained.get(0).equals("identity feed outage"),
                String.format("%d of %d explained by share and prediction PSI alone, "
                        + "the exception is %s", scen.size() - unexplained.size(), scen.size(),
                        unexplained.isEmpty() ? "none" : unexplained));

        // 4. README section 2: "once the threshold passes 7.5% the fault stops
        //    alerting too". Sweep the batch share threshold and find where.
        Map<String, String> segment = scen.stream()
                .filter(r -> r.get("scenario").equals("new customer segment"))
                .findFirst().orElseThrow();
        double lastAlerting = Double.NaN, firstSilent = Double.NaN;
        for (int step = 1; step <= 100; step++) {
            double t = step / 200.0;    // 0.005 to 0.5
            boolean fires = alerts(num(segment, "share"), 0, num(segment, "pred_psi"), t);
            if (fires) {
                lastAlerting = t;
            } else if (Double.isNaN(firstSilent)) {
                firstSilent = t;
            }
        }
        check("threshold sweep crossing", Math.abs(lastAlerting - 0.075) < 1e-9,
                String.format("the new segment fault still alerts at %.3f and is silent at %.3f, "
                        + "README says 7.5%%", lastAlerting, firstSilent));

        // 5. The headline correlation, for the cross-language agreement pool.
        List<Map<String, String>> win = readCsv(root.resolve("reports/drift_vs_degradation.csv"));
        double[] x = win.stream().mapToDouble(r -> num(r, "pred_psi")).toArray();
        double[] y = win.stream().mapToDouble(r -> num(r, "auc_drop")).toArray();
        double r = pearson(x, y);
        System.out.printf("RESULT java corr_pred_psi_auc_drop %.15e%n", r);
        check("correlation", Math.abs(r - PUBLISHED_CORR) <= CORR_TOL,
                String.format("Java %+.15f, README %+.3f", r, PUBLISHED_CORR));

        if (failures > 0) {
            System.out.printf("%n%d checks failed%n", failures);
            System.exit(1);
        }
        System.out.println();
        System.out.println("the alert rule reimplemented from drift.py reproduces every published verdict");
    }
}
