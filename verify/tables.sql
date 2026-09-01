-- Recompute the published window and scenario tables from reports/, in SQL.
--
-- Every number in the README and in INCIDENT.md comes out of pandas, in
-- experiments/detector_validation.py and experiments/calibrate.py. Nothing
-- checked those aggregations, because everything downstream reads the same
-- four CSVs they wrote. This derives the same quantities with nothing but
-- SQL, so an error in the pandas would have to be repeated here to survive.
--
-- Output is one "check|status|detail" row per check plus one RESULT line for
-- the cross-language agreement pool. verify/verify.sh fails on any FAIL.
--
-- Run: sqlite3 -init verify/tables.sql :memory: ""

.mode list
.separator |
.headers off

.import --csv reports/drift_vs_degradation.csv windows
.import --csv reports/calibration.csv calib
.import --csv reports/simulated_failures.csv scenarios
.import --csv reports/healthy_control.csv control

-- Everything arrives as TEXT from .import, so cast once here.
CREATE TEMP VIEW w AS
    SELECT CAST("window" AS INT)        AS win,
           CAST("n" AS INT)             AS rows_,
           CAST(share_flagged AS REAL)  AS share,
           CAST(pred_psi AS REAL)       AS psi,
           drifted                      AS drifted,
           CAST(auc AS REAL)            AS auc,
           CAST(auc_drop AS REAL)       AS drop_
    FROM windows;

CREATE TEMP VIEW c AS
    SELECT CAST("window" AS INT)        AS win,
           CAST(share_flagged AS REAL)  AS share,
           CAST(n_null_shift AS INT)    AS n_null,
           CAST(max_null_delta AS REAL) AS null_delta,
           CAST(pred_psi AS REAL)       AS psi,
           would_alert                  AS would_alert
    FROM calib;

CREATE TEMP VIEW s AS
    SELECT scenario, caught,
           features_flagged,
           CAST(substr(features_flagged, 1, instr(features_flagged, '/') - 1) AS INT) AS num,
           CAST(substr(features_flagged, instr(features_flagged, '/') + 1) AS INT)    AS den,
           CAST(share AS REAL)    AS share,
           CAST(pred_psi AS REAL) AS psi,
           CAST(top_psi AS REAL)  AS top_psi
    FROM scenarios;

-- Two-pass Pearson, the same order of operations the C, Go, Rust, R, Java and
-- JavaScript implementations use, so bit level disagreement would be real
-- disagreement and not a summation order artefact.
CREATE TEMP VIEW corr AS
    WITH m AS (SELECT AVG(psi) AS mx, AVG(drop_) AS my FROM w)
    SELECT SUM((psi - mx) * (drop_ - my))
           / sqrt(SUM((psi - mx) * (psi - mx)) * SUM((drop_ - my) * (drop_ - my))) AS r
    FROM w, m;

SELECT 'RESULT sql corr_pred_psi_auc_drop ' || printf('%.15e', r) FROM corr;

-- 1. The headline correlation. README section 1: "Prediction PSI correlates
--    -0.709 with that loss".
SELECT 'corr_vs_published|'
       || CASE WHEN abs(r + 0.709) <= 5e-4 THEN 'ok' ELSE 'FAIL' END
       || '|SQL ' || printf('%.6f', r) || ', README -0.709'
FROM corr;

-- 2. The AUC loss range. README: "the model loses 0.060 to 0.137 AUC".
SELECT 'auc_drop_range|'
       || CASE WHEN abs(lo - 0.060) <= 5e-4 AND abs(hi - 0.137) <= 5e-4
               THEN 'ok' ELSE 'FAIL' END
       || '|min ' || printf('%.4f', lo) || ' max ' || printf('%.4f', hi)
       || ', README 0.060 to 0.137'
FROM (SELECT MIN(drop_) AS lo, MAX(drop_) AS hi FROM w);

-- 3. auc_drop is defined as base_auc - auc, so auc + auc_drop must be the same
--    baseline number in every row. Both columns are rounded to 4 dp in the
--    published file, which allows a spread of 1e-4 and nothing more.
SELECT 'implied_baseline_auc|'
       || CASE WHEN spread <= 2.0e-4 THEN 'ok' ELSE 'FAIL' END
       || '|auc + auc_drop = ' || printf('%.6f', mid)
       || ' across 8 windows, spread ' || printf('%.1e', spread)
FROM (SELECT MAX(auc + drop_) - MIN(auc + drop_) AS spread,
             AVG(auc + drop_) AS mid FROM w);

-- 4. calibration.csv and drift_vs_degradation.csv are written by two different
--    scripts. share_flagged is a property of the feature windows alone, not of
--    the model scoring them, so the two files must agree window by window.
SELECT 'share_flagged_cross_file|'
       || CASE WHEN matched = 8 AND bad = 0 THEN 'ok' ELSE 'FAIL' END
       || '|' || matched || ' windows joined, ' || bad || ' disagree'
FROM (SELECT COUNT(*) AS matched,
             SUM(CASE WHEN w.share <> c.share THEN 1 ELSE 0 END) AS bad
      FROM w JOIN c ON w.win = c.win);

-- 5. 40 features are monitored, so every published share is k/40 for an
--    integer k. A share that is not a fortieth did not come from this pipeline.
SELECT 'share_is_k_over_40|'
       || CASE WHEN bad = 0 THEN 'ok' ELSE 'FAIL' END
       || '|' || total || ' shares checked in three files, ' || bad || ' not a fortieth'
FROM (SELECT COUNT(*) AS total,
             SUM(CASE WHEN abs(sh * 40 - round(sh * 40)) > 1e-9 THEN 1 ELSE 0 END) AS bad
      FROM (SELECT share AS sh FROM w
            UNION ALL SELECT share FROM c
            UNION ALL SELECT share FROM s));

-- 6. simulated_failures.csv publishes the same quantity twice, as the string
--    "3/40" and as the number 0.075. They must not disagree.
SELECT 'features_flagged_string|'
       || CASE WHEN bad = 0 AND total = 5 THEN 'ok' ELSE 'FAIL' END
       || '|' || total || ' scenarios, ' || bad || ' where k/40 does not match share'
FROM (SELECT COUNT(*) AS total,
             SUM(CASE WHEN den <> 40 OR num <> CAST(round(share * 40) AS INT)
                      THEN 1 ELSE 0 END) AS bad
      FROM s);

-- 7. Shapes the README quotes: eight windows, five scenarios, three corrections.
SELECT 'published_row_counts|'
       || CASE WHEN nw = 8 AND ndw = 8 AND nc = 8 AND ns = 5 AND nk = 3
               THEN 'ok' ELSE 'FAIL' END
       || '|windows ' || nw || ' (distinct ' || ndw || '), calibration ' || nc
       || ', scenarios ' || ns || ', corrections ' || nk
FROM (SELECT (SELECT COUNT(*) FROM w) AS nw,
             (SELECT COUNT(DISTINCT win) FROM w) AS ndw,
             (SELECT COUNT(*) FROM c) AS nc,
             (SELECT COUNT(*) FROM s) AS ns,
             (SELECT COUNT(*) FROM control) AS nk);

-- 8. README section 1: "3 of 3 injected failures caught, 0 false alarms on 2
--    controls". The two controls are the two rows the file marks not caught.
SELECT 'caught_three_of_three|'
       || CASE WHEN caught = 3 AND missed = 2 AND controls = 2
               THEN 'ok' ELSE 'FAIL' END
       || '|' || caught || ' caught, ' || missed || ' missed, of which '
       || controls || ' are the labelled controls'
FROM (SELECT SUM(caught = 'True') AS caught,
             SUM(caught = 'False') AS missed,
             SUM(caught = 'False' AND (scenario LIKE '%control%'
                                       OR scenario LIKE '%label shift%')) AS controls
      FROM s);

-- 9. The window row counts are equal width slices of one period, so they may
--    differ by at most one row. A window that is not the size linspace would
--    have made it means the windows were not the ones described.
SELECT 'window_sizes_equal_width|'
       || CASE WHEN hi - lo <= 1 THEN 'ok' ELSE 'FAIL' END
       || '|sizes ' || lo || ' to ' || hi || ', total ' || tot || ' rows'
FROM (SELECT MIN(rows_) AS lo, MAX(rows_) AS hi, SUM(rows_) AS tot FROM w);
