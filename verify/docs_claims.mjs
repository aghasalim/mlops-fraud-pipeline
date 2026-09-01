// Does the write-up still say what the data says?
//
// README.md and INCIDENT.md quote about sixty numbers between them, every one
// of them typed in by hand from a printed pandas frame. The CSVs under reports/
// can be regenerated at any time by `make validate`, and nothing would notice
// if a rerun moved a figure while the prose kept the old one. That is the most
// likely way this repository ends up making a false claim: not a wrong
// calculation, a stale sentence.
//
// So this reads both documents, pulls out every figure that has a source in
// reports/, and compares it at the precision the document itself uses. A number
// written to 3 dp is required to match the data rounded to 3 dp, and no better.
//
// Node has no CSV or markdown dependency here on purpose: the parsing needed is
// a split on commas and a split on pipes.

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? ".";
const checks = [];
const add = (name, ok, detail) => checks.push({ name, ok, detail });

function readCsv(name) {
  const text = readFileSync(join(root, "reports", name), "utf8").trim();
  const [head, ...lines] = text.split("\n");
  const cols = head.split(",").map((c) => c.trim());
  return lines.map((line) => {
    const cells = line.split(",").map((c) => c.trim());
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
}

// A document number is read at its own precision: "0.060" has to match the data
// rounded to three decimals, "11.9" to one. Anything else would either reject
// honest rounding or accept a figure that has drifted.
function agrees(written, value) {
  const clean = String(written).replace("\u2212", "-").replace("+", "");
  const dot = clean.indexOf(".");
  const dp = dot < 0 ? 0 : clean.length - dot - 1;
  return Number(clean) === Number(value.toFixed(dp));
}

function pearson(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy);
}

const windows = readCsv("drift_vs_degradation.csv");
const scenarios = readCsv("simulated_failures.csv");
const control = readCsv("healthy_control.csv");

const psi = windows.map((r) => Number(r.pred_psi));
const drop = windows.map((r) => Number(r.auc_drop));
const share = windows.map((r) => Number(r.share_flagged));
// Average ranks for ties, which is what R's rank() does and what the Spearman
// figure in verify/inference.R is computed from.
function ranks(v) {
  const order = [...v.keys()].sort((a, b) => v[a] - v[b]);
  const out = new Array(v.length);
  let i = 0;
  while (i < v.length) {
    let j = i;
    while (j + 1 < v.length && v[order[j + 1]] === v[order[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[order[k]] = avg;
    i = j + 1;
  }
  return out;
}

const corrPsi = pearson(psi, drop);
const corrShare = pearson(share, drop);
const rho = pearson(ranks(psi), ranks(drop));

console.log(`RESULT js corr_pred_psi_auc_drop ${corrPsi.toExponential(15)}`);
console.log(`RESULT js corr_share_auc_drop ${corrShare.toExponential(15)}`);
console.log(`RESULT js spearman ${rho.toExponential(15)}`);

// The documents use a typographic minus, which is not the character a parser
// expects and is not something to normalise away in the file itself.
const docs = Object.fromEntries(
  ["README.md", "INCIDENT.md"].map((f) => [
    f,
    readFileSync(join(root, f), "utf8").replaceAll("\u2212", "-"),
  ]),
);

const cells = (line) =>
  line.split("|").map((c) => c.trim()).filter((c, i, a) => !(c === "" && (i === 0 || i === a.length - 1)));

// 1. The scenario table, in both documents, against simulated_failures.csv.
for (const [file, text] of Object.entries(docs)) {
  const lines = text.split("\n");
  for (const s of scenarios) {
    const row = lines.find((l) => l.startsWith(`| ${s.scenario} |`));
    if (!row) {
      add(`${file} scenario table`, false, `no row for ${s.scenario}`);
      continue;
    }
    const caught = s.caught === "True";
    const says = /\*\*yes\*\*/.test(row) ? true : /\*\*no\*\*/.test(row) ? false : null;
    add(`${file}: ${s.scenario} verdict`, says === caught,
        `document says ${says}, simulated_failures.csv says ${caught}`);

    const detail = cells(row).slice(2).join(" ");
    const flagged = detail.match(/(\d+)\/(\d+) features?/);
    if (flagged) {
      const [, k, of] = flagged;
      const want = Math.round(Number(s.share) * Number(of));
      add(`${file}: ${s.scenario} features flagged`,
          Number(k) === want && s.features_flagged === `${k}/${of}`,
          `document ${k}/${of}, file ${s.features_flagged}, share ${s.share}`);
    }
    const pred = detail.match(/prediction PSI \*{0,2}(-?[\d.]+)/);
    if (pred) {
      add(`${file}: ${s.scenario} prediction PSI`, agrees(pred[1], Number(s.pred_psi)),
          `document ${pred[1]}, file ${s.pred_psi}`);
    }
    const top = detail.match(/`(\w+)` PSI \*{0,2}([\d.]+)/);
    if (top) {
      add(`${file}: ${s.scenario} top feature`,
          top[1] === s.top_feature && agrees(top[2], Number(s.top_psi)),
          `document ${top[1]} ${top[2]}, file ${s.top_feature} ${s.top_psi}`);
    }
  }
}

// 2. The window table in INCIDENT.md, against drift_vs_degradation.csv.
{
  const lines = docs["INCIDENT.md"].split("\n");
  const head = lines.findIndex((l) => l.startsWith("|") && /AUC drop/.test(l));
  const rows = lines
    .slice(head + 2, head + 2 + windows.length)
    .map(cells)
    .filter((c) => /^\d+$/.test(c[0]));
  add("INCIDENT.md window table shape", rows.length === windows.length,
      `${rows.length} rows against ${windows.length} in the file`);
  for (const c of rows) {
    const w = windows.find((r) => r.window === c[0]);
    if (!w) {
      add(`INCIDENT.md window ${c[0]}`, false, "no such window in the file");
      continue;
    }
    const flagged = c[3].replace(/\*/g, "") === "yes";
    const ok =
      agrees(c[1], Number(w.share_flagged)) &&
      agrees(c[2], Number(w.pred_psi)) &&
      flagged === (w.drifted === "True") &&
      agrees(c[4], Number(w.auc)) &&
      agrees(c[5].replace(/\*/g, ""), Number(w.auc_drop));
    add(`INCIDENT.md window ${c[0]}`, ok,
        `document ${c.slice(1).join(" ")} against file ${w.share_flagged} ${w.pred_psi} ${w.drifted} ${w.auc} ${w.auc_drop}`);
  }
}

// 3. The healthy-control table in INCIDENT.md, against healthy_control.csv.
{
  const names = { none: "none", Bonferroni: "bonferroni", "Benjamini-Hochberg": "bh" };
  const lines = docs["INCIDENT.md"].split("\n");
  for (const [written, key] of Object.entries(names)) {
    const row = lines.find((l) => l.startsWith(`| ${written} |`));
    const rec = control.find((r) => r.correction === key);
    if (!row || !rec) {
      add(`INCIDENT.md control ${written}`, false, "row missing from document or file");
      continue;
    }
    const c = cells(row).map((v) => v.replace(/\*/g, "").replace("%", ""));
    const ok =
      agrees(c[1], Number(rec.mean_flagged)) &&
      agrees(c[2], Number(rec.mean_ks_only)) &&
      agrees(c[3], 100 * Number(rec.batch_false_alarm_rate));
    add(`INCIDENT.md control ${written}`, ok,
        `document ${c.slice(1).join(" ")} against file ${rec.mean_flagged} ${rec.mean_ks_only} ${rec.batch_false_alarm_rate}`);
  }
}

// 4. The figures quoted in prose, wherever they appear in either document.
{
  const ksOnly = Number(control.find((r) => r.correction === "none").mean_ks_only);
  const lo = Math.min(...drop), hi = Math.max(...drop);
  const caught = scenarios.filter((r) => r.caught === "True").length;
  const missed = scenarios.length - caught;

  for (const [file, text] of Object.entries(docs)) {
    // Both correlations the documents quote in this range: the Pearson figure
    // the third finding is built on, and the rank correlation beside it. A
    // number that has drifted from the data matches neither.
    const negatives = [...text.matchAll(/-0\.7\d+/g)].map((m) => m[0]);
    add(`${file}: every quoted -0.7x correlation`,
        negatives.length > 0 &&
          negatives.every((v) => agrees(v, corrPsi) || agrees(v, rho)),
        `${negatives.length} occurrences ${[...new Set(negatives)].join(", ")}, recomputed ${corrPsi.toFixed(6)} and ${rho.toFixed(6)}`);

    const positives = [...text.matchAll(/\+0\.\d+/g)].map((m) => m[0]);
    if (positives.length) {
      add(`${file}: quoted feature-share correlation`,
          positives.every((v) => agrees(v, corrShare)),
          `${positives.join(", ")}, recomputed ${corrShare.toFixed(6)}`);
    }

    const ks = [...text.matchAll(/\b3\.\d+ (?:of 40|features)/g)].map((m) => m[0]);
    if (ks.length) {
      add(`${file}: KS-alone false alarms`,
          ks.every((v) => agrees(v.split(" ")[0], ksOnly)),
          `${ks.join(", ")}, healthy_control.csv ${ksOnly}`);
    }

    const range = text.match(/(\d\.\d+) to (\d\.\d+) AUC/);
    if (range) {
      add(`${file}: AUC loss range`, agrees(range[1], lo) && agrees(range[2], hi),
          `document ${range[1]} to ${range[2]}, file ${lo} to ${hi}`);
    }

    const score = text.match(/(\d+) of (\d+) (?:injected|real) failures caught, (\d+) false alarms on (\d+) controls/);
    if (score) {
      add(`${file}: headline count`,
          Number(score[1]) === caught && Number(score[2]) === caught &&
          Number(score[3]) === 0 && Number(score[4]) === missed,
          `document ${score[0]}, file ${caught} caught and ${missed} missed`);
    }
  }
}

const failed = checks.filter((c) => !c.ok);
console.log(`  ${checks.length} figures in README.md and INCIDENT.md traced back to reports/`);
for (const c of failed) console.log(`  FAIL ${c.name}: ${c.detail}`);
if (failed.length) {
  console.log(`\n${failed.length} of ${checks.length} document claims disagree with the data`);
  process.exit(1);
}
console.log("  every one agrees at the precision the document uses");
