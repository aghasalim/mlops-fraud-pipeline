// Structural validation of every published results file, plus one recompute.
//
// The four CSVs under reports/ are the evidence for every number in README.md
// and INCIDENT.md, and the JSON under artifacts/ is the evidence for the deploy
// gate. Nothing checked that any of them is well formed. A truncated write, a
// column that drifted, a NaN escaping a division or a boolean that stopped
// being a boolean would all be invisible until somebody read the table and
// believed it.
//
// This walks all of them, then recomputes the headline correlation as the C,
// SQL, R, Rust, Java and JavaScript implementations also do.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Row counts the README and INCIDENT.md quote: eight held-out windows, five
// injected scenarios, three multiple-testing corrections.
var expectedRows = map[string]int{
	"drift_vs_degradation.csv": 8,
	"calibration.csv":          8,
	"simulated_failures.csv":   5,
	"healthy_control.csv":      3,
}

// Columns that must parse as a number, and the range they must fall in. Shares
// and rates are proportions; PSI and AUC cannot be negative.
var numericRange = map[string][2]float64{
	"share_flagged":          {0, 1},
	"share":                  {0, 1},
	"pred_psi":               {0, math.Inf(1)},
	"top_psi":                {0, math.Inf(1)},
	"auc":                    {0, 1},
	"auc_drop":               {-1, 1},
	"fraud_rate":             {0, 1},
	"max_null_delta":         {0, 1},
	"batch_false_alarm_rate": {0, 1},
	"mean_flagged":           {0, 40},
	"mean_ks_only":           {0, 40},
}

var booleanCols = map[string]bool{
	"drifted": true, "would_alert": true, "caught": true,
}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validate reports every problem in one file rather than the first, so a broken
// run is diagnosed in one pass.
func validate(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for _, h := range header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	base := filepath.Base(path)
	if want, ok := expectedRows[base]; ok && len(rows) != want {
		problems = append(problems,
			fmt.Sprintf("%d data rows, the write-up says %d", len(rows), want))
	}

	for i, row := range rows {
		for j, cell := range row {
			cell = strings.TrimSpace(cell)
			name := header[j]
			if cell == "" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is empty", i+2, name))
				continue
			}
			low := strings.ToLower(cell)
			if low == "nan" || low == "inf" || low == "-inf" || low == "infinity" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, name, cell))
			}
			if booleanCols[name] && cell != "True" && cell != "False" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %q, not a boolean", i+2, name, cell))
			}
			if lohi, ok := numericRange[name]; ok {
				v, err := strconv.ParseFloat(cell, 64)
				switch {
				case err != nil:
					problems = append(problems,
						fmt.Sprintf("row %d column %s is %q, not a number", i+2, name, cell))
				case math.IsNaN(v) || math.IsInf(v, 0):
					problems = append(problems,
						fmt.Sprintf("row %d column %s is not finite", i+2, name))
				case v < lohi[0] || v > lohi[1]:
					problems = append(problems,
						fmt.Sprintf("row %d column %s is %g, outside [%g, %g]",
							i+2, name, v, lohi[0], lohi[1]))
				}
			}
		}
	}
	return problems
}

// The two window files are written by different scripts against different
// models. share_flagged depends only on the feature windows, which are the same
// slices of the same period in both, so the two files must agree window by
// window. Nothing in the repository ever compared them.
func crossFileShare(root string) []string {
	var problems []string
	load := func(name string) (map[string]string, error) {
		header, rows, err := readCSV(filepath.Join(root, "reports", name))
		if err != nil {
			return nil, err
		}
		w, s := col(header, "window"), col(header, "share_flagged")
		if w < 0 || s < 0 {
			return nil, fmt.Errorf("%s has no window/share_flagged column", name)
		}
		out := map[string]string{}
		for _, row := range rows {
			out[row[w]] = row[s]
		}
		return out, nil
	}

	a, err := load("drift_vs_degradation.csv")
	if err != nil {
		return []string{err.Error()}
	}
	b, err := load("calibration.csv")
	if err != nil {
		return []string{err.Error()}
	}
	if len(a) != len(b) {
		problems = append(problems,
			fmt.Sprintf("%d windows against %d", len(a), len(b)))
	}
	for k, va := range a {
		vb, ok := b[k]
		if !ok {
			problems = append(problems, fmt.Sprintf("window %s missing from calibration.csv", k))
			continue
		}
		fa, _ := strconv.ParseFloat(va, 64)
		fb, _ := strconv.ParseFloat(vb, 64)
		if fa != fb {
			problems = append(problems,
				fmt.Sprintf("window %s: share_flagged %s against %s", k, va, vb))
		}
	}
	return problems
}

// artifacts/deploy_decision.json is what the README points at for the gate.
// deployable must be the conjunction of the individual checks, or the artifact
// is claiming something its own contents do not support.
func deployDecision(root string) []string {
	var doc struct {
		Deployable   bool            `json:"deployable"`
		Checks       map[string]bool `json:"checks"`
		ModelVersion string          `json:"model_version"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "artifacts", "deploy_decision.json"))
	if err != nil {
		return []string{err.Error()}
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return []string{fmt.Sprintf("deploy_decision.json: %v", err)}
	}
	if len(doc.Checks) == 0 {
		return []string{"deploy_decision.json carries no checks"}
	}
	all := true
	for _, v := range doc.Checks {
		all = all && v
	}
	if doc.Deployable != all {
		return []string{fmt.Sprintf("deployable is %v but the %d checks conjoin to %v",
			doc.Deployable, len(doc.Checks), all)}
	}
	if doc.ModelVersion == "" {
		return []string{"deploy_decision.json names no model version"}
	}
	return nil
}

func pearson(x, y []float64) float64 {
	n := float64(len(x))
	var mx, my float64
	for i := range x {
		mx += x[i]
		my += y[i]
	}
	mx /= n
	my /= n
	var sxy, sxx, syy float64
	for i := range x {
		dx, dy := x[i]-mx, y[i]-my
		sxy += dx * dy
		sxx += dx * dx
		syy += dy * dy
	}
	return sxy / math.Sqrt(sxx*syy)
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	bad := 0
	report := func(label string, problems []string) {
		if len(problems) == 0 {
			return
		}
		bad += len(problems)
		for _, p := range problems {
			fmt.Printf("  %s: %s\n", label, p)
		}
	}

	files, err := filepath.Glob(filepath.Join(*root, "reports", "*.csv"))
	if err != nil || len(files) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s/reports\n", *root)
		os.Exit(2)
	}
	sort.Strings(files)

	fmt.Printf("validating %d files under reports/\n", len(files))
	before := bad
	for _, path := range files {
		report(filepath.Base(path), validate(path))
	}
	if bad == before {
		fmt.Println("  no ragged rows, duplicate columns, empty cells, NaN, Inf,")
		fmt.Println("  out-of-range proportions or non-boolean verdicts anywhere")
	}

	report("cross-file", crossFileShare(*root))
	report("deploy_decision.json", deployDecision(*root))

	// Fourth independent recomputation of the headline correlation.
	header, rows, err := readCSV(filepath.Join(*root, "reports", "drift_vs_degradation.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "drift_vs_degradation.csv: %v\n", err)
		os.Exit(2)
	}
	pc, dc := col(header, "pred_psi"), col(header, "auc_drop")
	if pc < 0 || dc < 0 {
		fmt.Fprintln(os.Stderr, "drift_vs_degradation.csv is missing a needed column")
		os.Exit(2)
	}
	x := make([]float64, 0, len(rows))
	y := make([]float64, 0, len(rows))
	for _, row := range rows {
		a, err1 := strconv.ParseFloat(strings.TrimSpace(row[pc]), 64)
		b, err2 := strconv.ParseFloat(strings.TrimSpace(row[dc]), 64)
		if err1 != nil || err2 != nil {
			fmt.Fprintln(os.Stderr, "a window row does not parse as a number")
			os.Exit(2)
		}
		x = append(x, a)
		y = append(y, b)
	}
	r := pearson(x, y)
	fmt.Printf("RESULT go corr_pred_psi_auc_drop %.15e\n", r)
	if math.Abs(r-(-0.709)) > 5e-4 {
		fmt.Printf("  correlation: Go gets %+.6f, the README publishes -0.709\n", r)
		bad++
	} else {
		fmt.Printf("\ncorrelation over %d windows %+.15f, README -0.709, ok\n", len(x), r)
	}

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Println("reports/ is well formed and the two window files agree")
}
