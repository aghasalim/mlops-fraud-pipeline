.PHONY: setup serve validate simulate test docker dashboard register clean
PY := .venv/bin/python

setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt pytest

serve:            ## run the scoring API locally
	.venv/bin/uvicorn src.pipeline.serving:app --reload --port 8000

validate:         ## the instrument check: false alarms, and does drift predict AUC loss
	$(PY) -m experiments.detector_validation

simulate:         ## inject failures, confirm the monitor catches them
	$(PY) -m experiments.simulated_failure

register:         ## log the model + gate metrics to MLflow
	$(PY) -m src.pipeline.registry

dashboard:
	.venv/bin/streamlit run app/dashboard.py

test:
	$(PY) -m pytest tests/ -q

docker:
	docker build -t mlops-fraud-pipeline .

clean:
	rm -rf reports mlruns data/predictions.jsonl
