# Convenience targets (macOS/Linux). Windows users: see scripts/setup.ps1 or the README.
PY ?= .venv/bin/python

.PHONY: setup test lint typecheck train kb rca rca-llm api worker dashboard run generate summary simulate up down logs clean

setup:            ## Create venv, install deps, run tests, generate first dataset
	bash scripts/setup.sh

test:             ## Run unit tests
	$(PY) -m pytest -v

lint:             ## Static checks
	$(PY) -m ruff check src tests dashboard

typecheck:        ## Strict type checking
	$(PY) -m mypy

train:            ## Train + evaluate detectors (writes models/ and reports/)
	$(PY) -m aiops_copilot.evaluation.run_detection

kb:               ## Build the RAG knowledge base
	$(PY) -m aiops_copilot.rag.build

rca:              ## Evaluate root-cause analysis (offline retrieval)
	$(PY) -m aiops_copilot.evaluation.run_rca

rca-llm:          ## Evaluate the Claude RCA agent on 20 incidents (needs LLM_API_KEY)
	$(PY) -m aiops_copilot.evaluation.run_rca --engine llm --limit 20

api:              ## Backend API on :8000
	$(PY) -m aiops_copilot.api

worker:           ## Real-time detection + RCA + alerts
	$(PY) -m aiops_copilot.pipeline.worker

dashboard:        ## Streamlit dashboard on :8501
	$(PY) -m streamlit run dashboard/app.py

run:              ## Start simulator, API, worker and dashboard
	bash scripts/run_local.sh

generate:         ## Generate the default 24h labelled dataset
	$(PY) -m aiops_copilot.generator generate

summary:          ## Print stats of the latest dataset
	$(PY) -m aiops_copilot.generator summary --examples 1

simulate:         ## Run the live simulator locally on :8001
	$(PY) -m aiops_copilot.simulator

up:               ## Start the full Docker stack
	docker compose up -d --build

down:             ## Stop the stack
	docker compose down

logs:             ## Follow worker logs
	docker compose logs -f worker

clean:            ## Remove generated data and caches
	rm -rf data models logs .pytest_cache .mypy_cache .ruff_cache
