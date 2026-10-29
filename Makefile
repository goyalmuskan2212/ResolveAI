SHELL := bash

.PHONY: help setup setup-local setup-cloud index run-api run-ui test smoke-test clean

help: ## Show available commands
	@echo
	@echo   AI Support Agent - Available Commands
	@echo   =====================================
	@echo
	@echo   Setup:
	@echo     make setup-local   Start local infrastructure + index knowledge base
	@echo     make setup-cloud   Index knowledge base (cloud services configured in .env)
	@echo
	@echo   Run:
	@echo     make run-api       Start FastAPI server (port 8000)
	@echo     make run-ui        Start Streamlit UI (port 8501)
	@echo
	@echo   Test:
	@echo     make test          Run all tests
	@echo     make smoke-test    Run smoke test (health checks + 3 demo scenarios)
	@echo
	@echo   Other:
	@echo     make index         Index knowledge base into Qdrant
	@echo     make clean         Stop local infrastructure and remove volumes
	@echo

setup: setup-local ## Alias for setup-local

setup-local: ## Start local Docker services + index knowledge base
	docker-compose --profile local up -d
	@echo Waiting for services to be ready...
	@sleep 5
	python scripts/index_knowledge_base.py
	@echo
	@echo Setup complete. Run 'make run-api' and 'make run-ui' to start.

setup-cloud: ## Index knowledge base (assumes cloud services in .env)
	python scripts/index_knowledge_base.py
	@echo
	@echo Setup complete. Run 'make run-api' and 'make run-ui' to start.

index: ## Index knowledge base into Qdrant
	python scripts/index_knowledge_base.py

run-api: ## Start FastAPI server with auto-reload
	uvicorn app.main:app --reload

run-ui: ## Start Streamlit UI
	streamlit run streamlit_app.py

test: ## Run all tests
	pytest tests/ -v

smoke-test: ## Run smoke test (3 demo scenarios via API)
	python scripts/smoke_test.py

clean: ## Stop local infrastructure and remove volumes
	docker-compose --profile local down -v
