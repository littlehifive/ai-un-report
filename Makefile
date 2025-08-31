# UN Reports RAG - Makefile

.PHONY: help install build app clean test lint

# Default target
help:
	@echo "UN Reports RAG - Available Commands"
	@echo "=================================="
	@echo "make install    - Install dependencies"
	@echo "make build      - Run full pipeline (discover->fetch->parse->index)"
	@echo "make app        - Launch Streamlit app"
	@echo "make clean      - Clean generated data files"
	@echo "make test       - Run basic tests"
	@echo "make lint       - Run code linting"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run full pipeline
build:
	@echo "🏗️  Running full pipeline..."
	./scripts/build_all.sh

# Launch Streamlit app
app:
	@echo "🚀 Starting Streamlit app..."
	@echo "Navigate to: http://localhost:8501"
	streamlit run src/app.py

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/raw/*
	rm -rf data/parsed/*
	rm -f data/*.faiss
	rm -f data/*.json
	rm -f data/*.parquet
	rm -rf logs/*
	@echo "✅ Clean complete!"

# Run basic tests
test:
	@echo "🧪 Running basic tests..."
	@cd src && python -c "import utils, discover, fetch, parse, index; print('✅ All modules import successfully')"
	@if [ ! -f ".env" ] && [ -z "$$OPENAI_API_KEY" ]; then \
		echo "⚠️  Warning: No OpenAI API key found. Set OPENAI_API_KEY or create .env file."; \
	else \
		echo "✅ OpenAI API key found"; \
	fi
	@echo "✅ Basic tests passed!"

# Lint code (if flake8 is available)
lint:
	@echo "🔍 Linting code..."
	@if command -v flake8 > /dev/null; then \
		flake8 src/ --max-line-length=100 --ignore=E501,W503; \
		echo "✅ Linting complete!"; \
	else \
		echo "ℹ️  flake8 not installed, skipping linting"; \
	fi

# Quick start - install and build
quickstart: install build
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo "Run 'make app' to launch the Streamlit interface"

# Development setup
dev-setup: install
	@echo "🔧 Setting up development environment..."
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		echo "📝 Created .env file from template. Please add your OpenAI API key."; \
	fi
	@echo "✅ Development setup complete!"