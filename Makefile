.PHONY: setup extract build train-xgb train-lstm train-all eval clean help

# Default target
help:
	@echo "OFI Prediction Project - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup         Install dependencies and create directories"
	@echo ""
	@echo "Data Processing:"
	@echo "  make build         Build modeling dataset from raw parquet files"
	@echo ""
	@echo "Training:"
	@echo "  make train-xgb     Train XGBoost model"
	@echo "  make train-lstm    Train LSTM model"
	@echo "  make train-all     Train all models"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval          Compare trained models"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove processed data and outputs"
	@echo "  make clean-all     Remove all generated files"
	@echo ""
	@echo "Note: WRDS extraction must be run separately (see README.md)"

# Setup environment
setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed outputs/models outputs/metrics

# Build dataset from raw parquet files
build:
	python src/build_dataset.py --input data/raw --output data/processed/dataset.parquet

# Train XGBoost model
train-xgb:
	python src/train_xgb.py \
		--input data/processed/dataset.parquet \
		--model-output outputs/models/xgb_model.json \
		--metrics-output outputs/metrics/xgb_metrics.json

# Train LSTM model
train-lstm:
	python src/train_lstm.py \
		--input data/processed/dataset.parquet \
		--model-output outputs/models/lstm_model.pt \
		--metrics-output outputs/metrics/lstm_metrics.json

# Train all models
train-all: train-xgb train-lstm

# Evaluate and compare models
eval:
	python src/eval.py --metrics-dir outputs/metrics

# Full pipeline (after WRDS data is downloaded)
pipeline: build train-all eval

# Clean processed data and outputs
clean:
	rm -rf data/processed/*
	rm -rf outputs/models/*
	rm -rf outputs/metrics/*

# Clean everything including raw data
clean-all: clean
	rm -rf data/raw/*

# Development: check syntax
check:
	python -m py_compile src/utils.py
	python -m py_compile src/wrds_extract.py
	python -m py_compile src/build_dataset.py
	python -m py_compile src/train_xgb.py
	python -m py_compile src/train_lstm.py
	python -m py_compile src/eval.py
	@echo "All syntax checks passed!"
