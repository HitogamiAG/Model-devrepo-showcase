VENV_DIR = .venv
PYTHON = python3
VENV_PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_PYTHON) -m pip

.PHONY: all create install install-dev download clean help

$(VENV_DIR):
	@echo ">>> Creating virtual environment $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo ">>> Virtual environment created."

create:
	@echo ">>> Creating virtual environment $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo ">>> Virtual environment created."

install: $(VENV_DIR) requirements.txt
	@echo ">>> Installing requirements from requirements.txt into $(VENV_DIR)..."
	@$(PIP) install -r requirements.txt
	@echo ">>> Installation complete."

install-dev: $(VENV_DIR) requirements-dev.txt
	@echo ">>> Installing development requirements from requirements-dev.txt into $(VENV_DIR)..."
	@$(PIP) install -r requirements-dev.txt
	@echo ">>> Installing pre-commit hooks..."
	@$(VENV_PYTHON) -m pre_commit install
	@echo ">>> Development installation complete."

activate:
	@echo ">>> Activating virtual environment $(VENV_DIR)..."
	@echo ">>> Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment."

download:
	@echo ">>> Downloading data files..."
	@mkdir data
	@curl -L -o data/uavid-v1.zip https://www.kaggle.com/api/v1/datasets/download/dasmehdixtr/uavid-v1
	@unzip data/uavid-v1.zip -d data
	@rm data/uavid-v1.zip
	@echo ">>> Data files downloaded."

clean:
	@echo ">>> Removing virtual environment $(VENV_DIR)..."
	@rm -rf $(VENV_DIR)
	@echo ">>> Removing Python cache files..."
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@echo ">>> Clean complete."

help:
	@echo ">>> Makefile Help:"
	@echo "  make create         - Create a virtual environment."
	@echo "  make install        - Install requirements from requirements.txt."
	@echo "  make install-dev    - Install development requirements from requirements-dev.txt."
	@echo "  make activate       - Activate the virtual environment."
	@echo "  make download       - Download data files."
	@echo "  make clean          - Remove the virtual environment and Python cache files."
	@echo "  make help           - Show this help message."