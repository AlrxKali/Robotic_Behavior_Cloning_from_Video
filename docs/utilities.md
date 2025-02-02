# 📦 Utilities

## Overview
Helper scripts for **file handling, logging, and configuration management**.

## Key Components
- **`file_utils.py`** - File and directory operations.
- **`logging_utils.py`** - Logging setup.
- **`timer_utils.py`** - Function timing utilities.
- **`config_loader.py`** - Loads configuration settings from JSON/YAML.

## Example Usage
```python
from utils.logging_utils import setup_logger
logger = setup_logger("main", "app.log")
logger.info("Application started.")
```