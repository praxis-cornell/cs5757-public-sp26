## HW2 (Programming): Nonlinear Programming

Instructions for this assignment are given in the HW2 PDF.

Programming assignments in CS 5757 have two components:

- Interactive scripts / Jupyter notebooks for visualization and experimentation
- A Python module, `solution.py`, where you will implement the required functions

You should edit **only** `solution.py`. The script `run_retargeting.py` is provided to help you run and visualize your code.

We recommend using **VS Code (or any other python IDE you like)** for coding the assignment.
---
## Configuring Python

We recommend working locally and using `uv` to manage a per-assignment virtual environment. This avoids dependency conflicts between assignments and makes kernel selection explicit.

If you do not already have `uv` installed, follow the instructions here:
https://docs.astral.sh/uv/getting-started/installation/

From the assignment root directory, run the following commands.

### 1. Create a virtual environment

    uv venv .venv --python 3.12

This creates a virtual environment in `.venv/`.

---

### 2. Install dependencies

    uv pip install -r requirements.txt --python .venv/bin/python

This installs all required packages into the assignment-specific environment.

---

### 3. Running the assignment scripts
This assignment has a single script, `run_retargeting.py`, which you can run with:
    
    uv run python run_retargeting.py

This will run the retargeting app, which you can use to visualize your optimization results.

---

Each assignment uses its own isolated environment. If something breaks, deleting `.venv/` and re-running the setup steps is safe.

---

## Alternative: Using Conda

If you already have conda installed, you can use it instead of `uv`.

### 1. Create a conda environment

    conda create -n hw2 python=3.12

---

### 2. Activate the environment

    conda activate hw2

---

### 3. Install dependencies

    pip install -r requirements.txt

---

### 4. Running the assignment scripts
This assignment has a single script, `run_retargeting.py`, which you can run with:
    
    python run_retargeting.py


---

To remove the environment if needed:

    conda deactivate
    conda env remove -n hw2