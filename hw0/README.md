## HW0 (Programming): Least Squares via JAX

Instructions for this assignment are given in the HW0 PDF.

Programming assignments in CS 5757 have two components:

- A Jupyter notebook for visualization and experimentation
- A Python module, `solution.py`, where you will implement the required functions

You should edit **only** `solution.py`. The notebook is provided to help you run and visualize your code.

We recommend using **VS Code (or any other python IDE you like)** for coding the assignment and only use Jupyter Notebook to test it.
---
## Configuring Jupyter

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

### 3. Register a Jupyter kernel

    uv pip install ipykernel --python .venv/bin/python
    .venv/bin/python -m ipykernel install --user \
      --name hw0 \
      --display-name "Python (HW0)"

This makes the environment available as a selectable Jupyter kernel.

---

### 4. Select the correct kernel

You can either run the notebook locally using Jupyter or in VS Code.
To run via Jupyter, run:

    uv run jupyter notebook

or, to run in VS Code, open the notebook file directly. 

In either case, when opening the notebook, select:

    Kernel → Change Kernel → Python (HW0)

You can verify that the correct environment is active by running:

    import sys
    print(sys.executable)

---

Each assignment uses its own isolated environment. If something breaks, deleting `.venv/` and re-running the setup steps is safe.

---

## Alternative: Using Conda

If you already have conda installed, you can use it instead of `uv`.

### 1. Create a conda environment

    conda create -n hw0 python=3.12

---

### 2. Activate the environment

    conda activate hw0

---

### 3. Install dependencies

    pip install -r requirements.txt

---

### 4. Register a Jupyter kernel

    pip install ipykernel
    python -m ipykernel install --user \
      --name hw0 \
      --display-name "Python (HW0)"

---

### 5. Select the correct kernel

When opening the notebook in Jupyter or VS Code, select:

    Kernel → Change Kernel → Python (HW0)

You can verify that the correct environment is active by running:

    import sys
    print(sys.executable)

---

To remove the environment if needed:

    conda deactivate
    conda env remove -n hw0