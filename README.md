# CS5757: Optimization Methods for Robotics

Spring 2026 Repo for [CS5757 - Optimization Methods for Robotics](https://www.cs.cornell.edu/courses/cs5757/2026sp/#).

# Homework Assignments

The `assignments/` folder contains homework assignments: Each homework assignment is in its own directory:
- `hw0/` - Homework 0
- `hw1/` - Homework 1
- etc.

## Getting Started

1. Clone this repository
2. Navigate to the relevant homework directory
3. Follow the instructions in the homework-specific README

## Requirements

See individual homework directories for specific dependencies.

## Submission

Please submit on Gradescope, more details can be found on the website

## Questions

For questions about assignments, please use the course Ed discussion forum or attend office hours.


#Lecture Materials

The `lectures/` folder contains lecture materials: Each lectures have its own directory: This directory contains example code and demos used in lecture for CS 5757: Optimization Methods for Robotics, as well as lecture notes.

## Environment Setup

Choose one of the following methods to set up your Python environment.

### Option 1: Using `uv` (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first if you haven't already.

```bash
# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

To register the environment as a Jupyter kernel:

```bash
uv pip install ipykernel
python -m ipykernel install --user --name=cs5757 --display-name="CS 5757"
```

### Option 2: Using `conda`

```bash
# Create a new conda environment
conda create -n cs5757 python=3.12 -y
conda activate cs5757

# Install dependencies
pip install -r requirements.txt
```

To register the environment as a Jupyter kernel:

```bash
conda install ipykernel -y
python -m ipykernel install --user --name=cs5757 --display-name="CS 5757"
```

## Running the Notebooks

### Jupyter Lab / Notebook

```bash
# If not already installed
pip install jupyterlab  # or: conda install jupyterlab

# Launch
jupyter lab
```

Select the **CS 5757** kernel from the kernel dropdown.

### VS Code

1. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) for VS Code.
2. Open any `.ipynb` file.
3. Click "Select Kernel" in the top-right corner of the notebook.
4. Choose "CS 5757" from the list (or select the Python interpreter from your virtual environment).

> **Note:** If the kernel doesn't appear, reload VS Code or run the `ipykernel install` command above. If you're using `uv`, the kernel may appear as `.venv`.

## Updating Dependencies

Requirements may be updated over the semester. If you encounter missing packages, re-run the install command:

```bash
#cd
cd lectures/

# uv
uv pip install -r requirements.txt

# conda
pip install -r requirements.txt
```

