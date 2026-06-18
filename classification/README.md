# Allotrope classification of borophene structures

## Setup

Using `uv` (recommended):

```bash
uv venv .venv --python 3.12
uv sync
```

## Usage

```bash
# Analyse a trajectory and classify the structures
source .venv/bin/activate  # Activate the virtual environment
python xanalyse_traj -h # to show the help
python xanalyse_traj -in trajectory.lammstrj -st # use to check that vacancies are correctly detected. If not, adjust the minarea, maxarea, dpi and pointsize parameters
python xanalyse_traj -in trajectory.lammstrj # to analyse the trajectory and classify the structures
```
