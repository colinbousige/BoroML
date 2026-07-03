# HPC Active Learning Workflow

This folder contains the bash/python workflow used to run active learning on HPC systems with:

- n2p2 for scaling/training/prediction
- LAMMPS for MD tests
- VASP for DFT labeling of selected structures

The main entry point is `goal.sh`, which dispatches the workflow stages.

## 1) What you need before starting

Make sure scripts in this folder are executable and available by adding it to your `PATH`.

Then, run all commands from your working directory, where you need the following files:

- Input/control files:
  - `input_AL`
  - `<STEP>_input.data` (example: `0_input.data`)
  - `<STEP>_STOCK.data` (example: `0_STOCK.data`)
  - `input1.nn`, `input2.nn`, ... `inputN.nn`
- Environment templates:
  - `job.env`
  - `python.env`
  - `n2p2.env`
  - `lammps.env`
  - `vasp.env`
- VASP files:
  - `INCAR`, `POTCAR`, `KPOINTS`
- LAMMPS files referenced by `input_AL`:
  - one LAMMPS input script (`LAMMPS_INPUT`)
  - one or more structure/data files (`LAMMPS_DAT_FILES`)

Template files are centralized in `templates/`:

- `templates/input_AL`
- `templates/input.lmp`
- `templates/job.env`
- `templates/python.env`
- `templates/n2p2.env`
- `templates/lammps.env`
- `templates/vasp.env`

Recommended: create a run directory from templates with the helper script:

```bash
scripts/active_learning/init_run_dir.sh <your-run-directory>
```

Use `--force` to overwrite existing files in the run directory:

```bash
scripts/active_learning/init_run_dir.sh <your-run-directory> --force
```

Manual alternative:

Copy templates into your run directory, then adapt to your cluster settings before launching jobs:

```bash
cp /path/to/BoroML/scripts/active_learning/templates/* .
```

If you are running from the repository root:

```bash
cp scripts/active_learning/templates/* <your-run-directory>/
```

## 2) Quick start for one AL step

Assume you are starting at step `0`.

1. Initialize the AL step:

```bash
./goal.sh init 0
```

This creates and configures:

- stock chunks (`stock1.data`, `stock2.data`, ...) to avoid memory issues with large stock files
- run directories (`0_NNP*`, `0_vasp*`, plots, logs)
- master AL job script: `0_job_al`

1. Submit the master AL job:

```bash
sbatch 0_job_al
```

1. Follow progress:

```bash
tail -f 0_status.log
```

## 3) Other common commands

The dispatcher is:

```bash
./goal.sh <command> [args]
```

Main commands:

- `init <step>`
  - Prepare one AL step and generate `<step>_job_al`.
- `train <step> <input.data> <YES|NO>`
  - Prepare `<step>_TRAIN` and optional standalone training submission.
- `xMDs <step> <epoch> <YES|NO>`
  - Build and optionally submit MD-test jobs from trained weights.
- `get_EWnb <step>`
  - Count extrapolation warnings from MD tests.
- `EW_DFT <step> <YES|NO>`
  - Build and optionally submit VASP jobs for selected EW structures.
- `rnw_dataset <step> <old_input.data>`
  - Create `<step+1>_input.data` by appending newly DFT-labeled structures.
- `rnw_stck <step> <YES|NO>`
  - Build jobs to regenerate stock candidates from dumps/randomized structures.
- `crt_stck <step>`
  - Create `<step+1>_STOCK.data` from renewed stock fragments.

## 4) Suggested multi-step cycle

Typical loop:

1. `./goal.sh init <k>`
2. `sbatch <k>_job_al`
3. optional dedicated final training: `./goal.sh train <k> <k>_input.data YES`
4. MD tests: `./goal.sh xMDs <k> <epoch> YES`
5. EW DFT: `./goal.sh EW_DFT <k> YES`
6. dataset update: `./goal.sh rnw_dataset <k> <k>_input.data`
7. stock update: `./goal.sh rnw_stck <k> YES` then `./goal.sh crt_stck <k>`
8. next step: `./goal.sh init <k+1>`

## 5) Important placeholders in env templates

The scripts expect these placeholders in `job.env`:

- `NNODES` (replaced by target node count)
- `jobname` (replaced by stage-specific name)

If either placeholder is missing, generated job scripts may be invalid.

## 6) Troubleshooting

- Error: missing `python.env`, `job.env`, `n2p2.env`, `lammps.env`, or `vasp.env`
  - Create the missing file in the run directory and retry.
- Error: missing `input_AL` keys
  - Check spelling and capitalization of parameter names.
- Error: `<step>_TRAIN` or `MDs_*` directory not found
  - Run earlier pipeline stages first (`train` before `xMDs`, `xMDs` before `EW_DFT`, etc.).
- Empty converted OUTCAR data
  - Very high-force structures are filtered in update scripts; this is expected for unstable configurations.

## Authors

- Initial Python version: [Colin Bousige](mailto:colin.bousige@cnrs.fr), Laboratoire des Multimatériaux et Interfaces, Lyon, France
- Conversion to shell scripts and adaptation to generic HPC workflow: [Pierre Mignon](mailto:pierre.mignon@univ-lyon1.fr), Institut Lumière Matière, Lyon, France
