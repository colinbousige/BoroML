"""
# Active Learning

List of different functions to be called in an active learning script to build the dataset for an NNP.

- Author: Colin BOUSIGE
- Email: colin.bousige@cnrs.fr
- Affiliation: CNRS, [Laboratoire des Multimatériaux et Interfaces](http://lmi.cnrs.fr), Lyon, France
- Date: 2024/11/12

## Usage

First, add this folder (`$HOME/Boro_ML/bin/active_learning` or wherever you put it) to your `$PATH` and `$PYTHONPATH`.

Then, go to your working directory, and make sure the following files are there:

- Necessary:
  - `stock.data`: `input.data` file with all stock structures
  - `initial_input.data.`: initial `input.data` structures dataset
  - `input1.nn`: `input.nn` for NNP1
  - `input2.nn`: `input.nn` for NNP2
- Optional:
  - `EW.data`: `input.data` file with only EW structures. They will be computed with VASP and added to the dataset at the beginning of the procedure. This file is created with the script `xLAMMPStoNNP`.
  - `input3.nn`: `input.nn` for NNP3
  - etc

Then do:

```bash
xjobactive node26
```

It will copy the python script `$HOME/Boro_ML/bin/active_learning/active_training.py` in the current directory, and launch the script on `node26`.

Alternatively, you can copy the script manually:

```bash
cp <path>/active_learning/jobactive .
```

Then edit `jobactive` to your needs and launch it with `sbatch jobactive`.

## Note

`stock.data` is a file containing all the structures that will be used in the active learning procedure. It is usually created with the script `xLAMMPStoNNP` that gathers all structures in a LAMMPS trajectory into an n2p2-type `stock.data` file. If created by this script, all energies and forces are set to 0 since they are not known at the DFT level.

In case you have already computed some structures with DFT, you can still include them in the `stock.data` file. In this case, all structures that have non zero energies will be extracted to individual `vasp/OUTCAR_i.inp` files. This way, they will be detected as already computed and not be computed again during the active learning procedure.

## Adapting to your own needs

- You must have access to the `potpaw_pbe` folder of VASP to use this script. The path to the folder containing `potpaw_pbe` is set in the `jobactive` script created by running `xjobactive`. It is set by default to `$HOME/bin/vasp`, you need to change it if it not there.
- `environments.py` is a file containing the environment variables for the cluster. You need to adapt it to your own cluster and where you have installed `n2p2`, `gsl`, `openmpi`, and `vasp`.
- `SlurmJob` is a class to run and manage `n2p2` and `VASP` jobs on `lynx`, the iLM cluster. You need to adapt it to your own cluster if your cluster is not using SLURM or if you don't need to specify the queue when submitting a job on your cluster.

## List of available classes

- `SlurmJob`
  - A class to run and manage `n2p2` and `VASP` jobs on `lynx`
- `Cluster`
  - A class to access cluster state
- `ActiveTraining`
  - A class to perform an Active Training

## List of available functions

- `read_inputdata(filename: str, copy_data=None)`
  - Read `input.data` file and return a list of Atoms objects and the comments. If `copy_data` is a path, it will copy the corresponding inp file to `copy_data/OUTCAR_i.inp` for all structures with non-zero energy and indices i.
- `write_inputdata(filename: str, atoms, comments)`
  - From a list of Atoms object and comments, write an `input.data` file to `filename`.
- `print_success_message()`
  - Print a geeky success message.

"""

from .Cluster import Cluster
from .SlurmJob import SlurmJob
from .ActiveTraining import ActiveTraining
from .functions import *
from .environment import *
