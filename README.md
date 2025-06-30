# Neural network approach for a rapid prediction of metal-supported borophene properties <a href="https://zenodo.org/badge/latestdoi/698260056"><img src="https://zenodo.org/badge/698260056.svg" alt="DOI" align="right"></a>

This archive contains various files and scripts linked with the articles listed in the [How to cite](#how-to-cite) section below. It includes sample LAMMPS input files, potential files for different machine learning interatomic potentials (MLIPs) for **borophene on silver**, and various scripts for structure generation, post-treatment of data, and adaptive training of neural network potentials.

The MLIPs files included are for the following methods:
- [DeePMD](https://github.com/deepmodeling/deepmd-kit)
- [n2p2](https://github.com/compPhysVienna/n2p2/)
- [NNMP](https://github.com/allouchear/NNMP-Pot)

Please [cite these articles](#how-to-cite) if you use any of the files in this archive.

----

#### Table of contents

- [Contents of this archive](#contents-of-this-archive)
- [Description of the scripts](#description-of-the-scripts)
  - [Structure generators](#structure-generators)
  - [Post-treatment scripts](#post-treatment-scripts)
  - [Adaptive training scripts](#adaptive-training-scripts)
- [How to cite](#how-to-cite)
- [Author](#author)
- [License](#license)
- [Acknowledgments](#acknowledgments)

----

## Contents of this archive

```bash
.
├── LICENSE
├── README.md
├── lammps # Sample LAMMPS input files for launching an MD simulation with the MLIPs
│   ├── DeePMD
│   │   ├── input.lmp
│   │   └── structure.data
│   ├── n2p2
│   │   ├── input.lmp
│   │   └── structure.data
│   └── NNMP
│       ├── input.lmp
│       └── structure.data
├── potential # The actual MLIPs potential files
│   ├── DeePMD
│   │   └── graph.pb
│   ├── n2p2
│   │   ├── input.data
│   │   ├── input.nn
│   │   ├── scaling.data
│   │   ├── weights.005.data
│   │   └── weights.047.data
│   ├── n2p2_JACS # The potential used in the JACS article
│   │   ├── input.data
│   │   ├── input.nn
│   │   ├── scaling.data
│   │   ├── weights.005.data
│   │   └── weights.047.data
│   └── NNMP
│       ├── best
│       │   ├── best_loss.npz
│       │   ├── best.ckpt.data-00000-of-00001
│       │   ├── best.ckpt.index
│       │   └── checkpoint
│       ├── model_parameters.txt
│       └── train.log
└── scripts # Various scripts used for post-treating the data or generating structures
    ├── adaptive_learning
    │   ├── __init__.py
    │   ├── adaptive_training.py
    │   ├── AdaptiveTraining.py
    │   ├── Cluster.py
    │   ├── environment.py
    │   ├── functions.py
    │   ├── jobadaptive
    │   ├── README.md
    │   ├── SlurmJob.py
    │   └── xjobadaptive
    ├── BoroPy.py
    ├── generatorfunctions.py
    ├── gofr.c
    ├── packages.txt
    ├── read_write.py
    ├── requirements.txt
    ├── xconvert
    ├── xGDOS
    ├── xgenerate-structure
    ├── xLAMMPStoNNP
    ├── xOUTCARtoLAMMPS
    ├── xplotLAMMPSlog
    ├── xplotLC
    ├── xprepareDPdata
    ├── xreadLAMMPSlog
    └── xSTM
```

## Description of the scripts

Install all necessary libraries with `pip install -r requirements.txt`.

### Structure generators

- `BoroPy.py`:
  - Streamlit GUI tool to build and visualize borophene structures
  - Usage (local): `streamlit run BoroPy.py`
  - Or use the [online app](https://boroml-6lykqznd6cqcahzupk96cr.streamlit.app/)
- `generatorfunctions.py`:
  - Set of functions to generate borophene structures, write LAMMPS input files, etc. Called in other scripts.
- `xgenerate-structure`:
  - Generate a borophene structure to stdout
  - Usage: `python xgenerate-structure -h` to get the help

### Post-treatment scripts

- `gofr.c`:
  - C code to compute the radial distribution function from a LAMMPS dump file.
  - Compile with `gcc gofr.c -o GofR -lm`
  - Usage: `gofr -h` to get the help
- `xconvert`:
  - Conversion from and to VASP, N2P2 and LAMMPS
  - Usage: `python xconvert -h` to get the help
- `xGDOS`:
  - python code to read a LAMMPS dump file containing atomic velocities and compute the GDOS
  - Usage: `python xGDOS -h` to get the help
- `xLAMMPStoNNP`:
  - Convert and concatenate many dump files into a single file to use with N2P2. Also look for structure generating extrapolation warnings and store them apart.
  - Usage: `python xLAMMPStoNNP -h` to get the help
- `xOUTCARtoLAMMPS`:
  - Convert an OUTCAR trajectory file into a LAMMPS dump file
  - Usage: `python xOUTCARtoLAMMPS -h` to get the help
- `xplotLAMMPSlog`:
  - Plot a LAMMPS log file
  - Usage: `python xplotLAMMPSlog -h` to get the help
- `xprepareDPdata`:
  - Prepare the data for a DeepMD potential training from a n2p2 data file
  - Usage: `python xprepareDPdata -h` to get the help
- `xreadLAMMPSlog`:
  - Read a LAMMPS log file and extract the thermodynamic properties, prints to stdout
  - Usage: `python xreadLAMMPSlog -h` to get the help
- `xSTM`:
  - Compute an STM image from a CHGCAR or PARCHGCAR file
  - Usage: `python xSTM -h` to get the help

### Adaptive training scripts

You **will** need to adapt these scripts to your own cluster and problem...
Especially the `xjobadaptive` and `adaptive_learning/SlurmJob.py` scripts where some paths and cluster configuration are hardcoded.

- `adaptive_learning/adaptive_training.py`:
  - Script using the following classes to perform an adaptive training of a NNP and distribute jobs on the fly on a SLURM cluster
- `adaptive_learning/AdaptiveTraining.py`:
  - Class `AdaptiveTraining` to perform an adaptive training of a NNP
- `adaptive_learning/Cluster.py`:
  - Class `Cluster` to help distributing jobs on a cluster
- `adaptive_learning/SlurmJob.py`:
  - Class `SlurmJob` to help launch and follow jobs on a SLURM cluster
- `adaptive_learning/functions.py`:
  - Some user-defined functions
- `adaptive_learning/xjobadaptive`:
  - Launch an adaptive training of a NNP on a SLURM cluster. **You need to edit the script to set the correct paths and cluster definition for you**.

## How to cite

Please cite the following articles if you use any of the files in this archive:

*"Neural network approach for a rapid prediction of metal-supported borophene properties"*, P. Mignon, A. Allouche, N.R. Innis, and C. Bousige, [*J. Am. Chem. Soc.* (2023)](https://doi.org/10.1021/jacs.3c11549).

```bibtex
@article{Mignon2023,
  title = {Neural Network Approach for a Rapid Prediction of Metal-Supported Borophene Properties},
  author = {Pierre Mignon and Abdul-Rahman Allouche and Neil Richard Innis and Colin Bousige},
  journal = {Journal of the American Chemical Society},
  year = {2023},
  doi = {10.1021/jacs.3c11549},
  volume = {145},
  number = {50},
  pages = {27857-27866}
}
```

## Author

[Colin BOUSIGE](mailto:colin.bousige@cnrs.fr), CNRS, [Laboratoire des Multimatériaux et Interfaces](http://lmi.cnrs.fr), Lyon, France

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This work was supported by the French National Research Agency grant ANR-21-CE09-0001-01.
