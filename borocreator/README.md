# BoroCreator

Interactive Streamlit app to build borophene structures, visualize them in 2D/3D, and export structures.

The app supports two workflows:

- **Manual structure creation** from borophene and slab parameters.
- **Importing external structures**, including multi-frame files, then selecting the active frame.

It also includes a **DeePMD-based relaxation step** in the Structure tab.

---

## Online Use

Go to the [online app](https://borocreator.streamlit.app/)

---

## Local Use (recommended for large structures and using MD relaxation/optimization features)

### 📋 Requirements

- **Python >=3.10,<3.12** (required for compatibility with dependencies).
- **[uv](https://docs.astral.sh/uv/)** (recommended for dependency management).

### Install Dependencies

#### CPU-Only (Works on All Platforms)

```bash
uv venv .venv --python 3.10
uv sync
```

- Works on **macOS (M1/M2/M3), Linux, and Windows**.
- Uses **CPU-only DeePMD-kit**.

#### GPU Support (Linux/Windows Only)

If you have a **CUDA-compatible GPU** (Linux/Windows):

```bash
uv venv .venv --python 3.10
uv sync --group gpu
```

- **Prerequisites**:
  - Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn).
  - Verify GPU detection with `nvidia-smi`.

### 🏃 Run the App

#### Option 1: Activate the Virtual Environment

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell or CMD)
.venv\Scripts\activate

# Run the app
streamlit run BoroCreator.py
```

#### Option 2: Use `uv run` (No Activation Needed)

```bash
uv run streamlit run BoroCreator.py
```

- Automatically uses the virtual environment created by `uv`.

After running, open the local URL shown by Streamlit (typically **[http://localhost:8501](http://localhost:8501)**).

---

## ✨ Features

### What You Can Do

- **Build borophene lattices** from predefined or custom hole patterns.
- **Add and tune metal slabs** (Ag, Al, Au, Cu, Pt, Ni, Ir, Si) with custom lattice parameters.
- **Create multilayered borophene** with AA/AB stacking.
- **Import structures** from files (supports multi-frame files).
- **Visualize structures** in **2D (Plotly) or 3D (py3Dmol)**.
- **Run NVT quenching** or **optimization** using DeePMD.
- **Export structures** in multiple formats (VASP, XYZ, LAMMPS, PDB).

---

## 📥 Import Workflow

1. In the **Structure tab**, open the **"Import structure from file"** expander.
2. Upload a supported structure file (e.g., `POSCAR`, `.xyz`, `.traj`, `.cif`, `.pdb`).
3. If the file contains **multiple structures**, use the **"Structure index"** slider to select the desired frame.

**Supported formats**: All file types handled by [ASE](https://wiki.fysik.dtu.dk/ase/ase/io/io.html), including:

- VASP (`POSCAR`, `CONTCAR`)
- XYZ (`.xyz`)
- LAMMPS (`.data`, `.lammpstrj`)
- PDB (`.pdb`)
- CIF (`.cif`)

---

## ⚙️ DeePMD Relaxation Notes

- The app expects the **DeePMD model** at `potential/graph.pb`.
  - Place your model file in this directory before running relaxation.
- **Relaxation parameters**:
  - Uses a **1 fs timestep**.
  - User-defined **start/end temperatures** and **number of steps**.
- **Constraints**:
  - The **lowest metal layer is fixed** automatically during relaxation.
  - Center-of-mass motion is constrained.

---

## 📤 Export

The **Output** section supports exporting structures in the following formats:

| Format      | File Extension | Description                     |
|-------------|----------------|---------------------------------|
| VASP        | `.POSCAR`      | VASP input format.              |
| XYZ         | `.xyz`         | Standard XYZ coordinates.       |
| LAMMPS      | `.data`        | LAMMPS data file.               |
| PDB         | `.pdb`         | Protein Data Bank format.       |

**Note**: For VASP output, you can optionally **fix the bottom N layers** of the metal slab.

---

## ❓ Troubleshooting

### Common Issues and Fixes

| Issue | Solution |
|-------|----------|
| **DeePMD import fails** | Ensure `deepmd-kit` is installed. Use the correct platform-specific group (e.g., `mac` for macOS, `linux-gpu` for Linux with GPU). |
| **TensorFlow errors on macOS** | Use the `mac` group to install `tensorflow-macos`. Avoid `tensorflow` or `tensorflow-gpu`. |
| **CUDA errors on Linux/Windows** | Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). Verify with `nvidia-smi`. |
| **`py3Dmol` fails to render** | Ensure your browser supports **WebGL** (Chrome, Firefox, Edge recommended). |
| **Large 3D structures are slow** | Reduce the number of repeated cells (`Nrepx`, `Nrepy`) or disable bonds in the 3D view. |
| **MD quench is slow** | Use your own **HPC facilities** for large structures. The built-in quench is for quick tests. |
| **File import fails** | Verify the file format is supported by ASE. Check the file path and permissions. |

---

### Dependency Conflicts
If you encounter conflicts:

1. **Delete the virtual environment**:
   ```bash
   rm -rf .venv  # Linux/macOS
   rmdir /s /q .venv  # Windows
   ```
2. **Recreate it**:
   ```bash
   uv venv .venv --python 3.10
   uv sync --group <your-group>
   ```

---

## 📁 Project Layout

```bash
BoroCreator/
├── BoroCreator.py          # Main Streamlit app
├── resources/              # Helper modules
│   ├── generator.py        # Structure generation tools
│   └── read_write.py       # File I/O helpers (e.g., LAMMPS, PDB)
├── potential/
│   └── graph.pb            # DeePMD model (required for relaxation)
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

---
## 🤝 Contributing

Found a bug or have a feature request? Open an issue or submit a pull request at:
🔗 [https://lmi.cnrs.fr/author/colin-bousige/](https://lmi.cnrs.fr/author/colin-bousige/)

---
## 📜 License

This project is developed by [Colin Bousige](https://lmi.cnrs.fr/author/colin-bousige/) at **CNRS**.
For support, requests, or bug reports, contact the author via the link above.
