# run with "streamlit run BoroCreator.py"

import streamlit as st
import py3Dmol
from urllib.parse import quote
import threading
import tempfile
import sys
import hashlib
from io import StringIO
import pandas as pd
from scipy.spatial import distance
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from resources.generator import *
from resources.read_write import boron_bond_pairs, write_lammps_traj, write_pdb
from fractions import Fraction
import os
import numpy as np
import itertools
from ase import *
from ase import units
from ase.io import read as ase_read, write
from ase.constraints import FixAtoms, FixCom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize.sciopt import SciPyFminBFGS

cpu_count = os.cpu_count() or 1
default_omp_threads = max(1, cpu_count)
default_inter_threads = max(1, cpu_count)

os.environ.setdefault("OMP_NUM_THREADS", str(default_omp_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(default_omp_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(default_omp_threads))
os.environ.setdefault("DP_INTRA_OP_PARALLELISM_THREADS",
                      str(default_omp_threads))
os.environ.setdefault("DP_INTER_OP_PARALLELISM_THREADS",
                      str(default_inter_threads))
os.environ.setdefault("TF_INTRA_OP_PARALLELISM_THREADS",
                      str(default_omp_threads))
os.environ.setdefault("TF_INTER_OP_PARALLELISM_THREADS",
                      str(default_inter_threads))

try:
    from deepmd.calculator import DP  # type: ignore[import-not-found]
except Exception:
    DP = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def atoms_signature(atoms):
    """Return a stable signature for an Atoms object to invalidate cached quenchs."""
    digest = hashlib.sha1()
    digest.update("".join(atoms.get_chemical_symbols()).encode("utf-8"))
    digest.update(np.round(atoms.positions, 6).astype(np.float64).tobytes())
    digest.update(np.round(atoms.cell.array, 6).astype(np.float64).tobytes())
    return digest.hexdigest()


def deepmd_runtime_summary():
    """Return a short runtime summary for DeePMD CPU/GPU and thread usage."""
    cpu_threads = os.environ.get(
        "DP_INTRA_OP_PARALLELISM_THREADS", str(cpu_count))
    inter_threads = os.environ.get(
        "DP_INTER_OP_PARALLELISM_THREADS", str(cpu_count))

    if DP is None:
        backend_label = "DeePMD unavailable"
    else:
        gpu_count = 0
        try:
            import tensorflow as tf  # type: ignore[import-not-found]

            gpu_count = len(tf.config.list_physical_devices("GPU"))
        except Exception:
            gpu_count = 0

        if gpu_count > 0:
            backend_label = f"GPU ({gpu_count} detected)"
        else:
            backend_label = "CPU"

    return f"""Running on: {backend_label} | Threads: intra-op {cpu_threads}, inter-op {inter_threads}"""


def showmol_iframe(view, height):
    """Render a py3Dmol view via Streamlit iframe without using deprecated components.v1.html."""
    html = view._make_html()
    html = html.replace(
        "<body>", '<body style="margin:0; overflow:hidden;">', 1)
    html = html.replace(
        "</head>",
        "<style>html, body { margin: 0; padding: 0; overflow: hidden; } ::-webkit-scrollbar { display: none; }</style></head>",
        1,
    )
    iframe_src = "data:text/html;charset=utf-8," + quote(html)
    st.iframe(iframe_src, width="stretch", height=height)


def manual_hole_picker_points(nx, ny):
    """Return sorted boron site positions for one borophene supercell before holes are removed."""
    aa = 1.7 * 2 * np.cos(np.pi / 6)
    bb = 1.7
    points = []
    for ix in range(int(nx)):
        for iy in range(int(ny)):
            ox = ix * aa
            oy = iy * bb
            points.append((ox, oy))
            points.append((ox + aa / 2, oy + bb / 2))
    points = np.array(points, dtype=float)
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


def repeat_points(points, cella, cellb, repx, repy, source_indices=None):
    """Repeat 2D points over a rectangular lattice and return repeated points with source indices."""
    repeated = []
    source_idx = []
    if source_indices is None:
        source_indices = np.arange(len(points), dtype=int)
    else:
        source_indices = np.array(source_indices, dtype=int)
    for ix in range(int(repx)):
        for iy in range(int(repy)):
            shift = np.array([ix * cella, iy * cellb], dtype=float)
            shifted = points + shift
            repeated.append(shifted)
            source_idx.extend(source_indices.tolist())
    if not repeated:
        return points.copy(), source_indices.copy()
    return np.vstack(repeated), np.array(source_idx, dtype=int)


def manual_hole_picker_points_with_base(nx, ny, lattice_repeatx=1, lattice_repeaty=1):
    """Return boron positions for the structural lattice and the corresponding base-cell hole indices."""
    base_points = manual_hole_picker_points(nx, ny)
    points, base_indices = repeat_points(
        base_points,
        int(nx) * 1.7 * 2 * np.cos(np.pi / 6),
        int(ny) * 1.7,
        lattice_repeatx,
        lattice_repeaty,
        source_indices=np.arange(len(base_points), dtype=int),
    )
    return points, base_indices


def metal_display_name(symbol):
    names = {
        "Ag": "Silver",
        "Al": "Aluminum",
        "Au": "Gold",
        "Cu": "Copper",
        "Pt": "Platinum",
        "Ni": "Nickel",
        "Ir": "Iridium",
        "Si": "Silicon",
        "Metal": "Metal",
        "": "Metal",
    }
    return names.get(symbol, str(symbol))


def manual_hole_picker_figure(
    points,
    selected_holes,
    show_bonds,
    metal_positions=None,
    metal_color="lightgrey",
    metal_label="Metal",
    nx=1,
    ny=1,
    repx=1,
    repy=1,
    base_indices=None,
    point_scale=1.0,
):
    """Build a selectable 2D lattice view used to pick boron holes interactively."""
    selected = {int(i) for i in selected_holes}
    if base_indices is None:
        base_indices = np.arange(len(points), dtype=int)
    else:
        base_indices = np.array(base_indices, dtype=int)
    marker_color = ["#f4a3b4" for _ in range(len(points))]
    base_size_sel = max(2.0, 12.0 * float(point_scale))
    base_size_unsel = max(2.0, 10.0 * float(point_scale))
    marker_size = [
        base_size_sel if int(base_indices[i]) in selected else base_size_unsel
        for i in range(len(points))
    ]
    marker_opacity = [
        0.18 if int(base_indices[i]) in selected else 0.95 for i in range(len(points))
    ]

    fig = go.Figure()

    if metal_positions is not None and len(metal_positions) > 0:
        fig.add_trace(
            go.Scatter(
                x=metal_positions[:, 0],
                y=metal_positions[:, 1],
                mode="markers",
                marker={
                    "size": max(2.0, 13.0 * float(point_scale)),
                    "color": metal_color,
                    "opacity": 0.65,
                    "symbol": "circle",
                    "line": {"color": "black", "width": 0.8},
                },
                hovertemplate=f"{metal_label}<br>x=%{{x:.2f}} A<br>y=%{{y:.2f}} A<extra></extra>",
                hoverlabel={"bgcolor": metal_color,
                            "font": {"color": "black"}},
                showlegend=False,
            )
        )

    if show_bonds and len(points) > 1:
        kept_idx = [
            i for i in range(len(points)) if int(base_indices[i]) not in selected
        ]
        if len(kept_idx) > 1:
            kept = points[kept_idx]
            pairs = np.array(list(itertools.combinations(range(len(kept)), 2)))
            dr = distance.pdist(kept)
            kept_pairs = pairs[dr < 2.2]
            if len(kept_pairs) > 0:
                x_lines = []
                y_lines = []
                for i, j in kept_pairs:
                    x_lines.extend([kept[i, 0], kept[j, 0], None])
                    y_lines.extend([kept[i, 1], kept[j, 1], None])
                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode="lines",
                        line={
                            "color": "#f4a3b4",
                            "width": max(1.0, 2.0 * float(point_scale)),
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker={
                "size": marker_size,
                "color": marker_color,
                "opacity": marker_opacity,
                "line": {"color": "black", "width": 0.5},
            },
            customdata=base_indices,
            hovertemplate="B index %{pointNumber}<br>x=%{x:.2f} A<br>y=%{y:.2f} A<extra></extra>",
            hoverlabel={"bgcolor": "#f4a3b4", "font": {"color": "black"}},
            showlegend=False,
        )
    )

    selection_trace_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker={
                "size": [max(8.0, s + 4.0) for s in marker_size],
                "color": "rgba(0,0,0,0.001)",
                "opacity": 0.001,
                "line": {"width": 0},
            },
            customdata=base_indices,
            hovertemplate="B index %{pointNumber}<br>x=%{x:.2f} A<br>y=%{y:.2f} A<extra></extra>",
            hoverlabel={"bgcolor": "#f4a3b4", "font": {"color": "black"}},
            showlegend=False,
        )
    )

    aa = 1.7 * 2 * np.cos(np.pi / 6)
    bb = 1.7
    # The dashed unit-cell box reflects the structural lattice only,
    # not the display-only replication (Nrepx/Nrepy).
    cella = int(nx) * aa
    cellb = int(ny) * bb
    fig.add_trace(
        go.Scatter(
            x=[0, cella, cella, 0, 0],
            y=[0, 0, cellb, cellb, 0],
            mode="lines",
            line={"color": "#cfcfcf", "dash": "dash", "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        dragmode="select",
        clickmode="event+select",
        hovermode="closest",
        meta={"selection_trace_index": selection_trace_index},
        margin={"l": 0, "r": 0, "t": 6, "b": 6},
        height=480,
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def boron_indices_from_full_reference(plot_struct, full_boron_struct, repx=1, repy=1):
    """Map displayed boron atoms to original no-hole boron indices used by listholes."""
    boron = plot_struct[plot_struct.get_atomic_numbers() == 5]
    if len(boron) == 0:
        return np.array([], dtype=int)
    if full_boron_struct is None or len(full_boron_struct) == 0:
        return np.arange(len(boron), dtype=int)

    full_rep = full_boron_struct.repeat((int(repx), int(repy), 1))
    full_rep.set_cell(full_boron_struct.cell)
    full_index_values = np.tile(
        np.arange(len(full_boron_struct), dtype=int), int(repx) * int(repy)
    )

    position_to_indices = {}
    for idx, pos in enumerate(full_rep.positions):
        key = tuple(np.round(pos, 6))
        position_to_indices.setdefault(key, []).append(
            int(full_index_values[idx]))

    mapped = np.full(len(boron), -1, dtype=int)
    missing = []
    for i, pos in enumerate(boron.positions):
        key = tuple(np.round(pos, 6))
        candidates = position_to_indices.get(key)
        if candidates:
            mapped[i] = candidates.pop(0)
        else:
            missing.append(i)

    # Fallback for relaxed/distorted structures: nearest-neighbor matching in 3D.
    if missing:
        dmat = distance.cdist(boron.positions[missing], full_rep.positions)
        nearest = np.argmin(dmat, axis=1)
        for miss_idx, ref_idx in zip(missing, nearest):
            mapped[miss_idx] = int(full_index_values[ref_idx])

    return mapped


def vacancy_positions_from_full_reference(
    plot_struct, full_boron_struct, repx=1, repy=1
):
    """Return vacancy positions and original no-hole boron indices."""
    if full_boron_struct is None or len(full_boron_struct) == 0:
        return np.empty((0, 3), dtype=float), np.array([], dtype=int)

    full_rep = full_boron_struct.repeat((int(repx), int(repy), 1))
    full_rep.set_cell(full_boron_struct.cell)
    full_pos = np.round(full_rep.positions, 6)
    full_index_values = np.tile(
        np.arange(len(full_boron_struct), dtype=int), int(repx) * int(repy)
    )

    boron = plot_struct[plot_struct.get_atomic_numbers() == 5]
    occ_counts = {}
    for pos in np.round(boron.positions, 6):
        key = tuple(pos)
        occ_counts[key] = occ_counts.get(key, 0) + 1

    vacancy_positions = []
    vacancy_indices = []
    for pos, base_idx in zip(full_pos, full_index_values):
        key = tuple(pos)
        count = occ_counts.get(key, 0)
        if count > 0:
            occ_counts[key] = count - 1
        else:
            vacancy_positions.append(pos)
            vacancy_indices.append(int(base_idx))

    if not vacancy_positions:
        return np.empty((0, 3), dtype=float), np.array([], dtype=int)
    return np.array(vacancy_positions, dtype=float), np.array(
        vacancy_indices, dtype=int
    )


def plotly_structure_2d_figure(
    plot_struct,
    metalchoice,
    show_bonds,
    point_scale,
    metal_label="Metal",
    boron_indices=None,
    vacancy_positions=None,
    vacancy_indices=None,
    plot_height=480,
    tight_cell_bounds=False,
):
    """Build the main 2D structure view with Plotly."""
    fig = go.Figure()
    boron = plot_struct[plot_struct.get_atomic_numbers() == 5]
    metal = plot_struct[plot_struct.get_atomic_numbers() != 5]

    if len(metal) > 0 and metalchoice != "":
        metal_color = collist.get(metalchoice, "lightgrey")
        fig.add_trace(
            go.Scatter(
                x=metal.positions[:, 0],
                y=metal.positions[:, 1],
                mode="markers",
                marker={
                    "size": 14 * point_scale,
                    "color": metal_color,
                    "opacity": 0.65,
                    "symbol": "circle",
                    "line": {"color": "black", "width": 0.8},
                },
                hovertemplate=f"{metal_label}<br>x=%{{x:.2f}} A<br>y=%{{y:.2f}} A<extra></extra>",
                hoverlabel={"bgcolor": metal_color,
                            "font": {"color": "black"}},
                showlegend=False,
            )
        )

    if show_bonds and len(boron) > 1:
        bond_pairs = boron_bond_pairs(boron, cutoff=2.2, include_pbc=False)
        if len(bond_pairs) > 0:
            x_lines = []
            y_lines = []
            for i, j in bond_pairs:
                x_lines.extend(
                    [boron.positions[i, 0], boron.positions[j, 0], None])
                y_lines.extend(
                    [boron.positions[i, 1], boron.positions[j, 1], None])
            fig.add_trace(
                go.Scatter(
                    x=x_lines,
                    y=y_lines,
                    mode="lines",
                    line={
                        "color": "#f4a3b4",
                        "width": max(1.0, 2.0 * float(point_scale)),
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if boron_indices is None or len(boron_indices) != len(boron):
        boron_indices = np.arange(len(boron), dtype=int)

    cella = plot_struct.cell[0, 0]
    cellb = plot_struct.cell[1, 1]
    fig.add_trace(
        go.Scatter(
            x=[0, cella, cella, 0, 0],
            y=[0, 0, cellb, cellb, 0],
            mode="lines",
            line={"color": "#cfcfcf", "dash": "dash", "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Keep all boron markers above the metal layer.
    if vacancy_positions is not None and len(vacancy_positions) > 0:
        if vacancy_indices is None or len(vacancy_indices) != len(vacancy_positions):
            vacancy_indices = np.arange(len(vacancy_positions), dtype=int)
        fig.add_trace(
            go.Scatter(
                x=vacancy_positions[:, 0],
                y=vacancy_positions[:, 1],
                mode="markers",
                customdata=np.array(vacancy_indices, dtype=int),
                marker={
                    "size": 10 * point_scale,
                    "color": "#f4a3b4",
                    "opacity": 0.2,
                    "line": {"color": "black", "width": 0.5},
                },
                hovertemplate="B index %{customdata}<br>x=%{x:.2f} A<br>y=%{y:.2f} A<extra></extra>",
                hoverlabel={"bgcolor": "#f4a3b4", "font": {"color": "black"}},
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=boron.positions[:, 0],
            y=boron.positions[:, 1],
            mode="markers",
            customdata=np.array(boron_indices, dtype=int),
            marker={
                "size": 10 * point_scale,
                "color": "#f4a3b4",
                "line": {"color": "black", "width": 0.5},
            },
            hovertemplate="B index %{customdata}<br>x=%{x:.2f} A<br>y=%{y:.2f} A<extra></extra>",
            hoverlabel={"bgcolor": "#f4a3b4", "font": {"color": "black"}},
            showlegend=False,
        )
    )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 6, "b": 6},
        height=int(plot_height),
        showlegend=False,
        plot_bgcolor="white",
    )
    if tight_cell_bounds:
        pad = max(0.15, 0.02 * min(float(cella), float(cellb)))
        fig.update_xaxes(
            visible=False, range=[-pad, float(cella) + pad], fixedrange=True
        )
        fig.update_yaxes(
            visible=False,
            scaleanchor="x",
            scaleratio=1,
            range=[-pad, float(cellb) + pad],
            fixedrange=True,
        )
    else:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


class RelaxationJob:
    def __init__(self, atoms, model_path, t_start, t_end, nsteps):
        self.atoms = atoms.copy()
        self.model_path = model_path
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.nsteps = int(nsteps)
        self.current_step = 0
        self.progress = 0
        self.status = "Queued"
        self.running = False
        self.done = False
        self.error = None
        self.stop_requested = False
        self.relaxed_atoms = None
        self.current_temperature = None
        self.current_energy = None
        self.temperature_history = []
        self.energy_history = []
        self.step_history = []
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".traj")
        self.traj_path = tmp.name
        tmp.close()


def get_deepmd_model_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "potential", "graph.pb"
    )


def build_lammpstrj_cache(traj_path):
    sig = (traj_path, os.path.getsize(traj_path), os.path.getmtime(traj_path))
    cache = st.session_state.get("stm_relax_lammpstrj_cache")
    if cache and cache.get("sig") == sig:
        return cache

    lammps_frames = ase_read(traj_path, index=":")
    if not isinstance(lammps_frames, list):
        lammps_frames = [lammps_frames]
    lammps_buffer = StringIO()
    write_lammps_traj(lammps_frames, outfile=lammps_buffer)
    cache = {
        "sig": sig,
        "data": lammps_buffer.getvalue(),
    }
    st.session_state.stm_relax_lammpstrj_cache = cache
    return cache


def refresh_lammpstrj_cache(traj_path):
    st.session_state.pop("stm_relax_lammpstrj_cache", None)
    st.session_state.pop("stm_relax_lammpstrj_error", None)
    try:
        build_lammpstrj_cache(traj_path)
    except Exception as exc:
        st.session_state.stm_relax_lammpstrj_error = str(exc)


def prepare_deepmd_atoms(atoms, model_path):
    if DP is None:
        raise ImportError("deepmd.calculator.DP could not be imported")

    prepared = atoms.copy()
    prepared.calc = DP(model=model_path)

    metal_mask = np.array(prepared.get_atomic_numbers()) != 5
    if np.any(metal_mask):
        metal_z = prepared.positions[metal_mask, 2]
        bottom_z = np.min(metal_z)
        fixed_mask = metal_mask & np.isclose(
            prepared.positions[:, 2], bottom_z, atol=0.2
        )
        if np.any(fixed_mask):
            prepared.set_constraint(FixAtoms(mask=fixed_mask))

    return prepared


def run_quench_job(job: RelaxationJob):
    try:
        from ase.io.trajectory import Trajectory

        relaxed = prepare_deepmd_atoms(job.atoms, job.model_path)

        constraints = list(relaxed.constraints) if relaxed.constraints else []
        constraints.append(FixCom())
        relaxed.set_constraint(constraints)

        MaxwellBoltzmannDistribution(relaxed, temperature_K=job.t_start)
        Stationary(relaxed)

        dyn = Langevin(
            relaxed,
            timestep=1.0 * units.fs,
            temperature_K=job.t_start,
            friction=0.02,
            fixcm=False,
        )

        job.running = True
        save_stride = max(1, job.nsteps // 10)
        ui_stride = max(1, job.nsteps // 10)
        with Trajectory(job.traj_path, "w", relaxed) as traj:
            traj.write(relaxed)
            for step in range(job.nsteps):
                if job.stop_requested:
                    job.status = f"Stopped at step {step}/{job.nsteps}"
                    break
                target_temp = job.t_start + (job.t_end - job.t_start) * (
                    step / max(1, job.nsteps - 1)
                )
                dyn.set_temperature(temperature_K=target_temp)
                dyn.run(1)
                job.current_temperature = float(relaxed.get_temperature())
                job.current_energy = float(relaxed.get_potential_energy())
                job.step_history.append(step + 1)
                job.temperature_history.append(job.current_temperature)
                job.energy_history.append(job.current_energy)
                # Save only every 10% of the total MD time, and always save the final step.
                if (step + 1) % save_stride == 0 or step == job.nsteps - 1:
                    traj.write(relaxed)

                job.current_step = step + 1
                if (step + 1) % ui_stride == 0 or step == job.nsteps - 1:
                    job.progress = int(100 * (step + 1) / job.nsteps)
                    job.status = f"Step {step + 1}/{job.nsteps} - target temperature: {target_temp:.2f} K"

        # Always display the final structure from the saved trajectory.
        try:
            job.relaxed_atoms = ase_read(job.traj_path, index=-1)
        except Exception:
            job.relaxed_atoms = relaxed
        if not job.stop_requested:
            job.status = "Relaxation complete."
        job.done = True
    except Exception as exc:
        job.error = str(exc)
        job.status = f"Relaxation failed: {exc}"
        job.done = True
    finally:
        job.running = False


def writeout(struct, extension_out, source_name=None):
    """
    Write output file in the selected format in the current directory
    """
    if source_name is not None:
        name = os.path.splitext(os.path.basename(source_name))[0]
    else:
        name = "Boro" + str(nx) + "," + str(ny)
        if ",".join(map(str, listholes)) != "":
            name += "_" + ",".join(map(str, sorted(listholes)))
        if metalchoice != "":
            name += f"-{NZ}{metalchoice}_{surfchoice}"
    extlist = {"VASP": ".POSCAR", "xyz": ".xyz",
               "LAMMPS": ".data", "PDB": ".pdb"}
    formlist = {
        "VASP": "vasp",
        "xyz": "xyz",
        "LAMMPS": "lammps-data",
        "PDB": "proteindatabank",
    }
    name += extlist[extension_out]
    if extension_out == "VASP":
        with Capturing() as outfile:
            write(sys.stdout, struct,
                  format=formlist[extension_out], vasp5=True)
    elif extension_out == "LAMMPS":
        with Capturing() as outfile:
            write_lammps(None, struct, atom_style="atomic", units="real")
    elif extension_out == "PDB":
        with Capturing() as outfile:
            sys.stdout.write(write_pdb(struct, name=name))
    else:
        with Capturing() as outfile:
            write(sys.stdout, struct, format=formlist[extension_out])
    return ("\n".join(outfile), name)


dico_predef = {
    "\u03b1": (3, 3, [0, 10]),  # alpha
    "\u03b11": (2, 4, [0, 10]),  # alpha1
    "\u03b12": (
        16, 16, [0, 18, 37, 55, 74, 92, 111, 7, 25, 44, 62, 65, 83, 102, 120, 139, 157, 113, 132, 150, 169, 187, 206, 160,
                 178, 197, 215, 234, 252, 208, 227, 245, 264, 282, 301, 319, 273, 292, 310, 329, 347, 366, 322,
                 368, 340, 359, 377, 396, 414, 387, 405, 424, 442, 461, 479, 417, 435, 482, 500, 454, 472, 491, 509, 271],
    ),  # alpha2
    "\u03b14": (
        9,
        9,
        [0, 11, 23, 34, 37, 48, 60, 71, 97, 85,
            74, 99, 111, 136, 122, 148, 134, 159],
    ),  # alpha4
    "\u03b15": (2, 6, [0, 15]),  # alpha5
    "\u03b16": (2, 3, [0]),  # alpha6
    "\u03b17": (
        12,
        8,
        [0, 4, 25, 29, 55, 51, 76, 102, 98, 72, 127, 123, 149, 174, 145, 170],
    ),  # alpha7
    "\u03b43": (1, 3, [0, 4]),  # delta3
    "\u03b44": (1, 2, [0]),  # delta4
    "\u03b45": (
        7,
        7,
        [0, 16, 11, 32, 52, 73, 93, 36, 57, 77, 48, 27, 68, 89],
    ),  # delta5
    "\u03b46": (2, 2, []),  # delta6
    "\u03c72": (3, 6, [0, 8, 16, 18, 26, 34]),  # chi2
    "\u03c73": (1, 5, [0, 7]),  # chi3
    "\u03c74": (6, 6, [0, 8, 17, 19, 30, 28, 39, 50, 61, 69, 58, 47]),  # chi4
    "\u03b24": (5, 3, [0, 6, 16]),  # beta4
    "\u03b25": (5, 3, [0, 6, 16, 22]),  # beta5
    "\u03b28": (1, 9, [0, 13]),  # beta8
    "\u03b210": (1, 4, [0]),  # beta10
    "\u03b211": (
        12, 8, [0, 4, 19, 28, 43, 53, 68, 77, 92, 102, 117, 126, 141, 151, 166,
                175, 190, 24, 23, 47, 49, 64, 73, 88, 98, 113, 122, 137, 147, 162, 171, 186],
    ),  # beta11
    "\u03b212": (1, 3, [0]),  # beta12
    "\u03b213": (2, 3, [0, 4]),  # beta13,
}

st.set_page_config(
    page_title="Borophene Creator",
    page_icon=":hammer_and_pick:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://lmi.cnrs.fr/author/colin-bousige/",
        "Report a bug": "https://lmi.cnrs.fr/author/colin-bousige/",
        "About": """
        ## Borophene Creator
        Version date 2026-06-18.

        This app was made by [Colin Bousige](https://lmi.cnrs.fr/author/colin-bousige/). Contact me for support, requests, or to signal a bug.
        """,
    },
)
padding_top = 0
css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
"""

st.markdown(
    f"""
    <style>
    .stTabs [data-baseweb="tab-list"] {{
		gap: 10px;
    }}

	.stTabs [data-baseweb="tab"] {{
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 5px 5px 0px 0px;
		gap: 1px;
		padding: 10px;
    }}

	.stTabs [aria-selected="false"] {{
  		background-color: #FFFFFF;
	}}

    .stTabs [aria-selected="false"] [data-testid="stMarkdownContainer"] p {{
        color: #000000;
    }}
 
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size:1.7rem;
        font-weight: bold;
    }}
    [data-testid="stSidebar"] {{
        display: none;
    }}
    [data-testid="collapsedControl"] {{
        display: none;
    }}
    .block-container{{
            padding-top: {padding_top}rem;
        }}
    thead tr th:first-child {{display:none}}
        tbody th {{display:none}}
    </style>
    """,
    unsafe_allow_html=True,
)


st.write("""
<br>

# Borophene Creator
""")


# # # # # # # # # # # # # # # # # #
# Borophene Structure
# # # # # # # # # # # # # # # # # #
imported_struct = None
imported_name = None

controls_col, view_col, outcol = st.columns((2.2, 3.8, 1.4))

with controls_col.expander(
    ":material/massage: Manual structure creation", expanded=False
):
    with st.container(border=False):
        with st.expander(":material/grid_4x4: Borophene lattice", expanded=False):
            predef_options = ["Manual definition"] + \
                [k for k in dico_predef.keys()]
            predef_default_idx = (
                predef_options.index(
                    "\u03b1") if "\u03b1" in predef_options else 0
            )
            predef = st.selectbox(
                "Predefined structures",
                predef_options,
                key="predef",
                index=predef_default_idx,
                help="Pick a known borophene allotrope template or choose Manual definition.",
            )

            previous_predef = st.session_state.get(
                "stm_previous_manual_predef")
            if previous_predef is None:
                st.session_state.stm_previous_manual_predef = predef
            elif previous_predef != predef:
                st.session_state.stm_previous_manual_predef = predef
                st.session_state.pop(
                    "stm_active_structure_position_overrides", None
                )
                for state_key in list(st.session_state.keys()):
                    if str(state_key).startswith("atomic_positions_editor_"):
                        st.session_state.pop(state_key, None)

            if "Nx" not in st.session_state:
                st.session_state.Nx = 1
            if "Ny" not in st.session_state:
                st.session_state.Ny = 3

            col1, col2 = st.columns((1, 1))
            if predef != "Manual definition":
                st.session_state.Nx = dico_predef[predef][0]
                st.session_state.Ny = dico_predef[predef][1]
                st.session_state.listholes = dico_predef[predef][2]
            nx = col1.number_input(
                "Nx",
                min_value=1,
                max_value=None,
                key="Nx",
                help="Number of unit cells along x in the borophene lattice.",
            )
            ny = col2.number_input(
                "Ny",
                min_value=1,
                max_value=None,
                key="Ny",
                help="Number of unit cells along y in the borophene lattice.",
            )
            st.multiselect(
                "Vacancies",
                range(2 * nx * ny),
                key="listholes",
                help="Indices of boron sites removed from the initial lattice.",
            )
            listholes = list(st.session_state.listholes)
            if go is None:
                st.warning(
                    "Interactive picking needs plotly. Install plotly and restart the app."
                )
            else:
                st.caption("Choose vacancies based on boron indices.")

            col1, col2 = st.columns((1, 1))
            repeatx = col1.number_input(
                "Repeat x",
                min_value=1,
                max_value=None,
                value=1,
                key="repeatx",
                help="Repeat the full structure along x after construction.",
            )
            repeaty = col2.number_input(
                "Repeat y",
                min_value=1,
                max_value=None,
                value=1,
                key="repeaty",
                help="Repeat the full structure along y after construction.",
            )
            size_min = st.number_input(
                "Min size of the box [Å]",
                min_value=0.0,
                max_value=None,
                value=0.0,
                help="If >0, expand/repeat to reach at least this lateral box size.",
            )

        with st.expander(":material/splitscreen_top: Metal slab", expanded=False):
            left, right = st.columns((1, 1))
            metalchoice = left.selectbox(
                "Metal",
                ("", "Ag", "Al", "Au", "Cu", "Pt", "Ni", "Ir", "Si"),
                index=1,
                help="Metal substrate element. Empty means no metal slab.",
            )
            surfchoice = right.selectbox(
                "Surface",
                ("fcc111", "fcc110", "fcc100", "fcc211", "diamond100"),
                help="Crystallographic orientation used to build the metal surface.",
            )
            NZ = left.number_input(
                "N layers",
                min_value=0,
                max_value=None,
                value=3,
                help="Number of metal layers in the slab.",
            )
            vdwdist = left.number_input(
                "Van der Waals distance [Å]",
                min_value=0.0,
                max_value=None,
                value=2.5,
                help="Initial borophene-substrate separation.",
            )
            default_metal_a = (
                float(metals[metalchoice]
                      ) if metalchoice in metals else 4.0
            )
            metal_name = metalchoice if metalchoice else "Metal"
            metal_lattice_a = right.number_input(
                f"{metal_name} parameter a [Å]",
                min_value=0.1,
                max_value=None,
                value=default_metal_a,
                step=0.01,
                format="%.4f",
                disabled=(metalchoice == ""),
                help="Lattice parameter used to build the selected metal slab.",
            )
            vac = right.number_input(
                "Add vaccum layer on top [Å]",
                min_value=0.0,
                max_value=None,
                value=25.0,
                help="Vacuum thickness added above the system in z direction.",
            )

        with st.expander(":material/transform: Borophene Transformations", expanded=False):
            left, right = st.columns((1, 1))
            angle = left.selectbox(
                "Rotate borophene [˚]",
                (0, 90),
                index=0,
                key="angle",
                help="Rotate the borophene lattice in-plane before placing it on the slab.",
            )
            random = right.number_input(
                "Add random motion of max value [Å]",
                min_value=0.0,
                max_value=None,
                value=0.0,
                help="Random displacement amplitude applied to atoms for perturbation.",
            )
            shiftX = left.number_input(
                "Shift Borophene X [Å]",
                min_value=None,
                max_value=None,
                value=0.0,
                step=5.0,
                help="Translate borophene along x after construction.",
            )
            shiftY = right.number_input(
                "Shift Borophene Y [Å]",
                min_value=None,
                max_value=None,
                value=0.0,
                step=5.0,
                help="Translate borophene along y after construction.",
            )

        with st.expander(":material/circle_circle: Borophene island", expanded=False):
            left, right = st.columns((1, 1))
            island_size = left.number_input(
                "Borophene island size [Å]",
                min_value=0.0,
                max_value=None,
                value=0.0,
                step=5.0,
                help="If >0, crop borophene into finite islands of this characteristic size.",
            )
            island_shape = right.selectbox(
                "Borophene island shape",
                ("Circle", "Square", "Triangle", "Hexagon"),
                help="Geometric shape used for island carving.",
            )
            island_angle = left.number_input(
                "Rotate 1 island [˚]",
                min_value=0.0,
                max_value=90.0,
                value=0.0,
                step=5.0,
                key="island_angle",
                help="In-plane rotation applied to the first island.",
            )
            island_rotate = right.number_input(
                "Rotate 2 island [˚]",
                min_value=0.0,
                max_value=90.0,
                value=0.0,
                step=5.0,
                key="island_rotate",
                help="In-plane rotation applied to the second island.",
            )

        with st.expander(":material/stacks: Multilayered borophene", expanded=False):
            left, right = st.columns((1, 1))
            Nboro = left.number_input(
                "N of borophene layers",
                min_value=1,
                value=1,
                key="Nboro",
                help="Total number of stacked borophene layers.",
            )
            toplayer = right.selectbox(
                "Top layer",
                [""] + [k for k in islands.keys()],
                key="predefislands",
                help="Optional predefined geometry for the upper layer.",
            )
            TopLayer = None if toplayer == "" else islands[toplayer]
            randshifttop = left.number_input(
                "Random lateral shift of top layers",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Random x/y offset range applied to upper borophene layers.",
            )
            stacking = right.selectbox(
                "Stacking type",
                ("AA", "AB"),
                help="Relative registry between borophene layers.",
            )
            stackshiftx = left.number_input(
                "Shift x between layers [Å]",
                min_value=0.0,
                max_value=None,
                value=0.0,
                step=1.0,
                help="Deterministic x shift applied between consecutive layers.",
            )
            stackshifty = right.number_input(
                "Shift y between layers [Å]",
                min_value=0.0,
                max_value=None,
                value=0.0,
                step=1.0,
                help="Deterministic y shift applied between consecutive layers.",
            )

        with st.expander(":material/blur_on: Solubilized boron", expanded=False):
            left, right = st.columns((1, 1))
            NsolubleB = left.number_input(
                "Solubilzed B atoms",
                min_value=0,
                max_value=None,
                value=0,
                step=1,
                help="Insert this many boron atoms into subsurface/metal sites.",
            )
            dilate = right.number_input(
                "Metal dilatation",
                min_value=0.0,
                max_value=None,
                value=1.2,
                step=0.1,
                help="Expansion factor used when searching insertion sites in the metal.",
            )

with controls_col.expander(
    ":material/publish: Import structure from file", expanded=False
):
    if "import_uploader_nonce" not in st.session_state:
        st.session_state.import_uploader_nonce = 0
    uploader_key = (
        f"imported_structure_file_{st.session_state.import_uploader_nonce}"
    )

    upload_col, clear_col = st.columns((1.5, 1))
    with upload_col:
        uploaded_file = st.file_uploader(
            "Structure file",
            key=uploader_key,
            help="Supported names and formats: all files supported by ASE (e.g. POSCAR, .lammpstrj, .cif, .xyz, .vasp, .pdb, etc.). For multi-frame files, use the slider to select the structure to display.",
        )
    with clear_col:
        st.write("")
        st.write("")
        clear_import = st.button(
            "Clear structure",
            key="clear_imported_structure",
            icon=":material/delete:",
        )

    if clear_import:
        st.session_state.import_cache_key = None
        st.session_state.imported_structures_cache = []
        st.session_state.imported_type_cache = ""
        st.session_state.imported_name_cache = ""
        st.session_state.imported_structure_index = 0
        st.session_state.import_uploader_nonce += 1
        st.rerun()

    if uploaded_file is not None:
        if "import_cache_key" not in st.session_state:
            st.session_state.import_cache_key = None
        if "imported_structures_cache" not in st.session_state:
            st.session_state.imported_structures_cache = []
        if "imported_type_cache" not in st.session_state:
            st.session_state.imported_type_cache = ""
        if "imported_name_cache" not in st.session_state:
            st.session_state.imported_name_cache = ""

        file_bytes = uploaded_file.getvalue()
        upload_key = (uploaded_file.name, len(
            file_bytes), hash(file_bytes))

        if st.session_state.import_cache_key != upload_key:
            st.session_state.import_cache_key = upload_key
            st.session_state.imported_structure_index = 0
            st.session_state.plot_show_3d = True
            tmp_path = None
            try:
                suffix = f"_{os.path.basename(uploaded_file.name)}"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name
                imported_structures = ase_read(tmp_path, index=":")
                if not isinstance(imported_structures, list):
                    imported_structures = [imported_structures]
                imported_type = "ase"
                st.session_state.imported_structures_cache = imported_structures
                st.session_state.imported_type_cache = imported_type
                st.session_state.imported_name_cache = os.path.basename(
                    uploaded_file.name
                )
            except Exception as exc:
                st.error(f"Could not import structure: {exc}")
                st.session_state.imported_structures_cache = []
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        imported_structures = st.session_state.imported_structures_cache
        imported_type = st.session_state.imported_type_cache
        imported_name = st.session_state.imported_name_cache

        if len(imported_structures) > 1:
            selected_index = st.slider(
                "Structure index",
                min_value=0,
                max_value=len(imported_structures) - 1,
                value=min(
                    st.session_state.get("imported_structure_index", 0),
                    len(imported_structures) - 1,
                ),
                step=1,
                key="imported_structure_index",
            )
        else:
            selected_index = 0

        if len(imported_structures) > 0:
            imported_struct = imported_structures[selected_index].copy()
            st.write("##### Imported structure transforms")
            shift_col1, shift_col2 = st.columns((1, 1))
            import_boron_shift_x = shift_col1.number_input(
                "Shift boron X [Å]",
                value=0.0,
                step=1.0,
                key="import_boron_shift_x",
            )
            import_boron_shift_y = shift_col2.number_input(
                "Shift boron Y [Å]",
                value=0.0,
                step=1.0,
                key="import_boron_shift_y",
            )
            import_boron_rotate_z = st.number_input(
                "Rotate boron around Z [°]",
                value=0.0,
                step=1.0,
                key="import_boron_rotate_z",
            )
            scale_col1, scale_col2 = st.columns((1, 1))
            import_box_scale_x = scale_col1.number_input(
                "Stretch/compress box X [%]",
                value=0.0,
                step=0.5,
                key="import_box_scale_x",
            )
            import_box_scale_y = scale_col2.number_input(
                "Stretch/compress box Y [%]",
                value=0.0,
                step=0.5,
                key="import_box_scale_y",
            )
            st.caption(
                f"Displaying structure {selected_index + 1}/{len(imported_structures)} from {imported_name} ({imported_type})."
            )


collist = {
    "Ag": "silver",
    "Al": "lightgrey",
    "Au": "gold",
    "Cu": "#fdb07d",
    "Pt": "lightgrey",
    "Ni": "#fed776",
    "Ir": "lightgrey",
    "Si": "lightgrey",
    "Metal": "lightgrey",
}

use_imported_structure = imported_struct is not None
if use_imported_structure:
    struct = imported_struct.copy()
    scale_x = 1.0 + st.session_state.get("import_box_scale_x", 0.0) / 100.0
    scale_y = 1.0 + st.session_state.get("import_box_scale_y", 0.0) / 100.0
    if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
        scaled_cell = struct.cell.array.copy()
        scaled_cell[0] *= scale_x
        scaled_cell[1] *= scale_y
        struct.set_cell(scaled_cell, scale_atoms=True)
    shift_boron_x = st.session_state.get("import_boron_shift_x", 0.0)
    shift_boron_y = st.session_state.get("import_boron_shift_y", 0.0)
    rotate_boron_z = st.session_state.get("import_boron_rotate_z", 0.0)
    boron_mask = np.array(struct.get_atomic_numbers()) == 5
    if not np.isclose(rotate_boron_z, 0.0) and np.any(boron_mask):
        boron_pos = struct.positions[boron_mask].copy()
        boron_masses = struct.get_masses()[boron_mask]
        boron_com = np.average(boron_pos, axis=0, weights=boron_masses)
        theta = np.deg2rad(rotate_boron_z)
        c, s = np.cos(theta), np.sin(theta)
        centered_xy = boron_pos[:, :2] - boron_com[:2]
        rotated_xy = np.empty_like(centered_xy)
        rotated_xy[:, 0] = c * centered_xy[:, 0] - s * centered_xy[:, 1]
        rotated_xy[:, 1] = s * centered_xy[:, 0] + c * centered_xy[:, 1]
        struct.positions[boron_mask, :2] = rotated_xy + boron_com[:2]
    if not np.isclose(shift_boron_x, 0.0) or not np.isclose(shift_boron_y, 0.0):
        struct.positions[boron_mask, 0] += shift_boron_x
        struct.positions[boron_mask, 1] += shift_boron_y
    struct.wrap()
    base = struct.copy()
    structfull = struct[np.array(struct.get_chemical_symbols()) == "B"]
    metal_symbols = sorted(
        {symbol for symbol in struct.get_chemical_symbols() if symbol != "B"}
    )
    if len(metal_symbols) == 1 and metal_symbols[0] in collist:
        metalchoice = metal_symbols[0]
    elif len(metal_symbols) > 0:
        metalchoice = "Metal"
    else:
        metalchoice = ""
else:
    if ny == 1:
        metalchoice = ""

    max_sites_effective = 2 * int(nx) * int(ny)
    listholes = sorted({int(i) for i in listholes if 0 <=
                       int(i) < max_sites_effective})

    struct = create_structure(
        nx=nx,
        ny=ny,
        listholes=listholes,
        repeatx=repeatx,
        repeaty=repeaty,
        metalchoice=metalchoice,
        surfchoice=surfchoice,
        vdwdist=vdwdist,
        vac=vac,
        angle=angle,
        NZ=NZ,
        random=random,
        size_min=size_min,
        glimpse=False,
        dmin=1,
        dmax=0,
        shiftX=shiftX,
        shiftY=shiftY,
        Nboro=Nboro,
        island_size=island_size,
        island_shape=island_shape.lower(),
        island_angle=island_angle,
        island_rotate=island_rotate,
        toplayer=TopLayer,
        randshifttop=randshifttop,
        stacking=stacking,
        stackshiftx=stackshiftx,
        stackshifty=stackshifty,
        a=metal_lattice_a,
    )

    base = create_structure(
        nx=nx,
        ny=ny,
        listholes=listholes,
        repeatx=repeatx,
        repeaty=repeaty,
        metalchoice="None",
        surfchoice=surfchoice,
        vdwdist=vdwdist,
        vac=vac,
        angle=angle,
        NZ=NZ,
        random=random,
        size_min=size_min,
        glimpse=False,
        dmin=1,
        dmax=0,
        shiftX=shiftX,
        shiftY=shiftY,
        Nboro=Nboro,
        a=metal_lattice_a,
    )

    structfull = create_structure(
        nx=nx,
        ny=ny,
        listholes=[],
        repeatx=repeatx,
        repeaty=repeaty,
        metalchoice=metalchoice,
        surfchoice=surfchoice,
        vdwdist=vdwdist,
        vac=vac,
        angle=angle,
        NZ=NZ,
        random=random,
        size_min=size_min,
        glimpse=False,
        dmin=1,
        dmax=0,
        shiftX=shiftX,
        shiftY=shiftY,
        Nboro=Nboro,
        a=metal_lattice_a,
    )

    structfull = structfull[np.array(structfull.get_chemical_symbols()) == "B"]

    if NsolubleB > 0:
        struct = solubilize_boron(struct, NsolubleB, dilate)

source_sig = atoms_signature(struct)
if (
    st.session_state.get("stm_minimized_source_signature") == source_sig
    and st.session_state.get("stm_minimized_atoms") is not None
):
    struct = st.session_state.stm_minimized_atoms.copy()
    structfull = struct[np.array(struct.get_chemical_symbols()) == "B"]
else:
    st.session_state.pop("stm_minimized_atoms", None)
    st.session_state.pop("stm_minimized_source_signature", None)
    st.session_state.pop("stm_minimization_summary", None)

if len(structfull) > 0:
    sortedpos = np.lexsort(
        (structfull.positions[:, 1], structfull.positions[:, 0]))
    structfull.positions = structfull.positions[sortedpos]

info_boron = struct[struct.get_atomic_numbers() == 5]
info_metal = struct[struct.get_atomic_numbers() != 5]

display_struct = struct
display_structfull = structfull
active_structure_id = f"base:{source_sig}"
display_job = st.session_state.get("stm_relax_job")
if display_job is not None and os.path.isfile(display_job.traj_path):
    try:
        display_frames = ase_read(display_job.traj_path, index=":")
        if not isinstance(display_frames, list):
            display_frames = [display_frames]
        if len(display_frames) > 0:
            if display_job.done:
                if "stm_active_frame_index" not in st.session_state:
                    st.session_state.stm_active_frame_index = len(
                        display_frames) - 1
                frame_index = int(
                    st.session_state.get(
                        "stm_active_frame_index", len(display_frames) - 1
                    )
                )
                frame_index = max(0, min(frame_index, len(display_frames) - 1))
                display_struct = display_frames[frame_index].copy()
                active_structure_id = f"traj:{display_job.traj_path}:{frame_index}"
            else:
                # While running, always display the latest saved frame.
                display_struct = display_frames[-1].copy()
                active_structure_id = f"traj:{display_job.traj_path}:latest"
            display_struct.wrap()
            display_structfull = display_struct[
                np.array(display_struct.get_chemical_symbols()) == "B"
            ]
            display_structfull.wrap()
            if len(display_structfull) > 0:
                sortedpos_display = np.lexsort(
                    (
                        display_structfull.positions[:, 1],
                        display_structfull.positions[:, 0],
                    )
                )
                display_structfull.positions = display_structfull.positions[
                    sortedpos_display
                ]
    except Exception:
        display_struct = struct
        display_structfull = structfull

if "stm_active_structure_position_overrides" not in st.session_state:
    st.session_state.stm_active_structure_position_overrides = {}

stored_override = st.session_state.stm_active_structure_position_overrides.get(
    active_structure_id
)
if stored_override is not None:
    override_applied = False
    if isinstance(stored_override, dict):
        override_indices = np.asarray(
            stored_override.get("indices", []), dtype=int)
        override_positions = np.asarray(
            stored_override.get("positions", []), dtype=float
        )
        valid_idx = (
            override_indices.ndim == 1
            and len(override_indices) > 0
            and len(np.unique(override_indices)) == len(override_indices)
            and np.all(override_indices >= 0)
            and np.all(override_indices < len(display_struct))
        )
        if valid_idx and override_positions.shape == (len(override_indices), 3):
            display_struct = display_struct[override_indices].copy()
            display_struct.positions[:] = override_positions
            override_applied = True
    else:
        # Backward compatibility with previous position-only overrides.
        override_positions = np.asarray(stored_override, dtype=float)
        if override_positions.shape == (len(display_struct), 3):
            display_struct.positions[:] = override_positions
            override_applied = True

    if override_applied:
        display_struct.wrap()
        display_structfull = display_struct[
            np.array(display_struct.get_chemical_symbols()) == "B"
        ]
        display_structfull.wrap()
        if len(display_structfull) > 0:
            sortedpos_display = np.lexsort(
                (display_structfull.positions[:, 1],
                 display_structfull.positions[:, 0])
            )
            display_structfull.positions = display_structfull.positions[
                sortedpos_display
            ]


# # # # # # # # # # # # # # # # # #
# Plotting
# # # # # # # # # # # # # # # # # #
with view_col:
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns((1, 1, 1.5, 1.5, 1.5))
    show_3d = pcol1.checkbox("3D plot", key="plot_show_3d")
    show_bonds = pcol2.checkbox("Bonds", value=True)
    Nrepx = pcol3.number_input(
        "Repeat x", min_value=1, max_value=None, value=1)
    Nrepy = pcol4.number_input(
        "Repeat y", min_value=1, max_value=None, value=1)
    if show_3d:
        size = 100
        pcol5.write("")
    else:
        size = pcol5.slider(
            "Point size", min_value=0, max_value=100, value=100, step=1, key="size"
        )

height3d = 500
plot_struct = display_struct.copy()
plot_struct.set_constraint()

if show_3d:
    struct3d = plot_struct.repeat((Nrepx, Nrepy, 1))
    struct3d.wrap(eps=1e-9)

    max_3d_atoms_soft = 12000
    max_3d_atoms_hard = 60000
    show_bonds_3d = bool(show_bonds)
    natoms_3d = len(struct3d)

    if natoms_3d > max_3d_atoms_soft:
        if natoms_3d > max_3d_atoms_hard:
            st.warning(
                f"Very large 3D model ({natoms_3d} atoms): applying aggressive sampling for responsiveness."
            )
        # Keep all metal atoms and subsample boron atoms to keep the model interactive.
        zvals_all = np.array(struct3d.get_atomic_numbers())
        metal_idx = np.where(zvals_all != 5)[0]
        boron_idx = np.where(zvals_all == 5)[0]
        keep_budget = max_3d_atoms_soft - len(metal_idx)
        if keep_budget <= 0:
            keep_step = int(np.ceil(len(metal_idx) / max_3d_atoms_soft))
            keep_idx = metal_idx[:: max(1, keep_step)]
        elif len(boron_idx) > keep_budget:
            keep_step = int(np.ceil(len(boron_idx) / keep_budget))
            keep_boron = boron_idx[:: max(1, keep_step)][:keep_budget]
            keep_idx = np.sort(np.concatenate([metal_idx, keep_boron]))
        else:
            keep_idx = np.arange(len(struct3d), dtype=int)

        if len(keep_idx) < len(struct3d):
            struct3d = struct3d[keep_idx]
            show_bonds_3d = False
            st.info(
                f"Large 3D structure: showing {len(struct3d)} sampled atoms out of {natoms_3d}. "
                "Bonds are disabled in this fast preview."
            )

    c_height = float(struct3d.cell[2, 2])
    # Hide periodic metal duplicates that appear exactly on the top boundary (z=c).
    top_tol = 0.15
    metal_top_mask = (np.array(struct3d.get_atomic_numbers()) != 5) & (
        struct3d.positions[:, 2] > (c_height - top_tol)
    )
    if np.any(metal_top_mask):
        struct3d = struct3d[~metal_top_mask]
    try:
        system = write_pdb(
            struct3d, include_pbc_bonds=False, include_bonds=show_bonds_3d
        )
    except TypeError:
        # Backward-compatibility: older write_pdb signatures don't accept include_bonds.
        system = write_pdb(struct3d, include_pbc_bonds=False)
    atB = {"atom": "B"}
    xyzview = py3Dmol.view(width=None, height=height3d)
    xyzview.addModelsAsFrames(str(system))
    if metalchoice != "":
        xyzview.setStyle({"sphere": {"color": collist[metalchoice]}})
    boron_style = {
        "sphere": {"color": "red", "radius": 1, "opacity": 0.9, "scale": 0.5}
    }
    if show_bonds_3d:
        boron_style["stick"] = {"color": "red",
                                "radius": 0.2, "opacity": 0.9}
    xyzview.setStyle(atB, boron_style)
    xyzview.setBackgroundColor("white")
    xyzview.addUnitCell({"box": {"color": "purple"}})
    xyzview.spin(False)
    # Fit all atoms, then zoom out slightly so the full repeated structure is visible.
    xyzview.zoomTo()
    xyzview.zoom(1.4)
    with view_col:
        showmol_iframe(xyzview, height=height3d)
else:
    if go is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        a = plot_struct.cell[0, 0]
        b = plot_struct.cell[1, 1]
        structrep = plot_struct.repeat((Nrepx, Nrepy, 1))
        structrep.set_cell(plot_struct.cell)
        boron = structrep[structrep.get_atomic_numbers() == 5]
        metal = structrep[structrep.get_atomic_numbers() != 5]
        full_boron_ref = structfull if not use_imported_structure else None
        vacancy_positions, _ = vacancy_positions_from_full_reference(
            structrep,
            full_boron_ref,
            repx=Nrepx,
            repy=Nrepy,
        )
        if metalchoice != "":
            ax.scatter(
                metal.positions[:, 0],
                metal.positions[:, 1],
                c=collist[metalchoice],
                zorder=1,
                s=200 * size / 100,
                edgecolor="black",
                linewidth=0.5,
            )
        if show_bonds:
            bond_pairs = boron_bond_pairs(
                boron, cutoff=2.2, include_pbc=False)
            if len(bond_pairs) > 0:
                segments = np.stack(
                    [
                        boron.positions[bond_pairs[:, 0], :2],
                        boron.positions[bond_pairs[:, 1], :2],
                    ],
                    axis=1,
                )
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors="pink",
                        linewidths=1.5,
                        zorder=2,
                    )
                )
        if len(vacancy_positions) > 0:
            ax.scatter(
                vacancy_positions[:, 0],
                vacancy_positions[:, 1],
                c="pink",
                alpha=0.2,
                zorder=3,
                s=120 * size / 100,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.scatter(
            boron.positions[:, 0],
            boron.positions[:, 1],
            c="pink",
            zorder=4,
            s=120 * size / 100,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.plot(
            [0, a, a, 0, 0],
            [0, 0, b, b, 0],
            c="#cfcfcf",
            linewidth=1.2,
            zorder=0,
            linestyle="--",
        )
        ax.axis("off")
        ax.set_aspect("equal", "datalim")
        with view_col:
            st.pyplot(fig)
    else:
        structrep = plot_struct.repeat((Nrepx, Nrepy, 1))
        structrep.set_cell(plot_struct.cell)
        full_boron_ref = structfull if not use_imported_structure else None
        vacancy_positions, vacancy_indices = vacancy_positions_from_full_reference(
            structrep,
            full_boron_ref,
            repx=Nrepx,
            repy=Nrepy,
        )
        boron_tooltip_indices = boron_indices_from_full_reference(
            structrep,
            full_boron_ref,
            repx=Nrepx,
            repy=Nrepy,
        )
        pfig = plotly_structure_2d_figure(
            structrep,
            metalchoice,
            show_bonds=show_bonds,
            point_scale=max(0.2, size / 100),
            metal_label=metal_display_name(metalchoice),
            boron_indices=boron_tooltip_indices,
            vacancy_positions=vacancy_positions,
            vacancy_indices=vacancy_indices,
        )
        with view_col:
            st.plotly_chart(pfig, width="stretch", key="main_plotly_2d")

with view_col:
    with st.expander(
        ":material/table_view: Atomic positions (editable)", expanded=False
    ):
        positions = display_struct.positions
        atomic_table = pd.DataFrame(
            {
                "Atom ID": np.arange(len(display_struct), dtype=int),
                "Element": display_struct.get_chemical_symbols(),
                "x [A]": positions[:, 0],
                "y [A]": positions[:, 1],
                "z [A]": positions[:, 2],
            }
        )
        edited_table = st.data_editor(
            atomic_table,
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            disabled=["Atom ID", "Element"],
            key=f"atomic_positions_editor_{active_structure_id}",
            column_config={
                "x [A]": st.column_config.NumberColumn(format="%.6f"),
                "y [A]": st.column_config.NumberColumn(format="%.6f"),
                "z [A]": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        valid_rows = edited_table.copy()
        valid_rows["Atom ID"] = pd.to_numeric(
            valid_rows["Atom ID"], errors="coerce"
        )
        valid_rows = valid_rows.dropna(
            subset=["Atom ID", "x [A]", "y [A]", "z [A]"]
        )
        if len(valid_rows) == 0:
            st.warning(
                "At least one atom must remain in the active structure.")
        else:
            edited_indices = valid_rows["Atom ID"].astype(int).to_numpy()
            edited_positions = valid_rows[["x [A]", "y [A]", "z [A]"]].to_numpy(
                dtype=float
            )

            valid_indices = (
                len(np.unique(edited_indices)) == len(edited_indices)
                and np.all(edited_indices >= 0)
                and np.all(edited_indices < len(display_struct))
            )
            if not valid_indices:
                st.warning(
                    "Invalid or duplicated Atom ID detected; please keep unique existing rows only."
                )
            else:
                same_indices = np.array_equal(
                    edited_indices, np.arange(
                        len(display_struct), dtype=int)
                )
                same_positions = (
                    edited_positions.shape == positions.shape
                    and np.allclose(
                        edited_positions,
                        positions,
                        atol=1e-9,
                        rtol=0.0,
                    )
                )
                if not (same_indices and same_positions):
                    st.session_state.stm_active_structure_position_overrides[
                        active_structure_id
                    ] = {
                        "indices": edited_indices.tolist(),
                        "positions": edited_positions.tolist(),
                    }
                    st.rerun()

with controls_col.expander(":material/trending_down: MD Quench", expanded=False):
    with st.container(height=450, border=False):
        captionplace = st.empty()
        st.caption(deepmd_runtime_summary())

        left_relax, right_relax = st.columns((1, 1))
        t_start = left_relax.number_input(
            "Starting temperature [K]",
            min_value=0.0,
            value=100.0,
            step=5.0,
            key="stm_t_start",
        )
        t_end = right_relax.number_input(
            "Ending temperature [K]",
            min_value=0.0,
            value=5.0,
            step=1.0,
            key="stm_t_end",
        )
        nsteps = st.number_input(
            "Number of MD steps",
            min_value=1,
            max_value=100000,
            value=1000,
            step=1000,
            key="stm_nsteps",
        )
        captionplace.caption(
            f"""{nsteps}-step NVT quench (1 fs) from {t_start} K to {t_end} K.

It is recommended to quench or minimize large structures using your own HPC facilities for better performance. The quench implemented here is intended for quick tests on small structures, and may not be efficient for large systems or long runs."""
        )
        current_sig = source_sig
        if st.session_state.get("stm_relaxed_signature") != current_sig:
            st.session_state.stm_relaxed_signature = current_sig
            st.session_state.stm_relaxed_atoms = None
            st.session_state.pop("stm_relax_job", None)
            st.session_state.pop("stm_active_frame_index", None)
            st.session_state.pop("stm_thermo_history", None)

        if "stm_relax_job" not in st.session_state:
            st.session_state.stm_relax_job = None

        @st.fragment(run_every=1)
        def render_quench_panel():
            btn_col1, btn_col2 = st.columns((1, 1))
            run_quench = btn_col1.button(
                "Run MD Quench",
                key="run_stm_quench",
                width="stretch",
                icon=":material/play_arrow:",
            )
            stop_quench = btn_col2.button(
                "Stop MD Quench",
                key="stop_stm_quench",
                width="stretch",
                icon=":material/stop:",
            )
            progress_slot = st.empty()
            status = st.empty()

            job = st.session_state.get("stm_relax_job")
            if stop_quench and job is not None and job.running:
                job.stop_requested = True

            if run_quench:
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "potential",
                    "graph.pb",
                )
                if not os.path.isfile(model_path):
                    st.error(
                        f"Could not find DeePMD model at: {model_path}")
                else:
                    job = RelaxationJob(
                        display_struct, model_path, t_start, t_end, nsteps
                    )
                    job.running = True
                    job.status = "Starting quench..."
                    st.session_state.stm_relax_job = job
                    worker = threading.Thread(
                        target=run_quench_job, args=(job,), daemon=True
                    )
                    worker.start()

            job = st.session_state.get("stm_relax_job")
            if job is None:
                return

            progress_slot.progress(job.progress)
            status.write(job.status)

            if job.done and job.relaxed_atoms is not None:
                st.session_state.stm_relaxed_atoms = job.relaxed_atoms
                st.session_state.stm_relaxed_signature = current_sig
                st.session_state.stm_thermo_history = {
                    "steps": list(job.step_history),
                    "temperature": list(job.temperature_history),
                    "energy": list(job.energy_history),
                }
                if job.error is None and not job.stop_requested:
                    st.success("Relaxation complete.")
                elif job.stop_requested:
                    st.warning("Relaxation was stopped by user.")

            if os.path.isfile(job.traj_path):
                st.write("**Download trajectory:**")
                download_col1, download_col2 = st.columns((1, 1))
                with open(job.traj_path, "rb") as f:
                    download_col1.download_button(
                        label="ASE (.traj)",
                        data=f.read(),
                        file_name="quench.traj",
                        mime="application/octet-stream",
                        key="download_quench_traj",
                        width="stretch",
                        icon=":material/download:",
                    )

                if job.done or job.stop_requested:
                    try:
                        cache = build_lammpstrj_cache(job.traj_path)

                        download_col2.download_button(
                            label="LAMMPS (.lammpstrj)",
                            data=cache["data"],
                            file_name="quench.lammpstrj",
                            mime="text/plain",
                            key="download_quench_lammpstrj",
                            on_click=refresh_lammpstrj_cache,
                            args=(job.traj_path,),
                            width="stretch",
                            icon=":material/download:",
                        )
                        cached_error = st.session_state.get(
                            "stm_relax_lammpstrj_error"
                        )
                        if cached_error:
                            download_col2.caption(
                                f"Last refresh failed: {cached_error}"
                            )
                    except Exception as exc:
                        download_col2.caption(
                            f"LAMMPS export unavailable: {exc}")
                else:
                    download_col2.caption(
                        "LAMMPS export will be available when quench finishes or is stopped."
                    )

        render_quench_panel()
        relax_job_for_slider = st.session_state.get("stm_relax_job")
        if (
            relax_job_for_slider is not None
            and relax_job_for_slider.done
            and os.path.isfile(relax_job_for_slider.traj_path)
        ):
            try:
                traj_frames = ase_read(
                    relax_job_for_slider.traj_path, index=":")
                if not isinstance(traj_frames, list):
                    traj_frames = [traj_frames]
                if len(traj_frames) > 1:
                    if "stm_active_frame_index" not in st.session_state:
                        st.session_state.stm_active_frame_index = (
                            len(traj_frames) - 1
                        )
                    if (
                        st.session_state.stm_active_frame_index
                        > len(traj_frames) - 1
                    ):
                        st.session_state.stm_active_frame_index = (
                            len(traj_frames) - 1
                        )
                    st.slider(
                        "Active frame",
                        min_value=0,
                        max_value=len(traj_frames) - 1,
                        key="stm_active_frame_index",
                        help="Choose which saved trajectory frame is used in 2D/3D views.",
                    )
            except Exception:
                pass

with controls_col.expander(
    ":material/keyboard_double_arrow_down: Minimization", expanded=False
):
    st.caption(
        "Run an ASE SciPy conjugate-gradient minimization with the DeePMD calculator. The lowest metal layer is fixed automatically."
    )
    st.caption(deepmd_runtime_summary())

    left_min, right_min = st.columns((1, 1))
    min_fmax = left_min.number_input(
        "Force convergence [eV/Å]",
        min_value=0.0,
        value=0.05,
        step=0.01,
        key="stm_min_fmax",
    )
    min_steps = right_min.number_input(
        "Maximum CG steps",
        min_value=1,
        max_value=100000,
        value=300,
        step=50,
        key="stm_min_steps",
    )

    relax_job = st.session_state.get("stm_relax_job")
    quench_running = relax_job is not None and not relax_job.done
    run_minimization = st.button(
        "Run minimization",
        key="run_stm_minimization",
        width="stretch",
        disabled=quench_running,
        icon=":material/play_arrow:"
    )
    if quench_running:
        st.info(
            "Wait for the MD quench to finish, or stop it, before running minimization."
        )

    if run_minimization:
        model_path = get_deepmd_model_path()
        if not os.path.isfile(model_path):
            st.error(f"Could not find DeePMD model at: {model_path}")
        else:
            try:
                with st.spinner("Running CG minimization..."):
                    minimized = prepare_deepmd_atoms(
                        display_struct, model_path)
                    optimizer = SciPyFminBFGS(minimized, logfile=None)
                    converged = optimizer.run(
                        fmax=float(min_fmax), steps=int(min_steps)
                    )
                    minimized.wrap()
                    max_force = 0.0
                    if len(minimized) > 0:
                        max_force = float(
                            np.linalg.norm(
                                minimized.get_forces(), axis=1).max()
                        )
                    st.session_state.stm_minimized_atoms = minimized.copy()
                    st.session_state.stm_minimized_source_signature = source_sig
                    st.session_state.stm_minimization_summary = {
                        "converged": bool(converged),
                        "steps": int(optimizer.nsteps),
                        "energy": float(minimized.get_potential_energy()),
                        "max_force": max_force,
                    }
                    st.session_state.pop("stm_relax_job", None)
                    st.session_state.pop("stm_active_frame_index", None)
                    st.session_state.pop("stm_thermo_history", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Minimization failed: {exc}")

    minimization_summary = st.session_state.get("stm_minimization_summary")
    if minimization_summary is not None:
        if minimization_summary["converged"]:
            st.success(
                f"Minimization converged in {minimization_summary['steps']} BFGS steps."
            )
        else:
            st.warning(
                f"Minimization stopped after {minimization_summary['steps']} BFGS steps without reaching the target force threshold."
            )
        st.caption(
            f"Final potential energy: {minimization_summary['energy']:.6f} eV | Max force: {minimization_summary['max_force']:.6f} eV/Å"
        )

with controls_col.expander(":material/archive: Output", expanded=False):
    extension_out = st.selectbox(
        "File format", ("VASP", "xyz", "LAMMPS", "PDB"))

    export_struct = struct.copy()
    if extension_out == "VASP":
        st.write("##### For VASP output")
        fixed = st.number_input(
            "Fixed number of layers",
            min_value=0,
            max_value=NZ + 1,
            value=NZ - 1,
            step=1,
        )
        if fixed > 0:
            c = FixAtoms(
                mask=export_struct.positions[:,
                                             2] <= 2.35 * (fixed - 1) + 1.2
            )
            export_struct.set_constraint(c)

    st.markdown("<br>", unsafe_allow_html=True)

    outfile, name = writeout(
        export_struct, extension_out, source_name=imported_name
    )

    st.download_button(
        label="Download Output file",
        data=outfile,
        file_name=name,
        mime="text/csv",
        width="stretch",
        icon=":material/download:",
    )


# Show table with info
a = str(np.round(struct.cell[0, 0], 4))
b = str(np.round(struct.cell[1, 1], 4))
c = str(np.round(struct.cell[2, 2], 4))
mx = (
    f"{(struct.get_cell()[0][0]-base.get_cell()[0][0])/base.get_cell()[0][0]*100:.2f} %"
)
my = (
    f"{(struct.get_cell()[1][1]-base.get_cell()[1][1])/base.get_cell()[1][1]*100:.2f} %"
)
hd = Fraction(len(listholes), (2 * nx * ny))
with outcol.container(border=True):
    st.write("##### Cell Info")
    if imported_name is not None:
        st.caption(f"Source: {imported_name}")
    if use_imported_structure and metalchoice != "":
        metal_label = "metal" if metalchoice == "Metal" else metalchoice
        df = pd.DataFrame(
            {
                "": [
                    "a [Å]",
                    "b [Å]",
                    "c [Å]",
                    "N<sub>total</sub>",
                    "N<sub>B</sub>",
                    f"N<sub>{metal_label}</sub>",
                ],
                "b": [a, b, c, len(struct), len(info_boron), len(info_metal)],
            }
        )
    elif use_imported_structure:
        df = pd.DataFrame(
            {
                "": ["a [Å]", "b [Å]", "c [Å]", "N<sub>total</sub>", "N<sub>B</sub>"],
                "b": [a, b, c, len(struct), len(info_boron)],
            }
        )
    elif metalchoice != "":
        df = pd.DataFrame(
            {
                "": [
                    "a [Å]",
                    "b [Å]",
                    "c [Å]",
                    "N<sub>total</sub>",
                    "N<sub>B</sub>",
                    f"N<sub>{metalchoice}</sub>",
                    "Hole density",
                ],
                "b": [
                    a,
                    b,
                    c,
                    len(struct),
                    len(info_boron),
                    len(info_metal),
                    f"{hd.numerator}/{hd.denominator}",
                ],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "": [
                    "a [Å]",
                    "b [Å]",
                    "c [Å]",
                    "N<sub>total</sub>",
                    "N<sub>B</sub>",
                    "Hole density",
                ],
                "b": [
                    a,
                    b,
                    c,
                    len(struct),
                    len(info_boron),
                    f"{hd.numerator}/{hd.denominator}",
                ],
            }
        )
    st.write(df.style.hide(axis=0).hide(
        axis=1).to_html(), unsafe_allow_html=True)

    st.write("##### System's energy")
    model_path = get_deepmd_model_path()
    struct.calc = DP(model=model_path)
    Etot = struct.get_potential_energy()
    Batoms = struct[struct.symbols.search("B")]
    Batoms.calc = DP(model=model_path)
    E_boron = Batoms.get_potential_energy()/len(Batoms)
    Matoms = struct[struct.symbols.search(metalchoice)]
    Matoms.calc = DP(model=model_path)
    E_metal = Matoms.get_potential_energy()/len(Matoms)
    df = pd.DataFrame(
        {
            "": ["Total energy [eV]",
                 "Boron energy [eV/atom]",
                 f"{metalchoice} energy [eV/atom]",
                 ],
            "b": [Etot, E_boron, E_metal],
        }
    )
    st.write(df.style.hide(axis=0).hide(
        axis=1).to_html(), unsafe_allow_html=True)

    if not use_imported_structure:
        st.write("")
        st.write("##### Borophene deformation")
        df = pd.DataFrame({"": ["x", "y"], "Borophene deformation": [mx, my]})
        st.write(df.style.hide(axis=0).hide(
            axis=1).to_html(), unsafe_allow_html=True)
