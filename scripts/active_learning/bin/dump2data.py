import argparse
from ase.io.lammpsrun import read_lammps_dump
from ase.io.lammpsdata import read_lammps_data
from ase import Atoms
import numpy as np
from checkdist import check_distances_verb
from read_write import write_inpdat
from joblib import Parallel, delayed
from joblib import cpu_count
import subprocess

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                               #
#       Selects the last NGetstruct structures taken every freq structures      #
#                                from file dumpfile                             #
#     and writes these structures to Dump_{prefix}.data file in n2p2 format     #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# ------------ DEFINE ARGUMENTS 
def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--dumpfile", type=str, default="dump.lammpstrj", help="LAMMPS dump file : --dumpfile=dump.lammpstrj")
    parser.add_argument("--atomsNNP", type=str, default="HOAlSi", help="Atoms ordering for the NNP in LAMMPS: --atomsNNP=HOAlSi")
    parser.add_argument("--prefix", type=str, default="0", help="prefix to be added to the POSCAR file")
    parser.add_argument("--freq", type=str, default="1", help="Nb of step between each EW structure taken at the end of dump file,default = 1")
    parser.add_argument("--NGetstruct", type=int, default=10, help="--NGetstruct=10, Nb of structures from wich random structures are made, default = 10")
    parser.add_argument("--chkfile", type=str, default="none", help="LAMMPS dat file : --chkfile=structure.dat")

    args = parser.parse_args()
    return args

# ------------  SET ATOMS SYMBOLS AND AT NB
def reset_Atoms(struct, atoms_NNP):
    at_nbs,at_sym = [0],[0]
    for j in Atoms(atoms_NNP):
        at_nbs.append(j.number)
        at_sym.append(j.symbol)
    atnb = np.empty([0],dtype=int)
    atsymb = []
    for AtNum in struct.get_atomic_numbers():
        atnb = np.append(atnb,at_nbs[AtNum])
        atsymb.append(at_sym[AtNum])
    struct.set_atomic_numbers(atnb)
    struct.set_chemical_symbols(atsymb)

# ------------------------- check if struct cell differs more than threshold as compared to lammps dat file  ----------------------------------
def check_cell(struct, dat_cell):
    struct_cell = struct.get_cell()
    a_s = np.linalg.norm(struct_cell[0])
    b_s = np.linalg.norm(struct_cell[1])
    c_s = np.linalg.norm(struct_cell[2])
    a_d = np.linalg.norm(dat_cell[0])
    b_d = np.linalg.norm(dat_cell[1])
    c_d = np.linalg.norm(dat_cell[2])
    deva = abs(a_s - a_d) / a_d
    devb = abs(b_s - b_d) / b_d
    devc = abs(c_s - c_d) / c_d
    return deva <= 0.15 and devb <= 0.15 and devc <= 0.15

#--------------------------------    write dump.data if distances are checked ok   --------------
def writeifchecked(traji, dumpdat, prefix, dat_cell):
    structs = []
    comments = []
    for struct in traji:
        if check_distances_verb(struct) and check_cell(struct, dat_cell):
            structs.append(struct)
            comments.append(f"from dump file {prefix}")
    write_inpdat(structs, dumpdat, comments)        

#--------------------------------    IN ORDER TO PARALLELIZE    -----------------------------------
def getNBNE(njobs, trajlen):
    print(f"Number of structures : {trajlen}   /   Number of procs : {njobs}")
    m = trajlen//njobs
    M =[m]*njobs
    for i in range(trajlen%njobs):
        M[i] += 1
    print("Parallell : Structs per Jobs = ",M)
    NB =[0]*njobs
    NE =[0]*njobs
    NE[0] = M[0]
    for i in range(1,njobs):
        NB[i] = NE[i-1]
        NE[i] = NB[i]+M[i]
    return NB, NE

# ------------ 
args = getArguments()
dumpfile = args.dumpfile
atoms_NNP = args.atomsNNP
prefix = args.prefix
freq = int(args.freq)
NGetstruct = args.NGetstruct
chkfile = args.chkfile

# ------------ get cell parameters from LAMMPS data file
dat = read_lammps_data(chkfile, style="molecular")
dat_cell = dat.get_cell()


# ------------ READ DUMP FILE
# -- check number of structures
N_STRUCT = subprocess.run(f"grep -c TIMESTEP {dumpfile}", 
                           shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
N_STRUCT = int(N_STRUCT)
f = N_STRUCT // NGetstruct
# -- read structures and adapt the freq variable to nb of structures
if f < 1:
    traj = read_lammps_dump(dumpfile,index=slice(0,-1))
else:
    freq = min(f,freq)
    traj = read_lammps_dump(dumpfile,index=slice(0,-1,freq))
# -- put structures in memory after checking geometries 
selected = 0
New_traj = []
for struct in reversed(traj):
    reset_Atoms(struct, atoms_NNP)
    if check_distances_verb(struct):
        New_traj.append(struct)
        selected += 1
        if selected == NGetstruct:
            break
traj = New_traj


# ------------ check number of procs
Ncpu_count = int(cpu_count() / 4)
njobs = min(Ncpu_count, len(traj))
print(f"!!!   Run on {njobs} procs   !!!")

# ------------ write data files after checking distances and cell parameters
if njobs == 1:
    writeifchecked(traj, "dumpdat_1.data", prefix, dat_cell)
else:
    NB, NE = getNBNE(njobs, len(traj))
    print(NB, NE)
    Parallel(n_jobs=njobs, verbose=20)(
        delayed(writeifchecked)(
            traj[NB[i]:NE[i]],
            f"dumpdat_{prefix}_{i}.data",
            prefix,
            dat_cell
            ) for i in range(njobs)
    )

    subprocess.run(f"cat dumpdat_{prefix}_*.data >> Dump_{prefix}.data", shell=True)
    subprocess.run(f"rm -f dumpdat_*.data", shell=True)
