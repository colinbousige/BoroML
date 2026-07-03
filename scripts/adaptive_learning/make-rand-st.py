from ase.geometry import wrap_positions
from ase.io.lammpsrun import read_lammps_dump
from ase.io.lammpsdata import read_lammps_data
from read_write import write_inpdat, read_inputdata
from ase import Atoms
import argparse
import numpy as np
from checkdist import check_distances, check_distances_verb
from joblib import Parallel, delayed
from joblib import cpu_count
import subprocess

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                               #
#       Selects the last NGetstruct structures taken every freq structures      #
#                           from file dumpfile or datafile                      #
#      creates NRandstruct random structures for each of theses structures      #
#         and writes all these structures to outfile file in n2p2 format        #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# ------------------------- DEFINE ARGUMENTS   ----------------------------------------------------
def getArguments():
    #define command line arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--dumpfile", type=str, default="none", help="LAMMPS dump file : --dumpfile=dump.lammpstrj ; if not specified = none")
    parser.add_argument("--datafile", type=str, default="none", help="n2p2 data file : --datafile=file.data ; if not specified = none")
    parser.add_argument("--NGetstruct", type=int, default=10,
                    help="--NGetstruct=10, Nb of structures from wich random structures are made, default = 10")
    parser.add_argument("--freq", type=int, default=-1, 
                        help="geometries from which the random is made, --freq=-1 => last, n => frequency(every n geoms), all")
    parser.add_argument("--maxNorm", type=float, default="0.1", 
                        help="--maxNorm=0.1 Angstrom max Norm displacements on each atom")
    parser.add_argument("--NRandstruct", type=int, default=10,
                        help="--NRandstruct=10, Nb of random structures created for each constain/maxNorm")
    parser.add_argument("--atomsNNP", type=str, default="HOAlSi", help="Atoms ordering for the NNP in LAMMPS: --atomsNNP=HOAlSi")
    parser.add_argument("--outfile", type=str, default='Rand_Structures.data', help='outfile : --outfile=Rand_Structures.data')
    parser.add_argument("--chkfile", type=str, default="none", help="LAMMPS dat file : --chkfile=structure.dat ; if not specified = none")
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

# ------------------------- CREATE RANDOM MOVES => NEW STRUCTURES  ----------------------------------
def shuffle(atoms, maxN):
    struct = atoms.copy()
    cell = struct.get_cell()
    for i in range(len(struct.positions)):
        x_norm = 2.0
        while x_norm > 1:  # This is to ensure uniform distribution in space
            xi = np.random.uniform(-1, 1, 3)
            x_norm = np.linalg.norm(xi)
        e_unitaire = xi/x_norm
        ran = np.random.uniform(0, 1) * maxN
        struct.positions[i] = atoms.positions[i] + ran * e_unitaire
    struct.positions = wrap_positions(struct.positions, cell, pbc=[1, 1, 1])
    return struct

#------------------    CREATE NEW RANDOM STRUCTURES   -------------------------
def create_struct(traji, commentsi, outfilei, maxNorm):
    randstructs = []
    for struct in traji:
        if check_distances_verb(struct):
            construct = struct.copy()
            conti = True
            count = 0
            while conti:
                count += 1
                newstruct = shuffle(construct, maxNorm)
    #            print("struct check dist : ", check_distances_verb(struct))
    #            print(outfilei, comment, "count:", count)
                if check_distances_verb(newstruct):
                    randstructs.append(newstruct)
                    conti = False
                if count == 20:
                    conti = False
                    print(f"!!!   WARNING : in {outfilei} {count} unsuccessful attempts to create a valid structure !!!")
    write_inpdat(randstructs, outfilei, commentsi)
#                print("!!!   Check the maxNorm value, it may be too large for the structure !!!")

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

#------------------------    DECLARE ARGUMENT VARIABLES, I/O FILES   ----------------------------------
args = getArguments()
dumpfile = args.dumpfile
datafile = args.datafile
NGetstruct = args.NGetstruct
NRandstruct = args.NRandstruct
freq = args.freq
outfile = args.outfile
atoms_NNP = args.atomsNNP
chkfile = args.chkfile

maxNorm = float(args.maxNorm)
print("Max Norm of atomic displacement : ", maxNorm, "Ang")

# ------------ get cell parameters from LAMMPS data file: usefull to check the geometries 
# ------------ cells should be ls than +/- 15% than the reference checkfile datafile cell
dat = read_lammps_data(chkfile, style="molecular")
dat_cell = dat.get_cell()

if dumpfile != "none":
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

if datafile != "none":
    # ------------ READ DUMP FILE
    # -- check number of structures
    N_STRUCT = subprocess.run(f"grep -c begin {datafile}", 
                            shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
    N_STRUCT = int(N_STRUCT)
    f = N_STRUCT // NGetstruct
    # -- read structures and adapt the freq variable to nb of structures
    if f < 1:
        traj = read_inputdata(datafile,index=slice(0,-1))
    else:
        freq = min(f,freq)
        traj = read_inputdata(datafile,index=slice(0,-1,freq))


# -- put structures in memory after checking geometries 
selected = 0
New_traj = []
for struct in reversed(traj):
    reset_Atoms(struct, atoms_NNP)
    if check_distances_verb(struct) and check_cell(struct, dat_cell):
        New_traj.append(struct)
        selected += 1
        if selected == NGetstruct:
            break

ToTtraj = []
ToTcomments = []
for struct, nb in zip(New_traj, range(len(New_traj))):
#    reset_Atoms(struct, atoms_NNP)
    for i in range(NRandstruct):
        ToTtraj.append(struct.copy())
        if dumpfile != "none":
            ffile = dumpfile
        if datafile != "none":
            ffile = datafile
        comment = f"Random structure {i+1} from geometry {nb+1} of {len(traj)} made from file {ffile}"
        ToTcomments.append(comment)

ffile = ffile[:-5]


print(f"For each Input Geometry {NRandstruct} created Random structures    => TOTAL = {len(ToTtraj)}")
#------------------------    CREATE RANDOM STRUCTURES   ----------------------------------

# Use the minimum of cpu_count() and number of structures to avoid empty jobs
Ncpu_count = int(cpu_count() / 4)
njobs = min(Ncpu_count, len(ToTtraj))
print(f"!!!   Run on {njobs} procs   !!!")

if njobs == 1:
    create_struct(ToTtraj, ToTcomments, outfile, maxNorm)
else:
    NB, NE = getNBNE(njobs, len(ToTtraj))
    print(NB, NE)
    Parallel(n_jobs=njobs, verbose=20)(
        delayed(create_struct)(
            ToTtraj[NB[i]:NE[i]],
            ToTcomments[NB[i]:NE[i]],
            f"{ffile}-outrand{i}.data",
            maxNorm
        ) for i in range(njobs)
    )
    subprocess.run(f"cat {ffile}-outrand*.data > {outfile}", shell=True)
    subprocess.run(f"rm -f {ffile}-outrand*.data", shell=True)
