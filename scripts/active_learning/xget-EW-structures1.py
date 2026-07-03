import argparse
from ase.io.lammpsrun import read_lammps_dump
from ase.io import write
from ase import Atoms
from ase.build.tools import sort
import os
import numpy as np
import subprocess
from checkdist import check_distances_verb

# ------------ DEFINE ARGUMENTS 
def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--dumpfile", type=str, default="dump.lammpstrj", help="LAMMPS dump file : --dumpfile=dump.lammpstrj")
    parser.add_argument("--outlammpsfile", type=str, default="input_lammps.out", help="LAMMPS out files : --outlammpsfile=input_lammps.out")
    parser.add_argument("--atomsNNP", type=str, default="HOAlSi", help="Atoms ordering for the NNP in LAMMPS: --atomsNNP=HOAlSi")
    parser.add_argument("--prefix", type=str, default="0", help="prefix to be added to the POSCAR file")
    parser.add_argument("--nbstruct", type=str, default="4", help="Nb of EW structures taken at the end of dump file,default = 4")
    parser.add_argument("--step", type=str, default="1", help="Nb of step between each EW structure taken at the end of dump file,default = 1")
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
    for index,AtNum in enumerate(struct.get_atomic_numbers()):
        atnb = np.append(atnb,at_nbs[AtNum])
        atsymb.append(at_sym[AtNum])
    struct.set_atomic_numbers(atnb)
    struct.set_chemical_symbols(atsymb)

# ------------ 
args = getArguments()
dumpfile = args.dumpfile
outlammpsfile = args.outlammpsfile
os.makedirs("EW_poscar", exist_ok=True)
atoms_NNP = args.atomsNNP
prefix = args.prefix
nbstruct = int(args.nbstruct)
step = int(args.step)

# ------------ GET EW STRUCTURES INDICES 
cmd = f"grep -A1 \"NNP EXTRAPOLATION WARNING\" {outlammpsfile}|"
cmd += "grep \"NNP EW SUMMARY\"| awk \'{print $7}\'"
#cmd += f"|tail -{nbstruct}"
EW_structures = subprocess.getoutput(cmd)
EW_structures = EW_structures.split('\n')
EW_structures = [x for x in EW_structures if x.strip()]  # Delete empty elements
if not EW_structures:
    print(f"No EW structures in {outlammpsfile}")
    exit(1)  # Quit if no EW structures
EW_structures = list(map(int,EW_structures))
EW_structures = [ a-1 for a in EW_structures ]
firstel = EW_structures[0]
EW_structures = list(reversed(EW_structures))
EW_structures = EW_structures[::step]
EW_structures.append(firstel)
#EW_structures = list(reversed(EW_structures))
#EW_structures = EW_structures[-nbstruct:]
print(f"=====> List of EW Structures potentially converted to poscar from ===>>>   {dumpfile} ")
print(EW_structures)

# ------------ WRITE POSCARs
file = open(outlammpsfile, "r")
nbcnvrtd = 0
maxN = min(len(EW_structures),nbstruct)
for i in EW_structures[:maxN]:
    print(f"     ===> Busy with EW Structure {i}")
    struct = read_lammps_dump(dumpfile, index = i)
    reset_Atoms(struct, atoms_NNP)
    # sort atoms by increasing mass
    struct=sort(struct, tags = struct.get_masses())
    print("           ==> Check Distances")
    if check_distances_verb(struct):
        write(f"../../EW_poscar/POSCAR_{prefix}_{i}", struct, format = "vasp", vasp5=True)
        print(f"               => OK ! converted to POSCAR")
        nbcnvrtd += 1
    else:
        print(f"               => NOT CONVERTED to POSCAR cause too small distance(s) observed")
    if nbcnvrtd == nbstruct:    break
file.close()
