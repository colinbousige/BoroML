#!/usr/bin/env python

###############################################################################
# File converter from VASP OUTCAR to input.data format.
# Works also if OUTCAR contains trajectories.
# Tested with VASP 5.2.12
###############################################################################

import numpy as np
import sys

def print_usage():
    sys.stderr.write("USAGE: {0:s} <in_file> <<out_file>>\n".format(sys.argv[0]))
    sys.stderr.write("       <in_file> .... OUTCAR file name.\n")
    sys.stderr.write("       <out_file> ... Output file name (optional).\n")
    return

if len(sys.argv) < 2 or sys.argv[1] in ["-?", "-h", "--help"]:
    print_usage()
    sys.exit(1)

file_name = sys.argv[1]
if len(sys.argv) > 2:
    outfile_name = sys.argv[2]
else:
    outfile_name = None

# Read in the whole file first.
f = open(file_name, "r")
lines = [line for line in f]
f.close()

# Check that
# "free energy    TOTEN"  appears less than 60 times
# "free  energy   TOTEN" exists
counter = 0
goodend = 0
for line in lines:
    if "free energy    TOTEN" in line:
        counter += 1
    if counter > 50:
        sys.stderr.write(f"File {file_name} was rejected: counter={counter}.\n")
        sys.exit(1)
    if "free  energy   TOTEN" in line:
        goodend = 1
        counter = 0


if goodend == 0:
    sys.stderr.write(f"File {file_name} was rejected: goodend={goodend}.\n")
    sys.exit(1)

# If OUTCAR contains ionic movement run (e.g. from an MD simulation) multiple
# configurations may be present. Thus, need to prepare empty lists.
lattices   = []
energies   = []
atom_lists = []

# Loop over all lines.
elements = []
for i in range(len(lines)):
    line = lines[i]
    # Collect element type information, expecting VRHFIN lines like this:
    #
    # VRHFIN =Cu: d10 p1
    #
    if "VRHFIN" in line:
        elements.append(line.split()[1].replace("=", "").replace(":", ""))
    # VASP specifies how many atoms of each element are present, e.g.
    #
    # ions per type =              48  96
    #
    if "ions per type" in line:
        atoms_per_element = [int(it) for it in line.split()[4:]]
    # Simulation box may be specified multiple times, I guess this line
    # introduces the final lattice vectors.
    if "VOLUME and BASIS-vectors are now" in line:
        lattices.append([lines[i+j].split()[0:3] for j in range(5, 8)])
    # Total energy is found in the line with "energy  without" (2 spaces) in
    # the column with sigma->0:
    #
    # energy  without entropy=     -526.738461  energy(sigma->0) =     -526.738365
    #
    if "energy  without entropy" in line:
        energies.append(line.split()[6])
    # Atomic coordinates and forces are found in the lines following
    # "POSITION" and "TOTAL-FORCE".
    if "POSITION" in line and "TOTAL-FORCE" in line:
        atom_lists.append([])
        count = 0
        for ei in range(len(atoms_per_element)):
            for j in range(atoms_per_element[ei]):
                atom_line = lines[i+2+count]
                atom_lists[-1].append(atom_line.split()[0:6])
                atom_lists[-1][-1].extend([elements[ei]])
                count += 1

# Sanity check: do all lists have the same length. 
if not (len(lattices) == len(energies) and len(energies) == len(atom_lists)):
    raise RuntimeError("ERROR: Inconsistent OUTCAR file.")

# Open output file or write to stdout.
if outfile_name is not None:
    f = open(outfile_name, "w")
else:
    f = sys.stdout

cdist=1.0/0.52917721
cener=1.0/27.21138469
cforce=cener/cdist
# Write configurations in "input.data" format.
for i, (lattice, energy, atoms) in enumerate(zip(lattices, energies, atom_lists)):
    fmax = -10
    zeroforce = 0
    for a in atoms:
        fmax = np.max([fmax, 
                      np.sqrt( float(a[3])**2 + float(a[4])**2 + float(a[5])**2 ) ])
        if float(a[3])==0 or float(a[4])==0 or float(a[5])==0:
            zeroforce += 1
    if fmax <= 25 and float(energy) < 0 and zeroforce == 0:
        f.write("begin\n")
        f.write("comment source_file_name={0:s} structure_number={1:d} fmax={2:f}\n".format(file_name, i + 1, fmax))
        f.write("lattice {0:22.14e} {1:22.14e} {2:22.14e}\n".format(float(lattice[0][0])*cdist, float(lattice[0][1])*cdist, float(lattice[0][2])*cdist))
        f.write("lattice {0:22.14e} {1:22.14e} {2:22.14e}\n".format(float(lattice[1][0])*cdist, float(lattice[1][1])*cdist, float(lattice[1][2])*cdist))
        f.write("lattice {0:22.14e} {1:22.14e} {2:22.14e}\n".format(float(lattice[2][0])*cdist, float(lattice[2][1])*cdist, float(lattice[2][2])*cdist))
        for a in atoms:
            f.write("atom {0:22.14e} {1:22.14e} {2:22.14e} {3:s} {4:s} {5:s} {6:22.14e} {7:22.14e} {8:22.14e}\n".format(
            float(a[0])*cdist, float(a[1])*cdist, float(a[2])*cdist, a[6], "0.0", "0.0", float(a[3])*cforce, float(a[4])*cforce, float(a[5])*cforce)
            )
        f.write("energy {0:22.14e}\n".format(float(energy)*cener))
        f.write("charge {0:s}\n".format("0.0"))
        f.write("end\n")
    else:
        sys.stderr.write(f"File {file_name} was rejected: fmax={fmax} eV/A\n")
