"""
Set of functions to generate borophene structures on metal substrates and write LAMMPS input files for MD simulations.
"""

import os
import sys
import numpy as np
from ase import *
from ase.io.lammpsrun import read_lammps_dump
from ase.io import write
from ase.build.tools import sort
from ase.visualize import view
from ase.build import fcc111, fcc110, fcc100, fcc211, make_supercell, diamond100
from ase.io.lammpsdata import Prism, convert
from pathlib import Path
from scipy.spatial import distance
import itertools
import datetime

metals = {'Ag': 4.0853,
          'Au': 4.0782,
          'Al': 4.0495,
          'Cu': 3.6149,
          'Pt': 3.9242,
          'Ni': 3.5240,
          'Ir': 3.8390,
          'Si': 5.4309}
"""FCC unit cell lattice constants of metals in Angstroms"""

funcmap = {'fcc111': fcc111, 'fcc110': fcc110,
           'fcc100': fcc100, 'fcc211': fcc211,
           'diamond100': diamond100}
"""Mapping of metal-building functions"""

predef = {'alpha'  :(3,3,[0,10]), #alpha
          'alpha1' :(2,4,[0,10]), #alpha1
          'alpha2' :(16,16,[0,18,37,55,74,92,111,7,25,44,62,65,83,102,120,139,157,113,132,150,169,187,206,160,178,197,215,234,252,208,227,245,264,282,301,319,273,292,310,329,347,366,322,368,340,359,377,396,414,387,405,424,442,461,479,417,435,482,500,454,472,491,509,271]),   #alpha2
          'alpha4' :(9,9,[0,11,23,34,37,48,60,71,97,85,74,99,111,136,122,148,134,159]), #alpha4
          'alpha5' :(2,6,[0,15]), #alpha5
          'delta3' :(1,3,[0,4]),  #delta3
          'delta4' :(1,2,[0]),    #delta4
          'delta5' :(7,7,[0,16,11,32,52,73,93,36,57,77,48,27,68,89]),    #delta5
          'delta6' :(2, 2, []),   #delta6
          'chi2' :(3,6,[0,8,16,18,26,34]),  #chi2
          'chi3' :(1,5,[0,7]),  #chi3
          'chi4' :(6,6,[0,8,17,19,30,28,39,50,61,69,58,47]),  #chi4
          'beta4' :(4,3,[0,6,16]),    #beta4
          'beta5' :(5,3,[0,6,16,22]),    #beta5
          'beta8' :(1,9,[0,13]), #beta8
          'beta10':(1,4,[0]),    #beta10
          'beta11':(12,8,[0,4,19,28,43,53,68,77,92,102,117,126,141,151,166,175,190,24,23,47,49,64,73,88,98,113,122,137,147,162,171,186]),      #beta11
          'beta12':(1,3,[0]),    #beta12
          'beta13':(2,3,[0,4]),  #beta13
          'island1':(3,6,[0,4,5,30,31,32,35,24,11,6,23,29,28,21,10,17,22,14,26,2,7,20,1,12]),
          'island2':(3,6,[0,4,5,30,31,32,35,24,11,6,23,29,21,10,17,22,14,26,2,7,20,1,12,33,27,28]),
          'island3':(3,6,[0,4,5,30,31,32,33,34,35,24,11,6,23,29,28,18,9,21,10,17,22,14,26,2]),
          'island4':(3,6,[0,1,3,4,5,30,31,32,33,34,35,12,24,11,6,23,29,28,27,25,18,16,9,21,10,17,22]),
          'island5':(3,6,[0,1,2,3,4,5,30,31,32,33,34,35,12,24,11,6,23,29,28,27,26,25,18,16,9,21,10,17,22]),
          'island6':(3,6,[0,1,2,3,4,5,30,31,32,33,34,35,12,24,11,6,23,29,28,27,26,25,18,15]),
          'island7':(3,6,[0,1,2,3,4,5,30,31,32,33,34,35,12,24,11,6,23,29,28,27,26,25,18]),
          'island8':(3,6,[0,1,2,3,4,5,11,35,21,7,13,6,18,12,24,25,31,30]),
          'island9':(3,6,[1,2,3,5,6,7,8,9,10,11,12,13,15,16,17,18,19,25,26,27,28,23,29,30,31,32,35,21,20,34]),
          'island10':(3,6,[1,2,3,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,25,26,27,28,29,30,31,32,34,35]),
          'island11':(3,6,[0,1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,24,25,26,27,28,29,31,32,34,35])
          }
"""Dict of predefined borophene structures:  'name': nx, ny, [listholes]"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  FUNCTIONS DEFINITIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def write_lammps_input(
    inputfile       = "Boro1,3_0-Ag_fcc111_2,2,4.data", 
    outputfile      = "scripts/in.lmp",
    dumpname        = "structure",
    substrate       = "Ag",
    T0              = 10,
    T1              = 300,
    T2              = None,
    P               = 1.01325,
    replicate       = (4,2),
    runtime         = [20000,20000,20000],
    ts              = [1.0e-4,1.0e-2,1.0],
    damp            = [1.0, 10.0, 100.0],
    thermoprint     = 100,
    dumptime        = 100,
    MDtype          = "nvt",#or "npt" or "nve"
    showew          = "no",
    showewsum       = 1,
    maxew           = 200, 
    nnpCutoff       = 6.36,
    potpath         = "~/Boro_ML/MD/potentials/nnp-NPT-multiconfig/",
    fix_sub         = None,
    seed            = 123456,
    minimize        = True,
    ftol            = 0.01,
    outPE           = False,
    velocities      = False):
    """ Writes a LAMMPS input file from the given parameters (for batch writing of input files)
    
    #### Parameters
    inputfile       : str
        Input file name
    outputfile      : str
        Output file name (LAMMPS input file)
    substrate       : str
        Subtrate atomic element
    T0              : float
        Initial temperature (K)
    T1              : float
        Final temperature (K) 
    T2              : float
        If provided, add a second production run with the same parameters as before, going from T1 to T2.
    P               : float
        Pressure (atm)
    replicate       : (int, int)
        (Nx, Ny) replication of the input structure
    thermoprint     : int
        Print thermo every?
    dumptime        : int
        Outputs a structure every?
    runtime         : list of int
        List of Number of MD steps for thermalization and production
        Must be the same length as ts and damp
    ts              : list of floats
        List of time steps (fs) for thermalization and production
        Must be the same length as runtime and damp
    damp            : list of floats
        List of damping parameters for the thermostat for thermalization and production.
        Must be the same length as ts and runtime
    MDtype          : str
        Type of MD fix: "nvt" or "npt" or "nve. If 'nve', thermalization is performed in 'nvt' and production in 'nve'.
    showew          : "yes" or "no"
        Show Ewald sum computation during MD ?
    showewsum       : int
        Write EW summary every this many timesteps
    nnpCutoff       : float
        Cutoff for NNP
    potpath         : str
        Path to potential files folder
    fix_sub         : int
        If provided, fix in z the `fix_sub` first bottom layers of the substrate
    seed            : int
        Seed for velocities random generator
    minimize        : logical
        Minimize structure before MD run?
    ftol            : float
        Tolerance on forces for energy minimization
    velocities      : logical
        Save velocities in the dump files?
    outPE           : logical
        Save individual elergies in the dump file?
    """
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    if len(ts)!=len(damp) or len(ts)!=len(runtime) or len(damp)!=len(runtime):
        sys.exit("\nCheck that the ts, runtime and damp lists have identical lengths.\n")
    if fix_sub is not None:
        moving = "mobile"
    else:
        moving = "all"
        fix_sub = 1
    MDrun = ""
    prodtype = "nvt"
    if MDtype == 'nve':
        prodtype = 'nve'
        MDtype = 'nvt'
    for i,(dt,N,dp) in enumerate(zip(ts,runtime,damp)):
        if(i<len(ts)-1):
            jobname = f"equil{i+1}"
            MDrun += f"""\n#------- Thermalization {i+1} -------
dump           dmp{jobname} all custom 1000 ${{dumpname}}_{jobname}.lammpstrj id element x y z
dump_modify    dmp{jobname} flush yes sort id element B {substrate}"""
            mymd = f"fix         md{jobname} {moving} {MDtype} temp ${{T0}} ${{T0}} $({dp}*dt)"
        else:
            jobname = "prod"
            MDrun += f"""\n#------- Production -------
dump           dmp{jobname} all custom ${{dumptime}} ${{dumpname}}_{jobname}.lammpstrj id element x y z{" vx vy vz" if velocities else ""}{" c_pe" if outPE else ""}
dump_modify    dmp{jobname} flush yes sort id element B {substrate}
reset_timestep 0"""
            if prodtype!='nve':
                mymd = f"fix         md{jobname} {moving} {MDtype} temp ${{T0}} ${{T1}} $({dp}*dt)"
            else:
                mymd = f"fix         md{jobname} {moving} nve"
        if MDtype == "npt":
            mymd += f" iso ${{P}} ${{P}} $(1000.0*dt)"
        MDrun += f"""
{mymd}
timestep    {dt}
run         {N}
unfix       md{jobname}
undump      dmp{jobname}
"""
    if T2 is not None and prodtype!='nve':
        MDrun += f"""\n#------- Cooling Production -------
dump           dmp{jobname} all custom ${{dumptime}} ${{dumpname}}_{jobname}2.lammpstrj id element x y z{" vx vy vz" if velocities else ""}{" c_pe" if outPE else ""}
dump_modify    dmp{jobname} flush yes sort id element B {substrate}
reset_timestep 0"""
        mymd = f"fix         md{jobname} {moving} {MDtype} temp ${{T1}} ${{T2}} $({dp}*dt)"""
        if MDtype == "npt":
            mymd += f" iso ${{P}} ${{P}} $(1000.0*dt)"
        MDrun += f"""
{mymd}
timestep    {dt}
run         {N}
unfix       md{jobname}
undump      dmp{jobname}
"""
    mini = ""
    miniheader=""    
    if minimize:
        mini = f"""
#-−-−−---−---−−−− Cell optimization −−−−−−−−-------
thermo         10
dump minidmp   all custom 10 ${{dumpname}}_mini.lammpstrj id element x y z{" c_pe" if outPE else ""}
dump_modify    minidmp flush yes sort id element B {substrate}
fix            freeze fixedlayer setforce 0.0 0.0 0.0
minimize       0 ${{ftol}} 1000 10000
unfix          freeze
undump         minidmp
thermo         ${{thermoprint}}
reset_timestep 0
"""
        miniheader=f"""
# Minimization tolerance on Forces
variable ftol        equal {ftol}"""

    f = open(outputfile, "w")
    writeT2=""
    if T2 is not None:
        writeT2=f"""
# Very final temperature (K)
variable T2          equal {T2}"""
    f.write(f"""#−−−−−−−−−−−−−−−−−−− Parameters −−−−−−−−−−−−−−−−−−−
# Input structure file
variable inputfile   string {inputfile}
# Initial temperature (K)
variable T0          equal {T0}
# Final temperature (K)
variable T1          equal {T1}{writeT2}
# Pressure (atm)
variable P           equal {P}
# Dump every dumptime
variable dumptime    equal {dumptime}
# Print thermo every
variable thermoprint equal {thermoprint}{miniheader}
# Output trajectory file name
variable dumpname    string {dumpname}-${{T0}}K-${{T1}}K
# NNP
variable nnpCutoff   equal  {nnpCutoff}
variable nnpDir      string \"{potpath}\"

#−−−−−−−−−−−−−−−−−−− Initialize −−−−−−−−−−−−−−−−−−−
processors * * *
units      metal
boundary   p p p
atom_style atomic

#−−−−−−−−−−−−−−−−−−− Input −−−−−−−−−−−−−−−−−−−-----
read_data  ${{inputfile}} # read input structure
replicate  {replicate[0]} {replicate[1]} 1 # make supercell
group      borophene type 1
group      substrate type 2
region     regfixedlayer block INF INF INF INF INF {1+(fix_sub-1)*2.5} # define region where atoms are fixed
group      fixedlayer region regfixedlayer
group      mobile subtract all fixedlayer

#−−−−−−−−−−−−−−−−−−− Potential -−−−−−−−−−−−−−−−−−−−
pair_style nnp dir ${{nnpDir}} showew {showew} showewsum {showewsum} maxew {maxew} &
           cflength 1.8897261328 cfenergy 0.0367493254 emap "1:B,2:{substrate}"
pair_coeff * * ${{nnpCutoff}}

#−−−−−−−−−−−−−−−−−−−− Output −−−−−−−−−−−−−−−−−−−−−−
compute       mytemp {moving} temp # the temperature is computed on moving atoms only
thermo_style  custom step c_mytemp ke pe etotal press lx ly lz
thermo_modify norm yes # extensive quantities will be normalized per atom
thermo        ${{thermoprint}}
{"compute       pe all pe/atom" if outPE else '' }
{mini}

#−−−−−−−−−−−−−−−−−−−− MD Run ---−−−−−−−−−−−−−−−−−−−
# Create initial velocities
velocity   {moving} create ${{T0}} {seed}
{MDrun}

print \"Job done.\"

""")
    f.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # #


def write_lammps_slurm_job(jobname="my_job", node="node24", N=0, workingpath="~/Boro_ML/", append=False):
    """ 
    Write a job file for submission to SLURM
    #### Parameters
    
    jobname   : str
        Name of the job in SLURM's queue
    node : str
        Name of the node to submit the job to
    N    : int or float
        If int > 1, number of processors to use. If float in [0,1], proportion of the node to use.
    workingpath   : str
        Path to the working directory
    append : logical
        If True, append to the file, otherwise write just the header.
    """
    
    f = open(jobname, "a" if append else "w")
    
    if node == "l-node01":
        queue = "a22"
        nproc = 96
    elif node == "l-node02":
        queue = "a22"
        nproc = 96
    elif node == "l-node03":
        queue = "f22"
        nproc = 112
    elif node == "l-node04":
        queue = "p22"
        nproc = 128
    elif node == "l-node05":
        queue = "p22"
        nproc = 112
    elif node == "l-node06":
        queue = "p22"
        nproc = 128
    elif node == "node23":
        queue = "tc2"
        nproc = 40
    elif node == "node24":
        queue = "tc2"
        nproc = 40
    elif node == "node25":
        queue = "tc2"
        nproc = 40
    elif node == "node26":
        queue = "tc1"
        nproc = 40
    elif node == "node27":
        queue = "tc1"
        nproc = 40
    elif node == "node28":
        queue = "tc1"
        nproc = 40
    elif node == "node29":
        queue = "tc2"
        nproc = 40
    elif node == "node30":
        queue = "tc2"
        nproc = 40
    elif node == "node31":
        queue = "tc3"
        nproc = 48
    elif node == "node32":
        queue = "tc2"
        nproc = 40
    elif node == "node33":
        queue = "tc3"
        nproc = 96
    elif node == "node34":
        queue = "tc3"
        nproc = 48
    elif node == "node35":
        queue = "tc3"
        nproc = 48
    elif node == "node36":
        queue = "tc3"
        nproc = 48
    elif node == "node37":
        queue = "tc4"
        nproc = 80
    elif node == "node38":
        queue = "tc5"
        nproc = 128
    else:
        sys.exit(f"Node {node} not found in the list of nodes.")
    if N>0:
        if N>1:
            nproc = int(N)
        else:
            nproc = int(N*nproc)
    NTHREAD = 2
    nppt = int(nproc/NTHREAD)
    
    if append==False: # write the header
        f.write(f"""#!/bin/bash

#SBATCH -J {jobname}
#SBATCH -A theochem
#SBATCH -p {queue}
#SBATCH -N 1
#SBATCH -n {nproc}
#SBATCH -w {node}
#SBATCH -t UNLIMITED
#SBATCH -o {workingpath}/{jobname}.o
#SBATCH -e {workingpath}/{jobname}.e

source /home/cbousige/bin/n2p2/env.sh

""")
    else:
        # create a new directory for the job with the jobname and current date and time
        ct = f"{datetime.datetime.now()}".replace(" ", "_")
        mydir = f"{jobname}_{os.path.basename(workingpath)}_{ct}"
        f.write(f"""
echo \"Working on: {workingpath}\"
mkdir -p /scratch/$USER/{mydir} || exit 1
cd /scratch/$USER/{mydir}
PAT=\"{workingpath}\"
cp $PAT/input.lmp .
cp $PAT/*.data .
cp $PAT/input.nn .
mpirun -np {nppt} /home/cbousige/bin/lammps-27May2021/src/lmp_mpi -sf omp -pk omp {NTHREAD} < $PAT/input.lmp > $PAT/lammps.out
cp -p *.lammpstrj $PAT
cp -p *log* $PAT

""")
    f.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # #

def LAMMPS_to_POSCAR(
    inputfile  = "test.lammpstrj"):
    """Process cleartext lammps dumpfiles to convert each image as a POSCAR file

    #### Parameters
    inputfile: str
        Input file name (LAMMPS dump file) (can be binary? to check)
    """
    output = Path(inputfile).stem
    outputpath = Path(inputfile).parent
    f = open(inputfile, "r")
    atoms = read_lammps_dump(f, index=slice(0, None))
    f.close()
    os.makedirs(os.path.dirname(f"{outputpath}"), exist_ok=True)
    for i in range(len(atoms)):
        write(f"{outputpath}/{output}_{i}.POSCAR",
              atoms[i], format = "vasp", vasp5=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # #

def write_pdb(struct, name=""):
    """Write an Atom object as a PDB file with computed bonds"""
    a, b, c, alpha, beta, gamma = struct.cell.cellpar() 
    out  = "TITLE       " + name + "\n"
    out += "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1\n".format(a,b,c,alpha,beta,gamma)
    boron = struct[struct.get_atomic_numbers() == 5]
    metal = struct[struct.get_atomic_numbers() != 5]
    for i, (at, x, y, z) in enumerate(zip(
        boron.get_chemical_symbols(), boron.positions[:,0], boron.positions[:,1], boron.positions[:,2])):
        out += "ATOM  {:5d}  {:<4s}             {:8.3f}{:8.3f}{:8.3f}  1.00  1.00           {:s}\n".format(i+1,at,x,y,z,at)
    for i, (at, x, y, z) in enumerate(zip(
        metal.get_chemical_symbols(), metal.positions[:,0], metal.positions[:,1], metal.positions[:,2])):
        out += "ATOM  {:5d}  {:<4s}             {:8.3f}{:8.3f}{:8.3f}  1.00  1.00           {:s}\n".format(i+1+len(boron),at,x,y,z,at)
    pairs = np.array(list(itertools.combinations(range(len(boron)),2)))
    dr = distance.pdist(boron.positions)
    connected = pairs[dr<1.9]
    for i, j in connected:
        out +=  "CONECT{:5d}{:5d}\n".format(i+1,j+1)
        out += "ENDMDL\n"
    return out

# # # # # # # # # # # # # # # # # # # # # # # # # # #

def mindist(atoms):
    """Gets minimum distance between atoms in a Atom object"""
    dists = np.array(atoms.get_all_distances()).flatten()
    dists = dists[ dists > 0 ]
    return(min(dists))

# # # # # # # # # # # # # # # # # # # # # # # # # # #

def shuffle(atoms, random=.1, v=1):
    """Shuffle atoms positions in an Atom object
    #### Parameters:
    atoms : Atoms object
    random: float
        Maximum distance for the random displacement
    v     : float, but typically 0 or 1
        Vertical multiplicator for random motion (0 for horizontal motion only)
    """
    at = atoms.copy()
    for i in range(len(atoms.positions)):
        x_norm = 2.0
        while x_norm > 1:  # This is to ensure uniform distribution in space
            xi = np.random.uniform(-1, 1, 3)
            x_norm = np.linalg.norm(xi)
        e_unitaire = xi/x_norm
        ran = np.random.uniform(0, 1) * random
        at.positions[i, 0] = atoms.positions[i,0] + ran * e_unitaire[0]
        at.positions[i, 1] = atoms.positions[i,1] + ran * e_unitaire[1]
        at.positions[i, 2] = atoms.positions[i,2] + ran * e_unitaire[2]*v
    return(at)

# # # # # # # # # # # # # # # # # # # # # # # # # # #

def create_structure(
    nx=3, 
    ny=3, 
    listholes=[0, 10],
    repeatx=1,
    repeaty=1,
    metalchoice="Ag",
    surfchoice="fcc111",
    vdwdist=2.45,
    vac=25,
    angle=0,
    NZ=4,
    random=0,
    v=1,
    size_min=None,
    glimpse=True,
    dmin=1,
    Nboro=1,
    dmax=0,
    shiftX=0,
    shiftY=0,
    a=None,
    island_size=0.,
    island_shape='circle',
    island_angle=0.,
    island_rotate=0.
    ):
    """Create a borophene structure on a substrate
    
    #### Parameters
    nx         : int
        Number of base borophene cell along x
    ny         : int
        Number of base borophene cell along y
    listholes  : [int]
        List of atoms to remove
    repeatx    : int
        Repeat the borophene stucture along x
    repeaty    : int
        Repeat the borophene stucture along y
    metalchoice: str
        Element name for the substrate ("None" for no substrate, 
        otherwise: 'Ag', 'Au', 'Cu', 'Pt', 'Ni', 'Ir', 'Si')
    surfchoice : str
        Surface orientation ('fcc111', 'fcc110', 'fcc100', 'fcc211' or 'diamond100')
    vdwdist    : float
        Van der Waals distance (A) 
    vac        : float
        Force z dimension of the structure. If 0, fall back to periodic cell in z.
    angle      : float
        Rotation of the borophene structure (in degrees)
    NZ : int
        Number of layers of the substrate
    random : float
        Displace all atoms with a random vector of max size "random"
    v : 0 or 1
        Vertical multiplicator for random motion (0 for horizontal motion only)
    size_min : float
        If size_min is provided, size_min becomes the minimum size of the box in any direction. 
        If the size is smaller than size_min in a direction, the box is duplicated along this direction as long as this is the case.
    glimpse : logical
        Show a 3D view of the structure?
    Nboro : int
        Number of borophene layers (default: 1)
    dmin : float
        Minimum distance between atoms if random>0.
    dmax : float
        If dmax>0, must have at least 1 distance between dmin and dmax.
    shiftX,shiftY : float
        Shift borophene structure by (shiftX,shiftY) in Angstrom
    a : float
        If a is provided, use it as the lattice parameter of the metal slab, otherwise use the default one ({'Ag': 4.04, 'Au': 4.0782, 'Cu': 3.6149, 'Pt': 3.9242, 'Ni': 3.5240, 'Ir': 3.8390, 'Si': 5.4309}).
    island_size : float
        If island_size>0 is provided, remove all B atoms outside a surface of shape island_shape and size island_size centered on the substrate center.
    island_shape : str, one of ('circle', 'square', 'triangle', 'hexagon')
        If island_size>0 is provided, remove all B atoms outside a surface of shape island_shape and size island_size centered on the substrate center.
    island_angle : float
        Rotation of the borophene island (in degrees) with respect to the substrate
    island_rotate : float
        Rotation of the borophene island (in degrees) with respect to the borophene structure
    """
    # # # # # # # # # 
    # Create borophene polymorph
    # # # # # # # # #
    aa = 1.62*2*np.cos(30*np.pi/180)
    bb = 1.62
    cc = 2.5
    if len(listholes)>0:
        structbase = Atoms('BB', pbc=[1,1,0], 
                            positions = [(0,0,0),(aa/2,bb/2,0)], 
                            cell = [aa,bb,cc])
    else:
        structbase = Atoms('BB', pbc=[1,1,0], 
                            positions = [(0,0,0),(aa/2,bb/2,0.89)], 
                            cell = [aa,bb,cc])
    struct = make_supercell(structbase,[[nx,0,0],[0,ny,0],[0,0,1]])
    sortedpos = np.lexsort((struct.positions[:,1], struct.positions[:,0])) 
    struct.positions = struct.positions[sortedpos]
    # Remove atoms defined in holes list to get unit cell
    if listholes!=['']:
        listholes = [int(i) for i in listholes]
        del struct[listholes]
    # Replicate unit cell
    if size_min is not None and size_min>0:
        repeatx = int(np.round(size_min/struct.cell[0,0]))
        repeaty = int(np.round(size_min/struct.cell[1,1]))
    struct = make_supercell(struct, [[repeatx, 0, 0], [0, repeaty, 0], [0, 0, 1]])

    structa = struct.cell[0,0]
    structb = struct.cell[1,1]
    structc = struct.cell[2,2]
    if angle==90:
        struct.rotate(angle, 'z', rotate_cell=True)
        struct.set_cell([ [structb,0,0],
                          [0,structa,0],
                          [0, 0, structc] ], scale_atoms=False)
        struct.positions[:,0] = struct.positions[:,0] % structb
        struct.positions[:,1] = struct.positions[:,1] % structa
        structa = struct.cell[0,0]
        structb = struct.cell[1,1]
        structc = struct.cell[2,2]
    # Borophene island
    if island_size>0:
        # center all positions
        structa,structb = struct.cell.cellpar()[:2]
        struct.positions[:,0] -= structa/2
        struct.positions[:,1] -= structb/2
        if island_shape == 'circle':
            B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                        (at.position[0]**2+at.position[1]**2)>(island_size/2)**2]
        if island_shape == 'square':
            # square summits coordinates
            delta = island_size*np.sqrt(2)/2
            A = (delta*np.cos(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*0), delta*np.sin(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*0))
            B = (delta*np.cos(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*1), delta*np.sin(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*1))
            C = (delta*np.cos(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*2), delta*np.sin(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*2))
            D = (delta*np.cos(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*3), delta*np.sin(island_angle*np.pi/180 + np.pi/4 + 2*np.pi/4*3))
            ABslope = (A[1]-B[1])/(A[0]-B[0])
            BCslope = (B[1]-C[1])/(B[0]-C[0])
            CDslope = (C[1]-D[1])/(C[0]-D[0])
            DAslope = (D[1]-A[1])/(D[0]-A[0])
            ABintercept = A[1] - ABslope*A[0]
            BCintercept = B[1] - BCslope*B[0]
            CDintercept = C[1] - CDslope*C[0]
            DAintercept = D[1] - DAslope*D[0]
            # Remove atoms outside the square
            if island_angle == 0:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                        (np.abs(at.position[0])>=ABintercept or 
                         np.abs(at.position[1])>=ABintercept)]
            else:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] <= CDslope*at.position[0]+CDintercept or
                             at.position[1] >= DAslope*at.position[0]+DAintercept)]
        if island_shape == 'triangle':
            # Triangle summits coordinates
            delta = island_size/2/np.cos(np.pi/6)
            A = (delta*np.cos(island_angle*np.pi/180)            , delta*np.sin(island_angle*np.pi/180))
            B = (delta*np.cos(island_angle*np.pi/180 + 2*np.pi/3), delta*np.sin(island_angle*np.pi/180 + 2*np.pi/3))
            C = (delta*np.cos(island_angle*np.pi/180 + 4*np.pi/3), delta*np.sin(island_angle*np.pi/180 + 4*np.pi/3))
            ABslope = (A[1]-B[1])/(A[0]-B[0])
            BCslope = (B[1]-C[1])/(B[0]-C[0])
            ACslope = (A[1]-C[1])/(A[0]-C[0])
            ABintercept = A[1] - ABslope*A[0]
            BCintercept = B[1] - BCslope*B[0]
            ACintercept = A[1] - ACslope*A[0]
            # Remove atoms outside the triangle
            if island_angle == 0:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[0] <= -delta/2 or
                             at.position[1] <= ACslope*at.position[0]+ACintercept)]
            elif island_angle <= 60:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] <= ACslope*at.position[0]+ACintercept)]
            elif island_angle>60:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] >= ACslope*at.position[0]+ACintercept)]
        if island_shape == 'hexagon':
            # Triangle1 summits coordinates
            delta = island_size/2/np.cos(np.pi/6)
            A = (delta*np.cos(island_angle*np.pi/180)            , delta*np.sin(island_angle*np.pi/180))
            B = (delta*np.cos(island_angle*np.pi/180 + 2*np.pi/3), delta*np.sin(island_angle*np.pi/180 + 2*np.pi/3))
            C = (delta*np.cos(island_angle*np.pi/180 + 4*np.pi/3), delta*np.sin(island_angle*np.pi/180 + 4*np.pi/3))
            ABslope = (A[1]-B[1])/(A[0]-B[0])
            BCslope = (B[1]-C[1])/(B[0]-C[0])
            ACslope = (A[1]-C[1])/(A[0]-C[0])
            ABintercept = A[1] - ABslope*A[0]
            BCintercept = B[1] - BCslope*B[0]
            ACintercept = A[1] - ACslope*A[0]
            # Remove atoms outside the triangle1
            if island_angle == 0:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[0] <= -delta/2 or
                             at.position[1] <= ACslope*at.position[0]+ACintercept)]
            elif island_angle <= 60:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] <= ACslope*at.position[0]+ACintercept)]
            elif island_angle>60:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] >= ACslope*at.position[0]+ACintercept)]
            Btriangle1 = np.array([i for i in range(len(struct)) if i not in B_to_remove])
            # Triangle2
            delta = island_size/2/np.cos(np.pi/6)
            island_angle = island_angle + 60
            A = (delta*np.cos(island_angle*np.pi/180)            , delta*np.sin(island_angle*np.pi/180))
            B = (delta*np.cos(island_angle*np.pi/180 + 2*np.pi/3), delta*np.sin(island_angle*np.pi/180 + 2*np.pi/3))
            C = (delta*np.cos(island_angle*np.pi/180 + 4*np.pi/3), delta*np.sin(island_angle*np.pi/180 + 4*np.pi/3))
            ABslope = (A[1]-B[1])/(A[0]-B[0])
            BCslope = (B[1]-C[1])/(B[0]-C[0])
            ACslope = (A[1]-C[1])/(A[0]-C[0])
            ABintercept = A[1] - ABslope*A[0]
            BCintercept = B[1] - BCslope*B[0]
            ACintercept = A[1] - ACslope*A[0]
            # Remove atoms outside the triangle1
            if island_angle == 0:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[0] <= -delta/2 or
                             at.position[1] <= ACslope*at.position[0]+ACintercept)]
            elif island_angle <= 60:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] <= ACslope*at.position[0]+ACintercept)]
            elif island_angle>60:
                B_to_remove = [i for i,at in enumerate(struct) if at.symbol=='B' and 
                            (at.position[1] >= ABslope*at.position[0]+ABintercept or
                             at.position[1] <= BCslope*at.position[0]+BCintercept or
                             at.position[1] >= ACslope*at.position[0]+ACintercept)]
            Btriangle2 = np.array([i for i in range(len(struct)) if i not in B_to_remove])
            Bstar = np.intersect1d(Btriangle1, Btriangle2)
            B_to_remove = [i for i in range(len(struct)) if i not in Bstar]
        del struct[B_to_remove]
        struct.positions[:,0] += structa/2
        struct.positions[:,1] += structb/2
        if island_rotate>0:
            struct.rotate(island_rotate, 'z', center='COM', rotate_cell=False)
    # # # # # # # # #
    # Substrate
    # # # # # # # # #
    if NZ < 1:
        metalchoice = None
    if metalchoice in metals.keys():
        celpar = a if a is not None else metals[metalchoice]
        surf_fonc = funcmap[surfchoice]
        # Create slab
        try:
            slab = surf_fonc(metalchoice, size=(1,1,NZ), 
                    a = celpar, orthogonal = True)
        except ValueError:
            slab = surf_fonc(metalchoice, size=(1,2,NZ), 
                    a = celpar, orthogonal = True)
        # Define the new unit cell by matching borophene on the slab:
        # get slab parameters a and b
        slaba,slabb = slab.cell.cellpar()[:2]
        # get borophene parameters a, b and c
        structa,structb,structc = struct.cell.cellpar()[:3]
        # make slab supercell to match closely the borophene
        slab = make_supercell(
                slab, [[np.round(structa/slaba), 0, 0],
                       [0, np.round(structb/slabb), 0],
                       [0, 0, 1]])
        cella = slab.cell[0, 0]
        cellb = slab.cell[1, 1]
        cellc = slab.cell[2, 2]
        # Rescale borophene atoms positions to fit the slab
        if island_size == 0:
            struct.positions[:, 0] = struct.positions[:, 0] / structa * cella
            struct.positions[:, 1] = struct.positions[:, 1] / structb * cellb
        # define new cell parameters for the slab
        slab.set_cell([ [cella,0,0],
                        [0,cellb,0],
                        [0,0,cellc] ])
        # define new cell parameters for the borophene
        struct.set_cell([[cella,0,0],
                         [0,cellb,0],
                         [0,0,structc]])
        # define new structure with the borophene sheet on top of the slab
        struct.positions[:,2] = struct.positions[:,2] + max(slab.positions[:,2]) + vdwdist
        struct = struct + slab
        cpar = vac if vac>0 else max(struct.positions[:,2]) + vdwdist
        struct.set_cell([ [cella,0,0],
                          [0,cellb,0],
                          [0, 0, cpar] ], scale_atoms=False)
    else:
        struct.set_cell([ [structa,0,0],
                          [0,structb,0],
                          [0, 0, vac] ], scale_atoms=False)
    # Add random displacement if any
    if random > 0:
        randstruct = struct.copy()
        MIN = 0
        while(MIN < dmin):
            randstruct = shuffle(struct, random, v)
            MIN = mindist(randstruct)
            if (dmax > 0 and MIN > dmax):
                MIN = 0
        struct = randstruct.copy()
    # Add supplementary layers of borophene if any
    if Nboro > 1:
        added = 1
        Bid  = [i for i,at in enumerate(struct.get_chemical_symbols()) if at=='B']
        boro = struct.copy()[Bid]
        while added < Nboro:
            boro.positions[:,2] += vdwdist
            struct = boro + struct
            added += 1
    # Sort atoms to have B first
    struct = sort(struct, tags = struct.get_masses())
    # Shift borophene position with respect to substrate
    Bid = [i for i,at in enumerate(struct.get_chemical_symbols()) if at=='B']
    if np.abs(shiftX) > 0:
        struct.positions[Bid,0] += shiftX
        struct.positions[Bid,0] = struct.positions[Bid,0] % structa
    if np.abs(shiftY) > 0:
        struct.positions[Bid,1] += shiftY
        struct.positions[Bid,1] = struct.positions[Bid,1] % structb
    # View structure with ASE
    if glimpse:
        view(struct)
    return(struct)


# # # # # # # # # # # # # # # # # # # # # # # # # # #

def write_lammps(name, atoms,
                 units="real",
                 atom_style='atomic', comments=''):
    """Write atomic structure data to a LAMMPS data file.
    #### Parameters
    name      : str
        Name of the output file
    atoms     : Atom object
        ASE Atom object
    units     : str
        LAMMPS units style
    atom_style: str
        LAMMPS atom style
    """
    if name==None:
        fd = sys.stdout
    else:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        fd = open(name, "w")
        comments=name
    
    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    if hasattr(fd, "name"):
        fd.write(f"{comments}\n\n")
    else:
        fd.write("\n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write("{0} \t atoms \n".format(n_atoms))

    species = set(symbols)
    species.remove('B')
    species = ['B'] + sorted(species)
    n_atom_types = len(species)
    fd.write("{0}  atom types\n".format(n_atom_types))

    p = Prism(atoms.get_cell())

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    fd.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    fd.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    fd.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if p.is_skewed():
        fd.write(
            "{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
    fd.write("\n\n")

    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write("Masses \n\n")
    for i in range(n_atom_types):
        m = Atoms(species[i]).get_masses()[0]
        fd.write(str(i+1) + "   " + str(m) + "\n")

    fd.write("\n\nAtoms \n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)

    if atom_style == 'atomic':
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n"
                     .format(*(i + 1, s, q) + tuple(r)))
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        # The label 'mol-id' has apparenlty been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has('mol-id'):
            molecules = atoms.get_array('mol-id')
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " mol-id dtype must be subtype of np.integer, and"
                    " not {:s}.").format(str(molecules.dtype)))
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " each atom must have exactly one mol-id."))
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferableabove assigning all atoms to a single molecule id per
            # default, as done within ase <= v 3.19.1. I.e.,
            # molecules = np.arange(start=1, stop=len(atoms)+1, step=1, dtype=int)
            # However, according to LAMMPS default behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of molecule
            #    assignments.

        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} "
                     "{6:23.17g}\n".format(*(i + 1, m, s, q) + tuple(r)))
    else:
        raise NotImplementedError

    fd.flush()
    if fd is not sys.stdout:
        fd.close()



