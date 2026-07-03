import numpy as np
from ase import Atoms
import datetime
from pathlib import Path
import argparse
import ast
import re
from itertools import combinations

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--Nadd", type=int, default="30", help="--Nadd=10 (default) : nb of structure to be added each iteration")
    parser.add_argument("--Nepoch", type=int, default="50", help="--Nepoch=20 (default) : nb of the NNPs training epochs")
    parser.add_argument("--request", type=str, default='init', help='--request=init (default), ScalTrainPred, CompEfSSDFT, DFT_vasp : action requested to the adaptive learning')
    parser.add_argument("--stepNB", type=str, default='0', help='--stepNB=0 (default) adaptive learning step number, used to name the output files')
    parser.add_argument("--nnodes", type=int, default='2', help='--nnodes=2 (default) nb of nodes used for this job')
    parser.add_argument("--minimize", type=str, default="energy", help="--minimize=energy (default), forces : the quantity to compare between NNPs during the adaptive")
    parser.add_argument("--GammaPoint", type=str, default="False", help="--GammaPoint=False (default), : Will use vasp_std or vasp_gam executables")
    parser.add_argument("--ds_increment", type=int, default="0", help="--ds_increment=0 (default), : Allows to Redo a CompEfSSDFT with a new dataset increment")
    args = parser.parse_args()
    return args

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def check_minimize(minimize: str):
    if minimize.lower() not in ["energy", "forces",'e','f','energies','force']:
        raise ValueError("minimize must be either 'energy' or 'forces'")
    if minimize.lower() in ['e', 'energies']:
        minimize = "energy"
    if minimize.lower() in ['f', 'force']:
        minimize = "forces"
    return minimize


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def readlogfile(logfile: str):
    with open(logfile, "r") as f:
        lines = f.read().splitlines()
    get_indices = False
    pairs_list = []
    for line in lines:
        if "Nb of added structures per iteration" in line:
            Nadd = int(line.split()[-1])
        if "Nb of epoch for training" in line:
            Nepoch = int(line.split()[-1])
        if "quantity to minimize" in line:
            minimize = (line.split()[-1])
        if "ITERATION" in line:
            ITERATION = int(line.split()[-1])
        if get_indices:
            i_line += " "+line
            if "]." in line:
                start = i_line.find(':[')
                end = i_line.find('].')
                if start != -1 and end != -1:
                    string = line[start+2:end]
                    pairs = re.findall(r"\('([^']+)', (\d+)\)", string)
                    pairs_list = [(pair[0], int(pair[1])) for pair in pairs]
                get_indices = False
        if "selected structures" in line:
            # Extracts the part of the line within square brackets
            if ":[" in line and "]." in line:
                start = line.find(':[')
                end = line.find('].')
                if start != -1 and end != -1:
                    string = line[start+2:end]
                    pairs = re.findall(r"\('([^']+)', (\d+)\)", string)
                    pairs_list = [(pair[0], int(pair[1])) for pair in pairs]
            else:
                i_line = line
                get_indices = True
            
    return(Nadd, Nepoch, minimize, ITERATION, pairs_list)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def read_inputdata(path, Nstock):
    """
    Read input.data file and return a list of Atoms objects and the comments
    """
    def read_data_block(lines, begin: int, end: int):
        """
        Read block of structure in input.data that have been read in to `lines`
        """
        lattice = np.array([[i for i in lines[j].split(' ') if i][1:4] for j in range(begin+2, begin+5)], dtype=float)
        pos = np.array([[i for i in lines[j].split(' ') if i][1:4] for j in range(begin+5, end-2)], dtype=float)
        symb = np.array([[i for i in lines[j].split(' ') if i][4] for j in range(begin+5, end-2)])
        return(Atoms(symbols   = symb,
                     positions = pos*0.529177, #convert to Å
                     cell      = lattice*0.529177, pbc=True))
    # # # # # # # # # # # # 
    atoms, energies, ids, compids, comments = [], [], [], [], []
    for i in range(Nstock):
        filename = (f"{path}/stock{i+1}.data")
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        begin = np.array([i for i, x in enumerate(lines) if x[0:5] == 'begin'])
        end = np.array([i for i, x in enumerate(lines) if x[0:3] == 'end'])
        comments += [x for x in lines if 'comment' in x]
        atoms += [read_data_block(lines, b, e) for b,e in zip(begin, end)]
        Es = np.array([float(x.split()[1]) for x in lines if 'energy' in x])
        Enull_ids = np.where(Es != 0)[0]
        energies += Es.tolist()
        ids += [(f"stock{i+1}", index) for index in range(len(begin))]
        for j in Enull_ids:
            compids += [(f"stock{i+1}", j)]
    return(atoms, energies, ids, compids, comments)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def write_trainingcsvfile(nNNP, path, stepNB, minimize):
    Ncomb = sum(1 for _ in combinations(range(nNNP), 2))
    with open(f"{path}/{stepNB}_training.csv", 'w') as f:
        f.write("###########################################\n")
        f.write("# Logfile for the iterative training\n")
        f.write("###########################################\n")
        f.write("#  1 : N                  : Number of structures in dataset\n")
        RMSE, dEmean, dEstd = [], [], []
        j = 1
        for i in range(1,nNNP+1):
            f.write(f"# {j:2d} : RMSE_E{i} (meV/at)   : energies RMSE from training NNP{i}\n")
            j += 1
            f.write(f"# {j:2d} : RMSE_F{i} (meV/Å)    : forces RMSE from training NNP{i}\n")
            j += 1
            RMSE += [(f'RMSE_E{i}', f'RMSE_F{i}')]
        k=1
        for i,j in combinations(range(1, nNNP+1), 2):
            if minimize == "energy":
                f.write(f"# {nNNP+j+k:2d} : dEmean{i}-{j} (meV/at) : mean(Ennp{i}-Ennp{j}) for all structures in stock\n")
                dEmean += [f'dEmean{i}-{j}']
            if minimize == "forces":
                f.write(f"# {nNNP+j+k:2d} : dFmean{i}-{j} (meV/Å)  : mean(Fnnp{i}-Fnnp{j}) for all structures in stock\n")
                dEmean += [f'dFmean{i}-{j}']
            k += 1
        k=1
        for i,j in combinations(range(1, nNNP+1), 2):
            if minimize == "energy":
                f.write(f"# {nNNP+Ncomb+1+k:2d} : dEstd{i}-{j} (meV/at)  : std(Ennp{i}-Ennp{j}) for all structures in stock\n")
                dEstd += [f'dEstd{i}-{j}']
            if minimize == "forces":
                f.write(f"# {nNNP+Ncomb+1+k:2d} : dFstd{i}-{j} (meV/at)  : std(Fnnp{i}-Fnnp{j}) for all structures in stock\n")
                dEstd += [f'dFstd{i}-{j}']
            k += 1
        f.write("###########################################\n")
        f.write(f"""N,{','.join(f"{r[0]},{r[1]}" for r in RMSE)},{','.join(dEmean)},{','.join(dEstd)}\n""")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def print_success_message():
    """
    Prints a super cool success message. Absolutely *not geeky*.
    """
    now = datetime.datetime.now()
    print(f"""
****************************************************************
* OMFG!!! The job finished without any problem !!!             *
* This is a godlike achievement, you get a beer on my account! *
* Time of ending: {now.strftime('%Y-%m-%d %H:%M:%S')}                          *
****************************************************************
""", flush=True)