import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import glob
import os
import time
import fileinput
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ase.io import write
from functools import reduce
from itertools import combinations
from .functions import read_inputdata, write_inputdata, print_success_message
from .SlurmJob import SlurmJob
from .Cluster import Cluster


class AdaptiveTraining:
    """
    A class to perform an Adaptive Training
    
    ## Parameters
    `path`: str. Default = `os.getcwd()`
        Working directory
    `Nadd`: int. Default = 20
        Structures to add at each iteration
    `restart`: Bool. Default = False
        If `restart=True`, set to restart from previously stopped job, so do not copy files but just check they are here.
    `cluster`: Cluster object
        Cluster object
    `vasp_Nnodes`: int. Default = 4
        Number of nodes to distribute vasp jobs to
    `Nepoch`: int
        Numbers of epochs in the training part, read from 'input1.nn', or overwrite this from the supplied value when itializing the class.
    `Niter`: int
        Total number of iterations to perform.
    `Nnnp`: int
        Numbers of NNPs: gotten from the number of 'inputX.nn' files
    `Ncomb`: int
        Numbers of combinations of NNPs: automatic determination
    `Nstock`: int
        Number of structures in stock initially.

    ## Methods
    `add_ew(self)`
        Compute with VASP all structures in `EW.data` and add them to the dataset. This `EW.data` file is created with the script `xLAMMPStoNNP`.
    `compare_energies(self)`
        Compare energies computed by NNP1 and NNP2.
    `compare_forces(self)`
        Compare forces computed by NNP1 and NNP2.
    `copy_weights(self)`
        Copy weight files from training to predict folder, as well as scaling.data and input.nn
    `do_NNPs(self)`
        Perform the, scaling, training and prediction of both NNPs and wait until it's done. The saved weights are the ones of the last epoch.
    `do_predict(self)`
        Compute energies of all remaining stock structures with both NNPs and wait until it's done
    `do_scaling(self, wait=.2)`
        Perform the scaling of both NNPs and wait until it's done
    `do_training(self)`
        Perform the training of both NNPs and wait until it's done
    `check_training(self)`
        Check that the jobs actually worked, and redo them if not
    `check_predict(self)`
        Check that the jobs actually worked, and redo them if not
    `do_vasp(self, indexes, Nnodes=4)`
        Perform the vasp calculations on the structures with indexes `indexes` and wait until it's done
    `get_best(self)`
        Get best epoch and RMSE from a training `learning-curve.out` file in `self.path`
    `get_epochs(self)`
        Get number of training epochs from an input.nn file
    `get_last_rmse(self)`
        Get RMSE from last epoch from a training `learning-curve.out` file in `self.path`
    `get_Nstruct(self, copy=False, stock=False)`
        Get number of structures in /NNP1/train/input.data. If stock=True, rather count structures in /NNP1/predict/input.data. If copy=True, save a copy of input.data in `inputs/input_N.data`
    `get_POSCARs(self, numbers=[])`
        Write POSCAR files for structures in the list `numbers`. Return the POSCAR indexes for easy retrieval later.
    `initialize(self)`
        Create arborescence, copy necessary files in folders, and make sure the necessary files are there. If `self.restart=True`, set to restart from previously stopped job, so do not copy files but just check they are here.
    `inputnn2predict(self)`
        Make an `input.nn` file ready for prediction on many structures in parallel
    `plot_convergence(self, save=True)`
        Plot convergence of adaptative training
    `plot_distrib(self, data, i=0, save=True)`
        Plot histogram of E_{NNP2} - E_{NNP1} to `plots/histogram_{i:05d}.jpg`
    `plot_RMSE_training(self, i=1, save=True)`
        Plot RMSE of training of NNP1 and NNP2
    `update_stock(self, numbers=[])`
        Remove structures in the list `numbers` from a stock `input.data` when they are added to the dataset.
    `write2log(self, N:int, RMSE1:float, RMSE2:float, dEmean:float,`
        Write N, RMSE1, RMSE2, dEmean, dEstd to `self.path/training.csv`
    
    ## Usage (see sample script in `adaptive_training.py`)
    >>> from adaptive_learning import *
    >>> training = AdaptiveTraining(restart=False)
    """
    
    def __init__(self, 
                 path        = os.getcwd(),
                 Nadd        = 20,
                 restart     = False,
                 clusterNNP  = Cluster(),
                 clusterVASP = Cluster(),
                 vasp_Nnodes = 4,
                 Nepoch: int = None,
                 atoms       = "BAg",
                 minimize    = "energy",
                 Kpoints     = "7 7 1",
                 ENCUT       = 700
                 ):
        self.path = path
        """Working directory"""
        self.Nadd = Nadd
        """Structures to add at each iteration"""
        self.restart = restart
        """If `restart=True`, set to restart from previously stopped job, so do not copy files but just check they are here."""
        self.clusterNNP = clusterNNP
        """Cluster object for NNP computations"""
        self.clusterVASP = clusterVASP
        """Cluster object for VASP computations"""
        self.vasp_Nnodes = vasp_Nnodes
        """Number of nodes to distribute vasp jobs to. Default=4"""
        self.Nnnp = len(glob.glob1(self.path,"input*.nn"))
        """Numbers of NNPs: gotten from the number of 'inputX.nn' files"""
        self.Ncomb = sum(1 for _ in combinations(range(self.Nnnp), 2))
        """Numbers of combinations of NNPs: automatic determination from self.Nnnp"""
        self.atoms = atoms
        """Atom types in the system"""
        self.minimize = minimize
        """During the adaptive training, the quantity to compare between NNP1 and NNP2. 
        If "energy", the energy difference is minimized. If "force", the force difference is minimized."""
        self.initialize()
        if Nepoch is None:
            self.Nepoch  = self.get_epochs()
            """Numbers of epochs in the training part, read from 'input1.nn', or overwrite this from the supplied value when itializing the class."""
        else:
            self.Nepoch = Nepoch
            for i in range(1, self.Nnnp+1):
                for line in fileinput.input(f"{self.path}/NNP{i}/train/input.nn", inplace = 1): 
                    print(line.replace("epochs ", f"epochs {self.Nepoch} #"), end='')
                for line in fileinput.input(f"{self.path}/NNP{i}/train/input.nn", inplace = 1): 
                    print(line.replace("write_weights_epoch ", f"write_weights_epoch 1 #"), end='')
        self.Nstock  = self.get_Nstruct(stock = True)
        """Number of structures in stock initially."""
        self.Niter   = int(self.Nstock / self.Nadd)
        """Total number of iterations to perform."""
        self.stock = self.read_stock()
        """Dict of {id, comment, structures} of stock structures."""
        self.Kpoints   = Kpoints
        """Kpoints for VASP calculations"""
        self.ENCUT     = ENCUT
        """Energy cutoff for VASP calculations"""
        print("""─────────────────────────────────────────────────────────────
*   Adaptive construction of the dataset for NNP training   *
─────────────────────────────────────────────────────────────""", flush=True)
        print(self.__str__())
        if self.restart == False and Path(f"{self.path}/EW.data").is_file():
            self.add_ew()
    
    def __repr__(self):
        return(self.__str__())
    
    def __str__(self):
        return(f"""├▶︎ Working directory    = {self.path}
├▶︎ Restart              = {self.restart}
├▶︎ Initial dataset size = {self.get_Nstruct()}
├▶︎ Structures in stock  = {self.Nstock}
├▶︎ N NNPs               = {self.Nnnp}
├▶︎ Nadd/iteration       = {self.Nadd}
├▶︎ Nepoch               = {self.Nepoch}
├▶︎ Nnodes for VASP      = {self.vasp_Nnodes}
├▶︎ Kpoints for VASP     = {self.Kpoints}
├▶︎ ENCUT for VASP       = {self.ENCUT}
└▶︎ Number of iterations = {self.Niter}
─────────────────────────────────────────────────────────────

""")
    
    
    def initialize(self):
        """
        Create arborescence, copy necessary files in folders, and make sure the necessary files are there. If `self.restart=True`, set to restart from previously stopped job, so do not copy files but just check they are here.
        """
        if self.Nnnp >= 2 and \
                Path(f"{self.path}/initial_input.data").is_file() and \
                Path(f"{self.path}/stock.data").is_file():
            Path(f"{self.path}/inputs").mkdir(parents=True, exist_ok=True)
            Path(f"{self.path}/plots").mkdir(parents=True, exist_ok=True)
            Path(f"{self.path}/vasp").mkdir(parents=True, exist_ok=True)
            Path(f"{self.path}/MDtests").mkdir(parents=True, exist_ok=True)
            Path(f"{self.path}/finaltraining").mkdir(parents=True, exist_ok=True)
            subprocess.run(f"cp {self.path}/input1.nn {self.path}/finaltraining/input.nn", shell=True)
            subprocess.run(f"cp /home/cbousige/Boro_ML/bin/xtrain {self.path}/finaltraining/", shell=True)
            if self.restart == False:
                for i in range(1, self.Nnnp+1):
                    Path(f"{self.path}/NNP{i}/predict").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.path}/NNP{i}/train").mkdir(parents=True, exist_ok=True)
                    subprocess.run(f"cp {self.path}/input{i}.nn {self.path}/NNP{i}/train/input.nn", shell=True)
                    subprocess.run(f"cp {self.path}/input{i}.nn {self.path}/NNP{i}/predict/input.nn", shell=True)
                    subprocess.run(f"cp {self.path}/initial_input.data {self.path}/NNP{i}/train/input.data", shell=True)
                    subprocess.run(f"cp {self.path}/stock.data {self.path}/NNP{i}/predict/input.data", shell=True)
                    for line in fileinput.input(f"{self.path}/NNP{i}/train/input.nn", inplace = 1): 
                        print(line.replace("test_fraction ", "test_fraction 0 #"), end='')
        else:
            print(f"""Make sure the following files are in the {self.path} folder:
    - `input1.nn`: input.nn for NNP1
    - `input2.nn`: input.nn for NNP2
    - `stock.data`: input.data file with all stock structures
    - `initial_input.data`: initial input.data structures dataset
    """, flush=True)
            exit()
        if self.restart == False:
            Ncomb = sum(1 for _ in combinations(range(self.Nnnp), 2))
            with open(f"{self.path}/training.csv", 'w') as f:
                f.write("###########################################\n")
                f.write("# Logfile for the iterative training\n")
                f.write("###########################################\n")
                f.write("#  1 : N                  : Number of structures in dataset\n")
                RMSE, dEmean, dEstd = [], [], []
                j = 1
                for i in range(1,self.Nnnp+1):
                    f.write(f"# {j:2d} : RMSE_E{i} (meV/at)   : energies RMSE from training NNP{i}\n")
                    j += 1
                    f.write(f"# {j:2d} : RMSE_F{i} (meV/Å)    : forces RMSE from training NNP{i}\n")
                    j += 1
                    RMSE += [(f'RMSE_E{i}', f'RMSE_F{i}')]
                k=1
                for i,j in combinations(range(1, self.Nnnp+1), 2):
                    if self.minimize == "energy":
                        f.write(f"# {self.Nnnp+j+k:2d} : dEmean{i}-{j} (meV/at) : mean(Ennp{i}-Ennp{j}) for all structures in stock\n")
                        dEmean += [f'dEmean{i}-{j}']
                    if self.minimize == "forces":
                        f.write(f"# {self.Nnnp+j+k:2d} : dFmean{i}-{j} (meV/Å)  : mean(Fnnp{i}-Fnnp{j}) for all structures in stock\n")
                        dEmean += [f'dFmean{i}-{j}']
                    k += 1
                k=1
                for i,j in combinations(range(1, self.Nnnp+1), 2):
                    if self.minimize == "energy":
                        f.write(f"# {self.Nnnp+Ncomb+1+k:2d} : dEstd{i}-{j} (meV/at)  : std(Ennp{i}-Ennp{j}) for all structures in stock\n")
                        dEstd += [f'dEstd{i}-{j}']
                    if self.minimize == "forces":
                        f.write(f"# {self.Nnnp+Ncomb+1+k:2d} : dFstd{i}-{j} (meV/at)  : std(Fnnp{i}-Fnnp{j}) for all structures in stock\n")
                        dEstd += [f'dFstd{i}-{j}']
                    k += 1
                f.write("###########################################\n")
                f.write(f"""N,{','.join(f"{r[0]},{r[1]}" for r in RMSE)},{','.join(dEmean)},{','.join(dEstd)}\n""")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def get_Nstruct(self, copy=False, stock=False):
        """
        Get number of structures in /NNP1/train/input.data. If stock=True, rather count structures in /NNP1/predict/input.data. If copy=True, save a copy of input.data in `inputs/input_N.data`
        """
        if stock == False:
            Nstruct=subprocess.run(f"grep -c 'begin' {self.path}/NNP1/train/input.data", 
                                shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
            if copy:
                subprocess.run(f"cp {self.path}/NNP1/train/input.data {self.path}/inputs/input_{Nstruct}.data", shell=True)
        else:
            Nstruct=subprocess.run(f"grep -c 'begin' {self.path}/NNP1/predict/input.data", 
                                shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
        return(int(Nstruct))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def inputnn2predict(self):
        """
        Make an `input.nn` file ready for prediction on many structures in parallel:
        - Replace the number of epochs by 0
        - Make test fraction 0
        - Use old weights
        - Remove normalization
        """
        for i in range(1, self.Nnnp+1):
            for line in fileinput.input(f"{self.path}/NNP{i}/predict/input.nn", inplace = 1): 
                print(line.replace("epochs ", "epochs 0 #"), end='')
            for line in fileinput.input(f"{self.path}/NNP{i}/predict/input.nn", inplace = 1): 
                print(line.replace("test_fraction ", "test_fraction 0 #"), end='')
            for line in fileinput.input(f"{self.path}/NNP{i}/predict/input.nn", inplace = 1): 
                print(line.replace("#use_old_weights_short", "use_old_weights_short"), end='')
            for line in fileinput.input(f"{self.path}/NNP{i}/predict/input.nn", inplace = 1): 
                print(line.replace("normalize_data_set", "#normalize_data_set"), end='')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def get_best(self):
        """
        Get best epoch and RMSE from a training `learning-curve.out` file in `self.path`
        """
        best = []
        rmse = []
        for i in range(1, self.Nnnp+1):
            data = pd.read_table(f'{self.path}/NNP{i}/train/learning-curve.out', comment='#', sep=r'\s{2,}', 
                            engine='python', usecols=[0,1,9], names=['epoch','RMSE_E','RMSE_F'], header=None)
            best += [data.RMSE.argmin()]
            rmse += [(data.RMSE_E.min()*1000*27.21138469, data.RMSE_F[data.RMSE.argmin()]*27211.38469/0.529177)]
        print(f"", flush=True)
        return(best, rmse)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def get_last_rmse(self):
        """
        Get RMSE from last epoch from a training `learning-curve.out` file in `self.path`
        """
        rmse = []
        for i in range(1, self.Nnnp+1):
            data = pd.read_table(f'{self.path}/NNP{i}/train/learning-curve.out', comment='#', sep=r'\s{2,}', 
                            engine='python', usecols=[0,1,9], names=['epoch','RMSE_E','RMSE_F'], header=None)
            rmse += [(data.RMSE_E.to_list()[-1]*1000*27.21138469, data.RMSE_F.to_list()[-1]*27211.38469/0.529177)]
        return(rmse)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def copy_weights(self):
        """
        Copy weight files of epochs `[Nepoch]` from training to predict folder, as well as `scaling.data` and `input.nn`
        """
        BestRmse = []
        for i in range(1, self.Nnnp+1):
            data = pd.read_table(f'{self.path}/NNP{i}/train/learning-curve.out', comment='#', sep=r'\s{2,}', 
                            engine='python', usecols=[0,1], names=['epoch','RMSE'], header=None)
            # locate the epoch with minimum RMSE
            BestRmse += [data.RMSE.argmin()]
        print(f"├─ Copying weights from epochs {BestRmse}...\n│", flush=True)
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            if self.atoms == "BAg":
                subprocess.run(f'cp {nnppath}/train/weights.047.{BestRmse[i-1]:06d}.out {nnppath}/train/weights.047.data', shell=True)
                subprocess.run(f'cp {nnppath}/train/weights.047.data {nnppath}/predict/', shell=True)
            if self.atoms == "BAu":
                subprocess.run(f'cp {nnppath}/train/weights.079.{BestRmse[i-1]:06d}.out {nnppath}/train/weights.079.data', shell=True)
                subprocess.run(f'cp {nnppath}/train/weights.079.data {nnppath}/predict/', shell=True)
            subprocess.run(f'cp {nnppath}/train/weights.005.{BestRmse[i-1]:06d}.out {nnppath}/train/weights.005.data', shell=True)
            subprocess.run(f'cp {nnppath}/train/weights.005.data {nnppath}/predict/', shell=True)
            subprocess.run(f'cp {nnppath}/train/input.nn {nnppath}/predict/', shell=True)
            subprocess.run(f'cp {nnppath}/train/scaling.data {nnppath}/predict/', shell=True)
            # subprocess.run(f'mv {nnppath}/train/input.nn.bak {nnppath}/train/input.nn', shell=True)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def reset_train(self):
        """
        Remove old files to allow checking the new jobs actually ran
        """
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            subprocess.run(f'rm -f {nnppath}/train/*.out', shell=True)
            subprocess.run(f'rm -f {nnppath}/train/function.data', shell=True)
            subprocess.run(f'rm -f {nnppath}/train/scaling.data', shell=True)
            subprocess.run(f'rm -f {nnppath}/train/nnp-*log*', shell=True)
            subprocess.run(f'rm -f {nnppath}/train/evsv.dat', shell=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def reset_predict(self):
        """
        Remove old files to allow checking the new jobs actually ran
        """
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            subprocess.run(f'rm -f {nnppath}/predict/*.out', shell=True)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def plot_RMSE_training(self, i:int=1, save:bool=True, forces:bool=False):

        """
        Plot RMSE of training of NNPs
        Parameters:
        -----------
        i: int, Adaptive learning Iteration 
        save: bool, save the learning curves of the of the NNP's through the training (single Adaptive learning iteration)
        forces: bool, if True beside the RMSE of energies, the RMSE of forces for the NNP's is also recorded.
        Return:
        ------
        fig : matplotlib.figure
        """

        if self.minimize == "force":
            forces = True

        ncols = 2 if forces else 1
        usecols = [0, 1, 9] if forces else [0,1] 
        names = ['epoch','RMSE_E', 'RMSE_F'] if forces else ['epoch','RMSE_E']
        fig, ax = plt.subplots(ncols=ncols, figsize=(12,6))
        fig.suptitle(f"Training NNPs – Step {i}", fontweight="bold")
        fig.supxlabel('Epochs')
	
        if ncols == 2:
            ax[0].set_ylabel('Energies RMSE [meV/at]')
            ax[1].set_ylabel('Forces RMSE [meV/Å]')
            for j in [0, 1]:
                ax[j].grid()
        else:
            ax.set_ylabel('Energies RMSE [meV/at]')
            ax.grid()
            ax.set_ylim([0, 200])

        for j in range(1, self.Nnnp+1):
            data = pd.read_table(f'{self.path}/NNP{j}/train/learning-curve.out', 
                                comment='#', sep=r'\s{2,}', engine='python',
                                usecols=usecols, names=names, header=None) 
            # remove first epoch to zoom in on the graph
            data = data.iloc[1:]
            if forces:
                ax[0].plot(data.epoch, data.RMSE_E*27.21138469*1000, label=f"NNP{j}")
                ax[1].plot(data.epoch, data.RMSE_F*27211.38469/0.529177, label=f"NNP{j}")
            else:
                ax.plot(data.epoch, data.RMSE_E*27.21138469*1000, label=f"NNP{j}")
        
        Line, Label = ax[0].get_legend_handles_labels() if forces else ax.get_legend_handles_labels()

        fig.legend(Line, Label, loc='upper right')
        
        if save:
            plt.savefig(f"{self.path}/plots/NNPtrainings_{i:05d}.jpg", dpi=300)
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def plot_convergence(self, save=True, type="energy"):
        """
        Plot convergence of adaptative training
        """
        typ = 'E' if type == "energy" else 'F'
        data = pd.read_csv(f'{self.path}/training.csv', comment='#')
        fig, (ax1,ax2) = plt.subplots(1,2, tight_layout=True, figsize=(8, 4))
        fig.suptitle(f"Convergence plot", fontweight="bold")
        ax1.set_title(f"$|\Delta {typ}|$", fontweight="bold")
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel(f'$|\Delta {typ}|$ [meV/at]')
        ax1.grid()
        ax1.set_ylim([0, max(data[f'd{typ}mean1-2'])])
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            ax1.plot(data.N, data[f'd{typ}mean{i}-{j}'], 
                     label=f'$NNP{i}-NNP{j}$')

        ax2.set_title(f"$std(\Delta {typ})$", fontweight="bold")
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel(f'$std(\Delta {typ})$ [meV/at]')
        ax2.grid()
        ax2.set_ylim([0, 5*min(data[f'd{typ}std1-2'])])
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            ax2.plot(data.N, data[f'd{typ}std{i}-{j}'], 
                     label=f'$NNP{i}-NNP{j}$')

        Line, Label = ax1.get_legend_handles_labels()
        Tot = self.Ncomb
        fig.legend(Line, Label, 
                   loc='lower center', ncol=int(Tot/2),
                   bbox_to_anchor=(0.5,-0.15))
        if save:
            plt.savefig(f"{self.path}/convergence.jpg", dpi=300, bbox_inches='tight')
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def write2log(self, N:int, RMSE, dEmean, dEstd):
        """
        Write N, RMSE, dEmean, dEstd to `training.csv`
        """
        rmse  = ','.join(f"{r[0]:.6e},{r[1]:.6e}" for r in RMSE)
        de    = ','.join(f"{r:.6e}" for r in dEmean)
        destd = ','.join(f"{r:.6e}" for r in dEstd)
        with open(f"{self.path}/training.csv", 'a') as f:
            f.write(f"{N},{rmse},{de},{destd}\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def plot_distrib(self, data, i=0, save=True, type="energy"):
        """
        Plot histogram of E_{NNP2} - E_{NNP1} to `plots/histogram_{i:05d}.jpg`
        """
        Tot = self.Ncomb
        Cols = 1
        Rows = Tot // Cols 
        if Tot % Cols != 0:
            Rows += 1
        # # # # # 
        typ = 'E' if type == "energy" else 'F'
        
        fig,axes = plt.subplots(Rows,Cols, tight_layout=True, sharex=True, sharey=True, figsize=(8, Tot*2.5))
        fig.suptitle(f'Iteration {i}')
        l = 0
        if Tot == 1:
            axes = [axes]
        for (k,j),ax in zip(combinations(range(1, self.Nnnp+1), 2), axes):
            if type == "energy":
                ax.set_xlabel(f'${typ}_{{NNP_i}} - {typ}_{{NNP_j}}$ [meV/atom]')
            else:
                ax.set_xlabel(f'${typ}_{{NNP_i}} - {typ}_{{NNP_j}}$ [meV/Å]')
            x = data[f'{typ}nnp{k}'] - data[f'{typ}nnp{j}']
            ax.hist(x, 50, facecolor=list(mcolors.TABLEAU_COLORS)[l], rwidth=.98, log=True, label=f'${typ}_{{NNP{k}}} - {typ}_{{NNP{j}}}$')
            ax.set_ylabel('Count')
            ax.grid()
            ax.legend(loc='upper right')
            l += 1

        for ax in plt.gcf().axes:
            try:
                ax.label_outer()
            except:
                pass

        if save:
            data.iloc[:, :(self.Nnnp+1)].to_csv(f"{self.path}/plots/histogram_{i:05d}.csv", index=False)
            plt.savefig(f"{self.path}/plots/histogram_{i:05d}.jpg", dpi=300)
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def compare_energies(self):
        """
        Compare energies computed by all NNPs. Returns [dEmean], [dEstd], [numbers] on all combinations, where `numbers` is the indexes of the structures (in input.data) sorted in decreasing order of energy differences per combination of NNPs.
        """
        def read_input(filename: str):
            """
            Read normalisation constants from input.nn
            """
            with open(filename, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                if "mean_energy" in line:
                    mean_energy = float(line.split()[1])
                if "conv_energy" in line:
                    conv_energy = float(line.split()[1])
            return(mean_energy, conv_energy)
        # # # # # # # # 
        def denorm(E: float, mean_energy: float, conv_energy: float)-> float:
            """
            Return de-normalized Energies in meV/atom
            """
            return((E/conv_energy + mean_energy)*27.21138469*1000)
        # # # # # # # # 
        fE = [f"{self.path}/NNP{i+1}/predict/trainpoints.000000.out" for i in range(self.Nnnp)]
        finput = [f"{self.path}/NNP{i+1}/predict/input.nn" for i in range(self.Nnnp)]
        # read energies for NNP1
        colnames=["index",f"Ennp1"] 
        data = pd.read_table(fE[0], comment='#', sep=r"\s{2,}", 
                            usecols=[0,2], engine="python", 
                            names=colnames, header=None)
        mean_energy, conv_energy = read_input(finput[0])
        data[f"Ennp1"] = denorm(data[f"Ennp1"], mean_energy, conv_energy)
        # append Ennp{i} for all other NNPs
        for i in range(1, self.Nnnp):
            colnames=["index",f"Ennp{i+1}"] 
            d = pd.read_table(fE[i], comment='#', sep=r"\s{2,}", 
                              usecols=[0,2], engine="python", 
                              names=colnames, header = None)
            mean_energy, conv_energy = read_input(finput[i])
            d[f"Ennp{i+1}"] = denorm(d[f"Ennp{i+1}"], mean_energy, conv_energy)
            data[f"Ennp{i+1}"] = d[f"Ennp{i+1}"]
        # Compute Energy Differences in meV/atom
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            data[f'dif{i}-{j}'] = np.abs(data[f"Ennp{i}"] - data[f"Ennp{j}"])
        dEmean, dEstd, numbers = [], [], []
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            dEmean  += [np.mean(data[f'dif{i}-{j}'])]
            dEstd   += [np.std(data[f'dif{i}-{j}'])]
            Esorted = data.sort_values(by=[f'dif{i}-{j}'], ascending=False)
            numbers += [np.array(Esorted.index)]
        return(dEmean, dEstd, numbers, data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def compare_forces(self):
        """
        Compare forces computed by all NNPs. Returns [dFmean], [dFstd], [numbers] on all combinations, 
        where `numbers` is the indexes of the structures (in input.data) sorted in decreasing order of forces differences per combination of NNPs.
        """
        def read_input(filename: str):
            """
            Read normalisation constants from input.nn
            """
            with open(filename, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                if "conv_energy" in line:
                    conv_energy = float(line.split()[1])
                if "conv_length" in line:
                    conv_length = float(line.split()[1])
            return(conv_energy, conv_length)
        # # # # # # # # 
        def denormF(F: float, conv_energy: float, conv_length: float)-> float:
            """
            Return de-normalized Forces in eV/Å
            """
            return(F*conv_length/conv_energy*27211.38469/0.529177) # further check meV/A
        # # # # # # # # 
        fF = [f"{self.path}/NNP{i+1}/predict/trainforces.000000.out" for i in range(self.Nnnp)]
        finput = [f"{self.path}/NNP{i+1}/predict/input.nn" for i in range(self.Nnnp)]
        colnames  = ["index_s"]
        colnames += [f"Fnnp{i}" for i in range(1,self.Nnnp+1)]                
        data = pd.DataFrame({c : [] for c in colnames}) # empty DataFrame
        # append Fnnp{i} for all other NNPs
        for i in range(self.Nnnp):
            colnames=["index_s", "index_a", f"Fnnp{i+1}"]
            d = pd.read_csv(fF[i], delim_whitespace=True, usecols=[0,1,3],
                            comment="#", skiprows=13, names=colnames)
            d.sort_values(by=['index_s', 'index_a'], inplace=True)
            conv_energy, conv_length = read_input(finput[i])
            d[f"Fnnp{i+1}"] = denormF(d[f"Fnnp{i+1}"], conv_energy, conv_length)
            data[f"Fnnp{i+1}"] = d[f"Fnnp{i+1}"].to_numpy()                     # Here I append the columns by np.ndarray objects to maintain the index_s order
        data["index_s"] = d[f"index_s"].to_numpy() # take the index from the last DataFrame they are the same and in increasing order    
        # Compute Forces Differences in meV/Å
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            data[f'dif{i}-{j}'] = np.abs(data[f"Fnnp{i}"] - data[f"Fnnp{j}"])
        # Drop Fnnp{i} columns
        #data = data.drop(columns=[f'Fnnp{i}' for i in range(1, self.Nnnp+1)]).drop(columns=['index_a']) # we need them for plot_distribution
        # group data by index_s and compute mean of all dif{i}-{j} columns
        datagrouped = data.groupby('index_s').mean()
        dFmean, dFstd, numbers = [], [], []
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            dFmean  += [np.mean(datagrouped[f'dif{i}-{j}'])]
            dFstd   += [np.std(datagrouped[f'dif{i}-{j}'])]
            Fsorted = datagrouped.sort_values(by=[f'dif{i}-{j}'], ascending=False)
            numbers += [np.array(Fsorted.index)]
        return(dFmean, dFstd, numbers, datagrouped)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def update_stock(self, numbers=[]):
        """
        Remove structures in the list `numbers` from a stock `input.data` when they are added to the dataset.
        """
        # # # # # # # # # # # # 
        # Remove unwanted structures
        newat = [a for i,a  in enumerate(self.stock['structures']) if i not in numbers]
        newco = [c for i,c  in enumerate(self.stock['comments']) if i not in numbers]
        newid = [c for i,c  in enumerate(self.stock['id']) if i not in numbers]
        self.stock = {'id'        : newid, 
                      'structures': newat, 
                      'comments'  : newco}
        for j in range(1, self.Nnnp+1):
            write_inputdata(f"{self.path}/NNP{j}/predict/input.data", 
                            newat, newco)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def read_stock(self):
        """
        Read initial stock file, and copy structures already computed at DFT level in /vasp
        """
        # # # # # # # # # # # # 
        atoms, comments = read_inputdata(filename  = f"{self.path}/NNP1/predict/input.data",
                                         copy_data = f"{self.path}/vasp")
        id = [i for i in range(len(atoms))]
        return({'id':id, 
                'structures':atoms, 
                'comments':comments})
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def get_POSCARs(self, numbers=[]):
        """
        Write POSCAR files for structures in the list `numbers`. Return the POSCAR indexes for easy retrieval later.
        """
        outputfolder = f"{self.path}/vasp"
        for i in numbers:
            write(f"{outputfolder}/POSCAR_{self.stock['id'][i]}", 
                  self.stock['structures'][i], 
                  format = "vasp", vasp5 = True)
        return([self.stock['id'][i] for i in numbers])
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_scaling(self, wait=3):
        """
        Perform the scaling of all NNPs and wait until it's done
        """
        self.reset_train()
        scale = []
        for i in range(1, self.Nnnp+1):
            scale += [SlurmJob(
                    atoms   = self.atoms,
                    type    = 'nnp-scaling', 
                    path    = f"{self.path}/NNP{i}/train", 
                    cluster = self.clusterNNP,
                    jobname = f"scale{i}", 
                    launch  = True,
                    wait    = wait)]
        while any([s.is_running() for s in scale]) == True:
            time.sleep(2)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
        self.check_scaling(wait)
        self.reset_predict()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def check_scaling(self, wait=3):
        """
        Check that the jobs actually worked, and redo them if not
        """
        todo = []
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            if not Path(f"{nnppath}/train/scaling.data").is_file():
                todo += [i]
        if len(todo)>0:
            print(f"├─▶︎ Scaling of NNPs[{todo}] failed, relaunching them...\n│",flush=True)
            scale = []
            for i in todo:
                scale += [SlurmJob(
                    atoms   = self.atoms,
                    type    = 'nnp-scaling', 
                    path    = f"{self.path}/NNP{i}/train", 
                    cluster = self.clusterNNP,
                    jobname = f"scale{i}", 
                    launch  = True,
                    wait    = wait)]
            while any([s.is_running() for s in scale]) == True:
                time.sleep(1)
            print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
            self.check_scaling(wait)
        else:
            print(f"├─▶︎ Scaling of all NNPs successful.\n│",flush=True)
            return(True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_training(self, wait=3):
        """
        Perform the training of both NNPs and wait until it's done
        """
        train = []
        for i in range(1, self.Nnnp+1):
            train += [SlurmJob(
                    atoms   = self.atoms,
                    type    = 'nnp-train', 
                    path    = f"{self.path}/NNP{i}/train", 
                    cluster = self.clusterNNP,
                    jobname = f"train{i}", 
                    launch  = True,
                    wait    = wait)]
        while any([t.is_running() for t in train]) == True:
            time.sleep(2)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
        self.check_training()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def check_training(self):
        """
        Check that the jobs actually worked, and redo them if not
        """
        todo = []
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            if not Path(f"{nnppath}/train/learning-curve.out").is_file():
                todo += [i]
        if len(todo)>0:
            print(f"├─▶︎ Training of NNPs[{todo}] failed, relaunching them...\n│",flush=True)
            train = []
            for i in todo:
                train += [SlurmJob(
                        atoms   = self.atoms,
                        type    = 'nnp-train', 
                        path    = f"{self.path}/NNP{i}/train", 
                        cluster = self.clusterNNP,
                        jobname = f"train{i}", 
                        launch  = True)]
            while any([t.is_running() for t in train]) == True:
                time.sleep(10)
            print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
            self.check_training()
        else:
            print(f"├─▶︎ Training of all NNPs successful.\n│",flush=True)
            return(True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_NNPs(self):
        """
        Perform the, scaling, training and prediction of both NNPs and wait until it's done. The saved weights are the ones of the last epoch.
        """
        nnps = []
        for i in range(1, self.Nnnp+1):
            nnps += [SlurmJob(
                    atoms   = self.atoms,
                    type    = 'nnp-all', 
                    path    = f"{self.path}/NNP{i}", 
                    Nepoch  = self.Nepoch,
                    cluster = self.cluster,
                    jobname = f"NNP{i}", 
                    launch  = True)]
        while any([nnp.is_running() for nnp in nnps]) == True:
            time.sleep(10)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_predict(self, wait=3):
        """
        Compute energies of all remaining stock structures with both NNPs and wait until it's done
        """
        self.copy_weights()
        self.inputnn2predict()
        predict = []
        for i in range(1, self.Nnnp+1):
            predict += [SlurmJob(
                    atoms   = self.atoms,
                    type    = 'nnp-train', 
                    path    = f"{self.path}/NNP{i}/predict", 
                    cluster = self.clusterNNP,
                    jobname = f"predict{i}", 
                    launch  = True,
                    wait    = wait)]
        while any([p.is_running() for p in predict]) == True:
            time.sleep(2)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
        self.check_predict()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def check_predict(self):
        """
        Check that the jobs actually worked, and redo them if not
        """
        todo = []
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            if not Path(f"{nnppath}/predict/testpoints.000000.out").is_file():
                todo += [i]
        if len(todo)>0:
            print(f"├─▶︎ Predict of NNPs[{todo}] failed, relaunching them...\n│",flush=True)
            predict = []
            for i in todo:
                predict += [SlurmJob(
                    atoms   = self.atoms,
                    type    = 'nnp-train', 
                    path    = f"{self.path}/NNP{i}/predict", 
                    cluster = self.clusterNNP,
                    jobname = f"predict{i}", 
                    launch  = True)]
            while any([p.is_running() for p in predict]) == True:
                time.sleep(10)
            print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
            self.check_predict()
        else:
            print(f"├─▶︎ Predict of all NNPs successful.\n│",flush=True)
            return(True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_vasp(self, indexes):
        """
        Perform VASP calculation on structures with indexes in list `indexes`. Distribute the jobs on `self.vasp_Nnodes` nodes if possible (i.e. if `len(indexes)>=4`), and wait until it's done.
        """
        # In case of restart, check for OUTCAR_x.inp already present to avoid recomputing them
        todo = [i for i in indexes if not Path(f"{self.path}/vasp/OUTCAR_{i}.inp").is_file()]
        print(f"├─ {len(indexes)} structures indexes to add: {indexes}", flush=True)
        print(f"├─ ...of which {len(todo)} to compute: {todo}", flush=True)
        if len(todo)>0:
            vasp = []
            Nnodes = self.vasp_Nnodes if len(todo)>=self.vasp_Nnodes else len(todo)
            for i in range(Nnodes):
                vasp += [SlurmJob(
                    atoms   = self.atoms,
                    type    = "vasp", 
                    path    = f"{self.path}/vasp", 
                    cluster = self.clusterVASP,
                    jobname = f"vasp{i+1}", 
                    numbers = todo[int(i*len(todo)/Nnodes):int((i+1)*len(todo)/Nnodes)],
                    Kpoints = self.Kpoints,
                    ENCUT   = self.ENCUT,
                    launch  = True)]
            while any([v.is_running() for v in vasp]) == True:
                time.sleep(10)
        print(f"├─▶︎ Adding OUTCAR_xx.inp to NNP[1-{self.Nnnp}]/train/input.data...",flush=True)
        for i in indexes:
            for j in range(1, self.Nnnp+1):
                subprocess.run(f"cat {self.path}/vasp/OUTCAR_{i}.inp >> {self.path}/NNP{j}/train/input.data", shell=True)
        print(f"│  └─▶︎ Now there are {self.get_Nstruct()} structures in the dataset.",flush=True)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def add_ew(self):
        """
        Compute with VASP all structures in `EW.data` and add them to the dataset.
        This `EW.data` file is created with the script `xLAMMPStoNNP`.
        """
        Path(f"{self.path}/EW").mkdir(parents=True, exist_ok=True)
        atoms = read_inputdata(f"{self.path}/EW.data")[0]
        print(f"├─────▶︎ Computing the EW structures to add them to the dataset",flush=True)
        print(f"├─▶︎ Found {len(atoms)} EW structures",flush=True)
        todo = [i for i in range(len(atoms)) if not Path(f"{self.path}/EW/OUTCAR_{i}.inp").is_file()]
        print(f"├─ ...of which {len(todo)} to compute: {todo}", flush=True)
        for i in range(len(atoms)):
            write(f"{self.path}/EW/POSCAR_{i}", atoms[i], format = "vasp", vasp5 = True)
        if len(todo)>0:
            vasp = []
            Nnodes = self.vasp_Nnodes if len(todo)>=self.vasp_Nnodes else len(todo)
            for i in range(Nnodes):
                vasp += [SlurmJob(
                    atoms   = self.atoms,
                    type    = "vasp", 
                    path    = f"{self.path}/EW", 
                    cluster = self.clusterVASP,
                    jobname = f"vasp{i+1}", 
                    numbers = todo[int(i*len(todo)/Nnodes):int((i+1)*len(todo)/Nnodes)],
                    Kpoints = self.Kpoints,
                    ENCUT   = self.ENCUT,
                    launch  = True)]
            while any([v.is_running() for v in vasp]) == True:
                time.sleep(10)
        print(f"├─▶︎ Adding OUTCAR_[0-{len(atoms)-1}].inp to NNP[1-{self.Nnnp}]/train/input.data...",flush=True)
        for i in range(len(atoms)):
            for j in range(1, self.Nnnp+1):
                subprocess.run(f"cat {self.path}/EW/OUTCAR_{i}.inp >> {self.path}/NNP{j}/train/input.data", shell=True)
        print(f"│  └─▶︎ Now there are {self.get_Nstruct()} structures in the dataset.",flush=True)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
        print(f"│",flush=True)
        print(f"│",flush=True)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def get_epochs(self):
        """
        Get number of training epochs from an input.nn file
        """
        with open(f'{self.path}/input1.nn', 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            if 'epochs ' in line:
                out = line.split(' ')
        self.Nepoch = int([o for o in out if o][1])
        return(int([o for o in out if o][1]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def run(self):
        """
        Run the adaptive training
        """
        for i in range(1, self.Niter+1):
            Nstruct = self.get_Nstruct(copy=True)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            print(f"""├─────▶︎ ITERATION {i} / {self.Niter} ◀︎──────
│
├─▶︎ {Nstruct} structures in input.data
├─▶︎ {self.get_Nstruct(stock=True)} structures in stock
│""", flush=True)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            print(f"├───▶︎ Training {self.Nnnp} NNPs\n│", flush=True)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # Perform the, scaling, training and prediction of both NNPs and wait until it's done
            # self.do_NNPs() # sometimes fails for whatever reason...
            self.do_scaling()
            self.do_training()
            self.do_predict()
            # Get RMSEs
            RMSE = self.get_last_rmse()
            # Compare energies/forces
            if self.minimize == "energy":
                dmean, dstd, numbers, quantities = self.compare_energies()
            elif self.minimize == "forces":
                dmean, dstd, numbers, quantities = self.compare_forces()
            else:
                print("\n\nERROR: `minimize` must be either 'energy' or 'forces'\n\nExiting...\n\n")
                exit(1)
            # Output to log
            self.write2log(Nstruct, RMSE, dmean, dstd)
            # Make plots
            self.plot_RMSE_training(i, forces=True)
            self.plot_distrib(quantities, i, type = self.minimize)
            if i > 1:
                self.plot_convergence(type = self.minimize)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            print(f"│\n├───▶︎ Making VASP calculations on the {self.Nadd} structures with largest E difference\n│", flush=True)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # Compute the self.Nadd first structures for each combination of NNP
            nums = np.unique(np.concatenate([nu[:self.Nadd] for nu in numbers]).ravel())
            indexes = self.get_POSCARs(nums)
            self.do_vasp(indexes)
            # Structures up to numbers[:to_add] have been added, remove them from stock:
            to_add = self.Nadd
            struct_done = np.unique(np.concatenate([nu[:to_add] for nu in numbers]).ravel())
            self.update_stock(struct_done)

        # It's done!
        print_success_message()




