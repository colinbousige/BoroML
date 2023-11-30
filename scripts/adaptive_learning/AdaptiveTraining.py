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
from .functions import read_inputdata, write_inputdata
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
    `copy_weights(self, best1:int, best2:int)`
        Copy weight files from training to predict folder, as well as scaling.data and input.nn
    `do_NNPs(self)`
        Perform the, scaling, training and prediction of both NNPs and wait until it's done. The saved weights are the ones of the last epoch.
    `do_predict(self)`
        Compute energies of all remaining stock structures with both NNPs and wait until it's done
    `do_scaling(self)`
        Perform the scaling of both NNPs and wait until it's done
    `do_training(self)`
    `do_vasp(self, indexes, Nnodes=4)`
        Perform the training of both NNPs and wait until it's done
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
                 Nepoch: int = None
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
                    print(line.replace("write_weights_epoch ", f"write_weights_epoch {self.Nepoch} #"), end='')
        self.Nstock  = self.get_Nstruct(stock = True)
        """Number of structures in stock initially."""
        self.Niter   = int(self.Nstock / self.Nadd)
        """Total number of iterations to perform."""
        self.stock = self.read_stock()
        """Dict of {id, comment, structures} of stock structures."""
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
                for i in range(1,self.Nnnp+1):
                    f.write(f"# {i+1:2d} : RMSE{i} (meV/at)     : RMSE from training NNP{i}\n")
                    RMSE += [f'RMSE{i}']
                k=1
                for i,j in combinations(range(1, self.Nnnp+1), 2):
                    f.write(f"# {self.Nnnp+1+k:2d} : dEmean{i}-{j} (meV/at) : mean(Ennp{i}-Ennp{j}) for all structures in stock\n")
                    dEmean += [f'dEmean{i}-{j}']
                    k += 1
                k=1
                for i,j in combinations(range(1, self.Nnnp+1), 2):
                    f.write(f"# {self.Nnnp+Ncomb+1+k:2d} : dEstd{i}-{j} (meV/at)  : std(Ennp{i}-Ennp{j}) for all structures in stock\n")
                    dEstd += [f'dEstd{i}-{j}']
                    k += 1
                f.write("###########################################\n")
                f.write(f"N,{','.join(RMSE)},{','.join(dEmean)},{','.join(dEstd)}\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def get_Nstruct(self, copy=False, stock=False):
        """
        Get number of structures in /NNP1/train/input.data. If stock=True, rather count structures in /NNP1/predict/input.data. If copy=True, save a copy of input.data in `inputs/input_N.data`
        """
        if stock == False:
            Nstruct=subprocess.run(f"grep 'begin' {self.path}/NNP1/train/input.data | wc -l", 
                                shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
            if copy:
                subprocess.run(f"cp {self.path}/NNP1/train/input.data {self.path}/inputs/input_{Nstruct}.data", shell=True)
        else:
            Nstruct=subprocess.run(f"grep 'begin' {self.path}/NNP1/predict/input.data | wc -l", 
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
        print(f"", flush=True)
        for i in range(1, self.Nnnp+1):
            data = pd.read_table(f'{self.path}/NNP{i}/train/learning-curve.out', comment='#', sep=r'\s{2,}', 
                            engine='python', usecols=[0,1], names=['epoch','RMSE'], header=None)
            best += [data.RMSE.argmin()]
            rmse += [data.RMSE.min()*1000*27.21138469]
            print(f"""NNP{i}/train: BestEpoch = {best[i-1]} – RMSE = {rmse[i-1]:.3f}""", flush=True)
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
                            engine='python', usecols=[0,1], names=['epoch','RMSE'], header=None)
            rmse += [data.RMSE.to_list()[-1]*1000*27.21138469]
        return(rmse)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def copy_weights(self, epoch):
        """
        Copy weight files of epochs `[epoch]` from training to predict folder, as well as `scaling.data` and `input.nn`
        """
        print(f"├─ Copying weights from epochs {epoch}...\n│", flush=True)
        for i in range(1, self.Nnnp+1):
            nnppath=f'{self.path}/NNP{i}'
            subprocess.run(f'cp {nnppath}/train/weights.047.{epoch[i-1]:06d}.out {nnppath}/train/weights.047.data', shell=True)
            subprocess.run(f'cp {nnppath}/train/weights.005.{epoch[i-1]:06d}.out {nnppath}/train/weights.005.data', shell=True)
            subprocess.run(f'cp {nnppath}/train/weights.005.data {nnppath}/predict/', shell=True)
            subprocess.run(f'cp {nnppath}/train/weights.047.data {nnppath}/predict/', shell=True)
            subprocess.run(f'cp {nnppath}/train/input.nn {nnppath}/predict/', shell=True)
            subprocess.run(f'cp {nnppath}/train/scaling.data {nnppath}/predict/', shell=True)
            # subprocess.run(f'mv {nnppath}/train/input.nn.bak {nnppath}/train/input.nn', shell=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def plot_RMSE_training(self, i=1, save=True):
        """
        Plot RMSE of training of NNPs
        """
        fig, ax = plt.subplots()
        ax.set_title(f"Training NNPs – Step {i}", fontweight="bold")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Energies RMSE [meV/at]')
        ax.grid()
        ax.set_ylim([0, 200])
        for j in range(1, self.Nnnp+1):
            data = pd.read_table(f'{self.path}/NNP{j}/train/learning-curve.out', 
                                comment='#', sep=r'\s{2,}', engine='python', 
                                usecols=[0,1], names=['epoch','RMSE'], header=None)
            plt.plot(data.epoch, data.RMSE*27.21138469*1000, label=f"NNP{j}")
        plt.legend(loc='upper right')
        if save:
            plt.savefig(f"{self.path}/plots/NNPtrainings_{i:05d}.jpg", dpi=300)
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def plot_convergence(self, save=True):
        """
        Plot convergence of adaptative training
        """
        data = pd.read_csv(f'{self.path}/training.csv', comment='#')
        fig, (ax1,ax2) = plt.subplots(1,2, tight_layout=True, figsize=(8, 4))
        fig.suptitle(f"Convergence plot", fontweight="bold")
        ax1.set_title("$|\Delta E|$", fontweight="bold")
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('$|\Delta E|$ [meV/at]')
        ax1.grid()
        ax1.set_ylim([0, 5*min(data['dEmean1-2'])])
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            ax1.errorbar(data.N, data[f'dEmean{i}-{j}'], yerr=data[f'dEstd{i}-{j}'], label=f'$NNP{i}-NNP{j}$')

        ax2.set_title("$std(\Delta E)$", fontweight="bold")
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('$std(\Delta E)$ [meV/at]')
        ax2.grid()
        ax2.set_ylim([0, 5*min(data['dEstd1-2'])])
        for i,j in combinations(range(1, self.Nnnp+1), 2):
            ax2.plot(data.N, data[f'dEstd{i}-{j}'], label=f'$NNP{i}-NNP{j}$')

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
        rmse  = ','.join(f"{r:.6e}" for r in RMSE)
        de    = ','.join(f"{r:.6e}" for r in dEmean)
        destd = ','.join(f"{r:.6e}" for r in dEstd)
        with open(f"{self.path}/training.csv", 'a') as f:
            f.write(f"{N},{rmse},{de},{destd}\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def plot_distrib(self, data, i=0, save=True):
        """
        Plot histogram of E_{NNP2} - E_{NNP1} to `plots/histogram_{i:05d}.jpg`
        """
        Tot = self.Ncomb
        Cols = 1
        Rows = Tot // Cols 
        if Tot % Cols != 0:
            Rows += 1
        # # # # # 

        fig,axes = plt.subplots(Rows,Cols, tight_layout=True, sharex=True, sharey=True, figsize=(8, Tot*2.5))
        fig.suptitle(f'Iteration {i}')
        l = 0
        if Tot == 1:
            axes = [axes]
        for (k,j),ax in zip(combinations(range(1, self.Nnnp+1), 2), axes):
            ax.set_xlabel('$E_{NNP_i} - E_{NNP_j}$ [meV/atom]')
            x = data[f'Ennp{k}'] - data[f'Ennp{j}']
            ax.hist(x, 50, facecolor=list(mcolors.TABLEAU_COLORS)[l], rwidth=.98, log=True, label=f'$E_{{NNP{k}}} - E_{{NNP{j}}}$')
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
        data = []
        for i in range(self.Nnnp):
            colnames=["index",f"Ennp{i+1}"] 
            d = pd.read_table(fE[i], comment='#', sep=r"\s{2,}", usecols=[0,2], 
                    engine="python", names=colnames, header=None)
            data += [d]
            mean_energy, conv_energy = read_input(finput[i])
            d[f"Ennp{i+1}"] = denorm(d[f"Ennp{i+1}"], mean_energy, conv_energy)
        # Merge tables and convert to meV/atom
        data = reduce(lambda df1,df2: pd.merge(df1, df2, on = 'index'), data)
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
    
    def do_scaling(self):
        """
        Perform the scaling of both NNPs and wait until it's done
        """
        scale = []
        for i in range(1, self.Nnnp+1):
            scale += [SlurmJob(
                    type    = 'nnp-scaling', 
                    path    = f"{self.path}/NNP{i}/train", 
                    cluster = self.clusterNNP,
                    jobname = f"scale{i}", 
                    launch  = True)]
        while any([s.is_running() for s in scale]) == True:
            time.sleep(10)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_training(self):
        """
        Perform the training of both NNPs and wait until it's done
        """
        train = []
        for i in range(1, self.Nnnp+1):
            train += [SlurmJob(
                    type    = 'nnp-train', 
                    path    = f"{self.path}/NNP{i}/train", 
                    cluster = self.clusterNNP,
                    jobname = f"train{i}", 
                    launch  = True)]
        while any([t.is_running() for t in train]) == True:
            time.sleep(10)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def do_NNPs(self):
        """
        Perform the, scaling, training and prediction of both NNPs and wait until it's done. The saved weights are the ones of the last epoch.
        """
        nnps = []
        for i in range(1, self.Nnnp+1):
            nnps += [SlurmJob(
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
    
    def do_predict(self):
        """
        Compute energies of all remaining stock structures with both NNPs and wait until it's done
        """
        self.copy_weights([self.Nepoch]*self.Nnnp)
        self.inputnn2predict()
        predict = []
        for i in range(1, self.Nnnp+1):
            predict += [SlurmJob(
                    type    = 'nnp-train', 
                    path    = f"{self.path}/NNP{i}/predict", 
                    cluster = self.clusterNNP,
                    jobname = f"predict{i}", 
                    launch  = True)]
        while any([p.is_running() for p in predict]) == True:
            time.sleep(10)
        print(f"└──▶︎ Done at {time.strftime('%H:%M:%S on %Y-%m-%d', time.gmtime())}\n│",flush=True)
    
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
                    type    = "vasp", 
                    path    = f"{self.path}/vasp", 
                    cluster = self.clusterVASP,
                    jobname = f"vasp{i+1}", 
                    numbers = todo[int(i*len(todo)/Nnodes):int((i+1)*len(todo)/Nnodes)],
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
        for i in range(len(atoms)):
            write(f"{self.path}/EW/POSCAR_{i}", atoms[i], format = "vasp", vasp5 = True)
        todo = [i for i in range(len(atoms)) if not Path(f"{self.path}/EW/OUTCAR_{i}.inp").is_file()]
        if len(todo)>0:
            vasp = []
            Nnodes = self.vasp_Nnodes if len(todo)>=self.vasp_Nnodes else len(todo)
            for i in range(Nnodes):
                vasp += [SlurmJob(
                    type    = "vasp", 
                    path    = f"{self.path}/EW", 
                    cluster = self.clusterVASP,
                    jobname = f"vasp{i+1}", 
                    numbers = todo[int(i*len(todo)/Nnodes):int((i+1)*len(todo)/Nnodes)],
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
