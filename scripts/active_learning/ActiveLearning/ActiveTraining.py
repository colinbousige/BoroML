import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import time
import glob
import fileinput
import matplotlib
matplotlib.use('Agg')  # Utilise un backend non-graphique
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ase.io import write
from ase.calculators.vasp import Vasp
from ase import Atoms
from functools import reduce
from itertools import combinations
from .functions import *
from .SlurmJob import SlurmJob
from .env import *

class ActiveTraining:
    """
    A class to perform an Active Training
    
    """
    
    def __init__(self, 
                 request         = "init",
                 stepNB          = "0",
                 Nadd:int        = 20,
                 Nepoch:int      = 20,
                 minimize        = "energy",
                 path            = ".",
                 nNNP:int        = 2,
                 nnodes:int      = 2,
                 Nstock:int      = 1,
                 logfile         = "status.log",
                 ITERATION:int   = 0,
                 GammaPoint      = "False"):
        self.request = request                            # request type
        self.stepNB = stepNB                              # step number of the active learning, used to name the output files   
        self.path = path                                  # Working directory
        self.nNNP = nNNP                                  # Numbers of NNPs: gotten from the number of 'inputX.nn' files
        self.nnodes = nnodes                              # Nb of nodes used in this job
        self.Nstock = Nstock                              # nb of stock files
        self.logfile = logfile                            # status logfile where output are written
        self.ITERATION = ITERATION                        # ITERATION of the active learning procedure
        self.Nadd = Nadd                                  # Structures to add at each iteration"""
        self.Nepoch = Nepoch                              # nb of the NNPs training epochs
        self.minimize = minimize
        if self.request == "init":
            self.do_initialize()
        if self.request == "CompEfSSDFT":
            self.Ncomb = sum(1 for _ in combinations(range(self.nNNP), 2))
            self.GammaPoint = GammaPoint
            self.Nstruct = self.get_Nstruct(stock = True)     # Number of structures in stock files
            self.compids = self.get_compids()
#            self.stock = self.read_stock()                    # of type ase : Atoms   ===> delete : Dict of {id, comments, structures, compids} of stock structures
#            self.atoms = np.sort(np.unique(self.stock['structures'][0].get_atomic_numbers()))  # Atom types aromic numbers in the system"""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def __repr__(self):
        return(self.__str__())
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def __str__(self):
        if self.request == "init":
            out = f"""├▶︎ Working directory                      = {self.path}
├▶︎ STEP NUMBER                            = {self.stepNB}
├▶︎ Nb of NNPs                             = {self.nNNP}
├▶︎ Nb of Nodes                            = {self.nnodes}
├▶︎ Nb of stock files                      = {self.Nstock}
├▶︎ Nb of added structures per iteration   = {self.Nadd}
├▶︎ Nb of epoch for training               = {self.Nepoch}
├▶︎ quantity to minimize                   = {self.minimize}
├──────────────────────────────────────────────────────────────────────────────────
│
"""
        if self.request == "ScalTrainPred":
            out = f"""├▶︎ Nb of Structures in dataset            = {sum(self.get_Nstruct(stock = False))}
├▶︎ Nb of Structures in stock              = {sum(self.get_Nstruct(stock = True))}
├▶︎ Nb of Structures in stock files        = {self.get_Nstruct(stock = True)}
├──────────────────────────────────────────────────────────────────────────────────
"""
        if self.request == "CompEfSSDFT":
            out = f"""├▶︎ Nb of Structures in stock already computed at the DFT level = {len(self.compids)}
├──────────────────────────────────────────────────────────────────────────────────
"""
        return out

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def do_initialize(self):
        """
        Create arborescence, copy necessary files in folders, and make sure the necessary files are there. If `self.restart=True`, set to restart from previously stopped job, so do not copy files but just check they are here.
        """
        inpdat = self.stepNB + "_input.data"
        stockfile = self.stepNB + "_STOCK.data"
        if self.nNNP < 2 and \
                not Path(f"{self.path}/{inpdat}").is_file() and \
                not Path(f"{self.path}/{stockfile}").is_file() and \
                not Path(f"{self.path}/n2p2.env").is_file() and \
                not Path(f"{self.path}/vasp.env").is_file() and \
                not Path(f"{self.path}/INCAR").is_file() and \
                not Path(f"{self.path}/POTCAR").is_file() and \
                not Path(f"{self.path}/KPOINTS").is_file():
            print(f"""Make sure the following files are in the {self.path} folder:
    - `input1.nn`: input.nn for NNP1
    - `input2.nn`: input.nn for NNP2
    - `stock.data`: input.data file with all stock structures
    - `initial_input.data`: initial input.data structures dataset
    - `INCAR POTCAR KPOINTS`: necessary vasp files
    - `n2p2.env : define path to n2p2 executables and other parameters
    - `vasp.env : define path to vasp executables and other parameters
    - `job.env : define slurm job parameters
    """, flush=True)
            exit()

        # Make directories necesssary fo further evaluations ...
#        Path(f"{self.path}/inputs").mkdir(parents=True, exist_ok=True)
        Path(f"{self.path}/{self.stepNB}_plots").mkdir(parents=True, exist_ok=True)
#        Path(f"{self.path}/MDtests").mkdir(parents=True, exist_ok=True)
#        Path(f"{self.path}/finaltraining").mkdir(parents=True, exist_ok=True)
        # Neural network input of NNP1 used for final training
#        subprocess.run(f"cp {self.path}/input1.nn {self.path}/finaltraining/input.nn", shell=True)
#            subprocess.run(f"cp {mybin}/xtrain {self.path}/finaltraining/", shell=True)
        # create links to data and stock data files at initial ITERATION
        subprocess.run(f"cp {self.path}/{inpdat} {self.path}/input.data", shell=True)

        for i in range(1, self.nNNP+1):
        # For each NNP : 
        # Make directories
            Path(f"{self.path}/{self.stepNB}_NNP{i}/train").mkdir(parents=True, exist_ok=True)
        # copy and modify input.nn 
            subprocess.run(f"cp {self.path}/input{i}.nn {self.path}/{self.stepNB}_NNP{i}/train/input.nn", shell=True)
            subprocess.run(f"sed -i 's/test_fraction /test_fraction 0 #/g' {self.path}/{self.stepNB}_NNP{i}/train/input.nn", shell=True)
            subprocess.run(f"sed -i 's/write_weights_epoch /write_weights_epoch 1 #/g' {self.path}/{self.stepNB}_NNP{i}/train/input.nn", shell=True)
            subprocess.run(f"sed -i 's/write_trainpoints /write_trainpoints 100 #/g' {self.path}/{self.stepNB}_NNP{i}/train/input.nn", shell=True)
            subprocess.run(f"sed -i 's/write_trainforces /write_trainforces 100 #/g' {self.path}/{self.stepNB}_NNP{i}/train/input.nn", shell=True)
            subprocess.run(f"sed -i 's/write_neuronstats /write_neuronstats 100 #/g' {self.path}/{self.stepNB}_NNP{i}/train/input.nn", shell=True)
            for line in fileinput.input(f"{self.path}/{self.stepNB}_NNP{i}/train/input.nn", inplace = 1): 
                print(line.replace("epochs ", f"epochs {self.Nepoch} #"), end='')
            for line in fileinput.input(f"{self.path}/{self.stepNB}_NNP{i}/train/input.nn", inplace = 1): 
                print(line.replace("write_weights_epoch ", f"write_weights_epoch 1 #"), end='')
        # input.data files used in train and predict directories are links
            subprocess.run(f"ln -s {self.path}/input.data {self.path}/{self.stepNB}_NNP{i}/train/input.data", shell=True)
            for j in range(1,self.Nstock+1):
                Path(f"{self.path}/{self.stepNB}_NNP{i}/predict{j}").mkdir(parents=True, exist_ok=True)
                subprocess.run(f"cp {self.path}/input{i}.nn {self.path}/{self.stepNB}_NNP{i}/predict{j}/input.nn", shell=True)
                subprocess.run(f"ln -s {self.path}/stock{j}.data {self.path}/{self.stepNB}_NNP{i}/predict{j}/input.data", shell=True)

        # create and copy VASP necessary directories and files
        for i in range(1, self.nnodes+1):
            Path(f"{self.path}/{self.stepNB}_vasp{i}").mkdir(parents=True, exist_ok=True)
            subprocess.run(f"cp {self.path}/INCAR {self.path}/{self.stepNB}_vasp{i}", shell=True)
            subprocess.run(f"cp {self.path}/POTCAR {self.path}/{self.stepNB}_vasp{i}", shell=True)
            subprocess.run(f"cp {self.path}/KPOINTS {self.path}/{self.stepNB}_vasp{i}", shell=True)

        write_trainingcsvfile(self.nNNP, self.path, self.stepNB, self.minimize)



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



    def get_Nstruct(self, stock=False):
        """
        Get number of structures in input.data. If stock=True : stock.data
        """
        Nstr = []
        if stock == False:
            got=subprocess.run(f"grep -c 'begin' {self.path}/input.data", 
                                shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
            Nstr.append(int(got))
        else:
            for j in range(1,self.Nstock+1):
                got=subprocess.run(f"grep -c 'begin' {self.path}/stock{j}.data", 
                                shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
                Nstr.append(int(got))
        return(Nstr)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def read_stock(self):
        """
        Read initial stock file, and copy structures already computed at DFT level in /vasp
        """
        # # # # # # # # # # # #
        atoms, energies, ids, compids, comments = read_inputdata(self.path, self.Nstock)
        return({'structures': atoms,              # ase atoms objects
                'energies'  : energies,           # floats
                'ids'       : ids,                # list of (stocki, j) pairs : [('stock1', 0), ('stock1', 1), ('stock1', 2), ... ]
                'compids'   : compids,            # list of (stocki, j) pairs : [('stock1', 1245), ... ] already computed with DFT : energy!= 0
                'comments'  : comments})
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def get_compids(self):
        """
        Read initial stock file, and copy structures already computed at DFT level in /vasp
        """
        # # # # # # # # # # # #
        compids = []
        for i in range(self.Nstock):
            null_ids=subprocess.run(f"grep energy stock{i+1}.data|awk 'BEGIN{{count=0}} $2 != 0 {{print count}} ; count++'|sed '/energy/d'", shell=True, capture_output=True, text=True).stdout
            null_ids=null_ids.split("\n")
            null_ids = [item for item in null_ids if item]
            compids += [(f"stock{i+1}", int(value)) for value in null_ids]
        return compids

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def do_ScalTrainPred(self):
        """
        Perform the scaling and training of all NNPs
        """
        self.reset_train()
        self.reset_predict()

        scale = []
        for i in range(1, self.nNNP+1):
            jobname = f"{self.stepNB}_job_STP{i}"
            subprocess.run(f"cp {self.path}/n2p2.env {self.path}/{jobname}", shell=True)
            scale += [SlurmJob(
                    nNNP         = self.nNNP,
                    NNP          = i,
                    nnodes       = self.nnodes,
                    stepNB       = self.stepNB,
                    Nstock       = self.Nstock,
                    ITERATION    = self.ITERATION,
                    logfile      = self.logfile,
                    type         = 'scaltrainpred', 
                    path         = f"{self.path}",
                    jobname      = jobname,
                    nameinjob    = f"STP{i}_{self.ITERATION}")]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def reset_train(self):
        """
        Remove old files to allow checking the new jobs actually ran
        """
        for i in range(1, self.nNNP+1):
            nnppath=f'{self.path}/{self.stepNB}_NNP{i}'
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
        for i in range(1, self.nNNP+1):
            nnppath=f'{self.path}/{self.stepNB}_NNP{i}'
            subprocess.run(f'rm -f {nnppath}/predict/*.out', shell=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def do_compare_energies(self):
        """
        Compare energies computed by all NNPs. from file : NNP{i+1}/predict/trainpoints.000000.out
        Returns [dEmean], [dEstd], [numbers] on all combinations, 
        `numbers`: indexes of structures as found in NNP{i+1}/predict/input.data (i.e. stock.data) 
                   sorted in decreasing order of energy differences per combination of NNPs.
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
        for i in range(self.nNNP):
            colnames=["index",f"Ennp{i+1}"] 
            fE = f"{self.path}/{self.stepNB}_NNP{i+1}/predict1/trainpoints.000000.out"
            finput = f"{self.path}/{self.stepNB}_NNP{i+1}/predict1/input.nn"
            data = pd.read_table(fE, comment='#', sep=r"\s{2,}", 
                                usecols=[0,2], engine="python", 
                                names=colnames, header=None)
            mean_energy, conv_energy = read_input(finput)
            data[f"Ennp{i+1}"] = denorm(data[f"Ennp{i+1}"], mean_energy, conv_energy)
            data.insert(0, "stock", "stock1") # insert une première colonne avec le nom stock qui contient : "stock1"
            for j in range(1,self.Nstock):
                fE = f"{self.path}/{self.stepNB}_NNP{i+1}/predict{j+1}/trainpoints.000000.out"
                finput = f"{self.path}/{self.stepNB}_NNP{i+1}/predict{j+1}/input.nn"
                d = pd.read_table(fE, comment='#', sep=r"\s{2,}", 
                                usecols=[0,2], engine="python", 
                                names=colnames, header=None)
                mean_energy, conv_energy = read_input(finput)
                d[f"Ennp{i+1}"] = denorm(d[f"Ennp{i+1}"], mean_energy, conv_energy)
                d.insert(0, "stock", f"stock{j+1}")
                data = pd.concat([data,d])
                data.reset_index(drop=True)
            if i == 0:
                dataT = data.copy()
            else:
                dataT[f"Ennp{i+1}"] = data[f"Ennp{i+1}"]
        # Compute Energy Differences in meV/atom
        for i,j in combinations(range(1, self.nNNP+1), 2):
            dataT[f'dif{i}-{j}'] = np.abs(dataT[f"Ennp{i}"] - dataT[f"Ennp{j}"])
        dEmean, dEstd, StokIndPairs = [], [], []
        for i,j in combinations(range(1, self.nNNP+1), 2):
            dEmean  += [np.mean(dataT[f'dif{i}-{j}'])]
            dEstd   += [np.std(dataT[f'dif{i}-{j}'])]
            Esorted = dataT.sort_values(by=[f'dif{i}-{j}'], ascending=False)  # new DF with dE sorted by decreasing values 
            Esorted["paired"] = list(zip(Esorted["stock"], Esorted["index"]))
#            StokIndPairs += Esorted["paired"].iloc[:self.Nadd].tolist()        # StokIndPairs list of tupples [(stock,index), ... ] of largest energy differences
            StokIndPairs += Esorted["paired"].tolist()        # StokIndPairs list of tupples [(stock,index), ... ] of largest energy differences
        return(dEmean, dEstd, StokIndPairs, dataT)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def get_last_rmse(self):
        """
        Get RMSE from last epoch from a training `learning-curve.out` file in `self.path`
        """
        rmse = []
        for i in range(1, self.nNNP+1):
            data = pd.read_table(f'{self.path}/{self.stepNB}_NNP{i}/train/learning-curve.out', comment='#', sep=r'\s{2,}', 
                            engine='python', usecols=[0,1,9], names=['epoch','RMSE_E','RMSE_F'], header=None)
            rmse += [(data.RMSE_E.to_list()[-1]*1000*27.21138469, data.RMSE_F.to_list()[-1]*27211.38469/0.529177)]
        return(rmse)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def write2log(self, N:int, RMSE, dEmean, dEstd):
        """
        Write N, RMSE, dEmean, dEstd to `training.csv`
        """
        rmse  = ','.join(f"{r[0]:.6e},{r[1]:.6e}" for r in RMSE)
        de    = ','.join(f"{r:.6e}" for r in dEmean)
        destd = ','.join(f"{r:.6e}" for r in dEstd)
        with open(f"{self.path}/{self.stepNB}_training.csv", 'a') as f:
            f.write(f"{N},{rmse},{de},{destd}\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def plot_RMSE_training(self, i:int=1, save:bool=True, forces:bool=False):

        """
        Plot RMSE of training of NNPs
        Parameters:
        -----------
        i: int, Active learning Iteration 
        save: bool, save the learning curves of the of the NNP's through the training (single Active learning iteration)
        forces: bool, if True beside the RMSE of energies, the RMSE of forces for the NNP's is also recorded.
        Return:
        ------
        fig : matplotlib.figure
        """

        if self.minimize == "force":
            forces = True

        ncols = 2 if forces else 1
        usecols = [0, 1, 9] if forces else [0, 1] 
        names = ['epoch','RMSE_E', 'RMSE_F'] if forces else ['epoch','RMSE_E']
        fig, ax = plt.subplots(ncols=ncols, figsize=(12,6))
        fig.suptitle(f"Training NNPs – ITERATION {i}", fontweight="bold")
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

        for j in range(1, self.nNNP+1):
            data = pd.read_table(f'{self.path}/{self.stepNB}_NNP{j}/train/learning-curve.out', 
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
            plt.savefig(f"{self.path}/{self.stepNB}_plots/NNPtrainings_{i:05d}.jpg", dpi=300)
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def plot_distrib(self, data, i=0, save=True):
        """
        Plot histogram of E_{NNP2} - E_{NNP1} or F_{NNP2} - F_{NNP1} to `plots/histogram_{i:05d}.jpg`
        """
        Tot = self.Ncomb
        Cols = 1
        Rows = Tot // Cols 
        if Tot % Cols != 0:
            Rows += 1
        # # # # # 
        typ = 'E' if self.minimize == "energy" else 'F'
        
        fig,axes = plt.subplots(Rows,Cols, tight_layout=True, sharex=True, sharey=True, figsize=(8, Tot*2.5))
        fig.suptitle(f'Iteration {i}')
        l = 0
        if Tot == 1:
            axes = [axes]
        for (k,j),ax in zip(combinations(range(1, self.nNNP+1), 2), axes):
            if self.minimize == "energy":
                ax.set_xlabel(f'${typ}_{{NNP_i}} - {typ}_{{NNP_j}}$ [meV/atom]')
                x = data[f"Ennp{k}"] - data[f"Ennp{j}"]
            else:
                ax.set_xlabel(f'${typ}_{{NNP_i}} - {typ}_{{NNP_j}}$ [meV/Å]')
                x = data[f"dif{k}-{j}"]
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
#            data.iloc[:, :(self.nNNP+1)].to_csv(f"{self.path}/plots/histogram_{i:05d}.csv", index=False)
            plt.savefig(f"{self.path}/{self.stepNB}_plots/histogram_{i:05d}.jpg", dpi=300)
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    
    
    def plot_convergence(self, save=True):
        """
        Plot convergence of adaptative training
        """
        typ = 'E' if self.minimize == "energy" else 'F'
        unit = 'meV/at' if self.minimize == "energy" else 'meV/Å'
        data = pd.read_csv(f'{self.path}/{self.stepNB}_training.csv', comment='#')
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, tight_layout=True, figsize=(9, 4),
                                        gridspec_kw={'width_ratios':[3,3,1]})
        fig.suptitle(f"Convergence plot (step ${self.stepNB})", fontweight="bold")
        ax1.set_title(f"$|\Delta {typ}|$", fontweight="bold")
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel(f'$|\Delta {typ}|$ [{unit}]')
        ax1.grid()
        ax1.set_ylim([0, max(data[f'd{typ}mean1-2'])])
        for i,j in combinations(range(1, self.nNNP+1), 2):
            ax1.plot(data.N, data[f'd{typ}mean{i}-{j}'], 
                        label=f'$NNP{i}-NNP{j}$')

        ax2.set_title(f"$std(\Delta {typ})$", fontweight="bold")
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel(f'$std(\Delta {typ})$ [{unit}]')
        ax2.grid()
        ax2.set_ylim([0, 5*min(data[f'd{typ}std1-2'])])
        for i,j in combinations(range(1, self.nNNP+1), 2):
            ax2.plot(data.N, data[f'd{typ}std{i}-{j}'], 
                        label=f'$NNP{i}-NNP{j}$')

        Line, Label = ax1.get_legend_handles_labels()
        ax3.legend(Line, Label)
        ax3.axis('off')
        if save:
            plt.savefig(f"{self.path}/{self.stepNB}_convergence.jpg", dpi=300, bbox_inches='tight')
            plt.cla()
            plt.close(fig)
        return(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def do_vasp(self, StokIndPairs):
        """
        make VASP jobs for DFT calculations on structures with indexes in list `indexes`. 
        Distribute the jobs on `self.nNNP` nodes.
        """        
        Njobs = self.nnodes
        NStokIndPairs = len(StokIndPairs)
        for i in range(Njobs):
            jobname = f"{self.stepNB}_job_vasp{i+1}"
            subprocess.run(f"cp {self.path}/vasp.env {self.path}/{jobname}", shell=True)
#            if NStokIndPairs == 0:
#                with open(f"{self.path}/{jobname}", "a") as f:
#                    f.write(f"echo \"│  [$(date +%T)]   ├─▶︎ No Vasp job {i+1} for ITERATION {self.ITERATION}\" >> {self.path}/{jobname}\n")
#            else:
            vasp = []
            SelectedPairs = StokIndPairs[(i*NStokIndPairs//Njobs):((i+1)*NStokIndPairs//Njobs)]
            for pair in SelectedPairs:
                subprocess.run(f"{mybin}/SelStructFromDat.sh {self.path}/{pair[0]}.data {pair[1]}", shell=True)
                subprocess.run(f"python {mybin}/xconvert -in {self.path}/SelStruct.data -out vasp > {self.path}/{self.stepNB}_vasp{i+1}/POSCAR_{self.ITERATION}_{pair[0]}-{pair[1]} 2> convert.out", shell=True)
                subprocess.run(f"rm -f {self.path}/SelStruct.data", shell=True)
            nameinjob = f"vasp{i+1}"
            vasp += [SlurmJob(nNNP         = self.nNNP,
                NNP           = i+1,
                nnodes       = self.nnodes,
                stepNB        = self.stepNB,
                Nstock        = self.Nstock,
                ITERATION     = self.ITERATION,
                logfile       = self.logfile,
                type          = "vasp",
                path          = f"{self.path}", 
                jobname       = jobname,
                nameinjob     = nameinjob,
                GammaPoint    = self.GammaPoint,
                SelectedPairs = SelectedPairs)]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



