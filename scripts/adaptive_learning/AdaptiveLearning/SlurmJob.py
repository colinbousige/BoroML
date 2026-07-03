import time
import re
import subprocess
from .env import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class SlurmJob:
    """
    A class to run and manage `n2p2` and `VASP` jobs with Slurm.
    """
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def __init__(self, 
                 nNNP: int, 
                 NNP: int,
                 nnodes: int,
                 stepNB: int,
                 Nstock: int, 
                 ITERATION: int, 
                 logfile:str, 
                 type: str, 
                 path='.',
                 jobname='job', 
                 nameinjob='job', 
                 GammaPoint='False', 
                 SelectedPairs=[]):        
        if not type in ['vasp', 'scaltrainpred']:
            raise TypeError("type must be one of 'vasp' or 'scaltrainpred'.")        
        self.nNNP          = nNNP                # nb of NNPs
        self.NNP           = NNP                 # NNP Id
        self.nnodes        = nnodes              # nb of nodes
        self.stepNB        = stepNB              # step number of the adaptive learning
        self.Nstock        = Nstock              # nb of stock files
        self.ITERATION     = ITERATION           # Adaptive Learning ITERATION
        self.logfile       = logfile             # status file of the adaptive learning calculations
        self.type          = type                # Type of job, one of 'vasp', 'scaltrainpred'
        self.path          = path                # Path of the adaptive learning job
        self.jobname       = jobname             # Name of the jobfile
        self.nameinjob     = nameinjob           # Name of the job in Slurm job file
        self.GammaPoint    = GammaPoint          # to select the appropriate vasp executable
        self.SelectedPairs = SelectedPairs       # For 'vasp'-type jobs, [('stockx', index), ...] pairs of thestructures to be computed.
        self.write_job()
   
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def __repr__(self):
        return(self.__str__())
    
    def __str__(self):
        out = f"""Slurm Job Information:
Nb of NNPs             = {self.nNNP}
NNP Id                 = {self.NNP}
Nb of nodes            = {self.nnodes}
stepNB                 = {self.stepNB}
Nb of stock files      = {self.Nstock}
Adp.Learn.ITERATION    = {self.ITERATION}
logfile                = {self.logfile}
type                   = {self.type}
path                   = {self.path}
jobname                = {self.jobname}
name in job            = {self.nameinjob}
Gamma Point            = {self.GammaPoint}
SelectedPairs          = {self.SelectedPairs}
"""
        return(out)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def write_job(self):
        """Write job script to `path/jobname` and make executable"""
  # # #     VASP job     # # # 
        if self.type == "vasp":
            out = f"cd {self.path}/{self.stepNB}_vasp{self.NNP}\n"
            #out = ''
            vaspexec = 'vasp_std' if self.GammaPoint == 'False' else 'vasp_gam'
            for pair in self.SelectedPairs:
                out += f"""
cp POSCAR_{self.ITERATION}_{pair[0]}-{pair[1]} POSCAR
srun --nodes=1 {vaspexec} >> VASP_{self.ITERATION}.out
mv OUTCAR OUTCAR_{self.ITERATION}_{pair[0]}-{pair[1]}
rm -f PCDAT CHG XDATCAR out VASP.* ICONST IBZKPT REPORT EIGENVAL DOSCAR vaspout.h5 CHGCAR WAVECAR OSZICAR CONTCAR vasprun.xml
#python {mybin}/convert-VASP_OUTCAR_AU.py OUTCAR_{self.ITERATION}_{pair[0]}-{pair[1]} OUTCAR_{self.ITERATION}_{pair[0]}-{pair[1]}.inp 2>> VASP_{self.ITERATION}.out
python {mybin}/xconvert -in OUTCAR_{self.ITERATION}_{pair[0]}-{pair[1]} -out n2p2 > OUTCAR_{self.ITERATION}_{pair[0]}-{pair[1]}.inp 2>> VASP_{self.ITERATION}.out
"""
            out += f"""
echo "│  [$(date +%T)]   ├─▶︎ Vasp job {self.NNP} finished" >> {self.path}/{self.logfile}

"""
  # # #  scaltrainpred job  # # # 
        nnode = int(self.nnodes) / int(self.nNNP)
        nnode = int(nnode)
        if self.type == "scaltrainpred":
            out = f"""
cd {self.path}/{self.stepNB}_NNP{self.NNP}/train
rm -f evsv.dat function.data neighbors.histo learning-curve.out neuron-stats* nnp-* weights* output.data scaling.data t* updater*

srun --nodes={nnode} --nodelist=$NODE{self.NNP} nnp-scaling 5 > scal.out
rm -f sf*.histo
echo "│  [$(date +%T)]   ├─▶︎ Scaling finished for NNP{self.NNP}" >> {self.path}/{self.logfile}
srun --nodes={nnode} --nodelist=$NODE{self.NNP} nnp-train > train.out
#nnp-norm       #needed for n2p2 version < 2.2
echo "│  [$(date +%T)]   ├─▶︎ Training finished for NNP{self.NNP}" >> {self.path}/{self.logfile}

# get best epoch weigths, copy files and clean
{mybin}/copy_weights.sh
mv input.nn.bak input.nn
"""
            for i in range(1,self.Nstock+1):
                out += f"""
cd {self.path}/{self.stepNB}_NNP{self.NNP}/predict{i}
# clean
rm -f learning-curve.out neuron-stats* nnp-* t* updater*
srun --nodes={nnode} --nodelist=$NODE{self.NNP} nnp-train > train.out
"""
            out += f"""
echo "│  [$(date +%T)]   ├─▶︎ Predict finished for NNP{self.NNP}" >> {self.path}/{self.logfile}
"""
        with open(f"{self.path}/{self.jobname}", "a") as f:
            f.write(out)
        subprocess.run(f"chmod u+x {self.path}/{self.jobname}", shell=True)
        subprocess.run(f"sed -i 's/NAME/{self.nameinjob}/' {self.path}/{self.jobname}", shell=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

