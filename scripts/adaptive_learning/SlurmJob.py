import time
import re
import subprocess
import os
from . import Cluster
from .environment import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class SlurmJob:
    """
    A class to run and manage `n2p2` and `VASP` jobs with Slurm.
    
    ## Attributes
    `type`: str
        Type of job, one of 'vasp', 'nnp-train', 'nnp-scaling' or 'nnp-all'. 
        'nnp-all' will successively scale, train, copy weigths to predict/, and predict on stock structures
    `cluster`: Cluster object
    `path`: str. Default = `'.'`
        Path of the job to write and run.
    `jobname`: str, optional. Default = `'job'`
        Name of the jobfile.
    `numbers`: int list, optional. Default = `[]`
        For 'vasp'-type jobs, indexes of the POSCAR_xx files to compute.
    `launch`: bool, optional. Default = `False`
        Launch the job upon creation or not?
    `maxtries`: int, optional. Default = 20
        Maximum number of submission tries for jobs before giving up
    
    ## Methods
    `write_job()`
        Write job file to `self.path`
    `submit()`
        Launch job and retrieve job number
    `is_running()`
        Test if job is still running or not.
    `assign_node()`
        Assign to available node. If no available node, queue to `preferred_node` on half the total CPUs.
    
    ## Usage
    >>> from adaptive_learning import *
    >>> import os
    >>> lynx = Cluster()
    >>> AT = AdaptiveTraining(Nadd=20, restart=False)
    >>> # Without launching job directly
    >>> scaling1 = SlurmJob(
    ...     type    = 'nnp-scaling', 
    ...     path    = f"{PAT}/NNP1/train", 
    ...     cluster = lynx,
    ...     jobname = f"scale1")
    >>> scaling1.write_job()
    >>> scaling1.submit()
    Launching scale1 with ID 4301 on node33 with 4 cores.
    >>> myjob.is_running()
    True
    >>> # Launching the job upon creation
    >>> scaling2 = SlurmJob(
    ...     type    = 'nnp-scaling', 
    ...     path    = f"{PAT}/NNP2/train", 
    ...     jobname = f"scale2", 
    ...     launch  = True)
    Launching scale2 with ID 4300 on node33 with 4 cores.
    >>> scaling2.is_running()
    True
    """
    
    def __init__(self, type: str, cluster: Cluster, path='.', 
                 jobname='job', numbers=[], launch=False, maxtries=20, wait=3):
        if not type in ['vasp', 'nnp-train', 'nnp-scaling', 'nnp-all']:
            raise TypeError("type must be one of 'vasp', 'nnp-train', 'nnp-scaling', or 'nnp-all'.")
        self.node      = ''
        """Node name for this given job"""
        self.queue     = ''
        """Queue name for this given job"""
        self.nproc     = 0
        """Number of processors to run this given job on"""
        self.jobname   = jobname
        """Name of the jobfile."""
        self.path      = path
        """Path of the job to write and run."""
        self.type      = type
        """Type of job, one of 'vasp', 'nnp-train', 'nnp-scaling' or 'nnp-all'. 
        'nnp-all' will successively scale, train, copy weigths to `predict/`, and predict on stock structures"""
        self.numbers   = numbers
        """For 'vasp'-type jobs, indexes of the POSCAR_xx files to compute."""
        self.jobnumber = ''
        """Once submitted, ID of the job"""
        self.running   = False
        """Is the job running or not?"""
        self.maxtries  = maxtries
        """Maximum number of submission tries for jobs before giving up"""
        self.cluster   = cluster
        """Cluster object"""
        self.wait      = wait
        """Time to wait between submission tries"""
        if launch:
            """Launch the job upon creation or not?"""
            self.assign_node()
            self.write_job()
            self.submit()

    
    def __repr__(self):
        return(self.__str__())
    
    def __str__(self):
        out = f"""Slurm Job Information:
jobname   = {self.jobname}
type      = {self.type}
path      = {self.path}
node      = {self.node}
nproc     = {self.nproc}
queue     = {self.queue}
numbers   = {self.numbers}
jobnumber = {self.jobnumber}
running   = {self.is_running()}"""
        return(out)
    
    def write_header(self):
        """Write job header of job script"""
        if self.type == "vasp":
            add = f"""VASPDIR={vasp_dir}
WDIR={clusterwdir}-{self.jobname}
source $VASPDIR/env.sh

mkdir -p $WDIR || exit 1
cd $WDIR
cp {self.path}/INCAR .
cp {self.path}/POTCAR .
cp {self.path}/KPOINTS .
"""
        if "nnp" in self.type:
            add = f"""source {openmpi_env}
source {gsl_env}
export N2P2DIR={n2p2_dir}
export LD_LIBRARY_PATH=$N2P2DIR/lib:$LD_LIBRARY_PATH
export PATH={mybin}:$N2P2DIR/bin:$PATH
"""
        return(f"""#!/bin/bash
#SBATCH -J {self.jobname}
#SBATCH -A theochem
#SBATCH -p {self.queue}
#SBATCH -N 1
#SBATCH -n {self.nproc}
#SBATCH -w {self.node}
#SBATCH -t UNLIMITED
#SBATCH -o {self.path}/{self.jobname}.o
#SBATCH -e {self.path}/{self.jobname}.e

export OMP_NUM_THREADS=1
MPIBIN=mpirun
PAT={self.path}
{add}
""")
    
    def write_command(self):
        """Write job command section of job script"""
        if self.type == "vasp":
            out = ''
            for i in self.numbers:
                out += f"""
cp $PAT/POSCAR_{i} $WDIR/POSCAR
$MPIBIN -np {int(self.nproc//2)} $VASPDIR/vasp_std
mv $WDIR/OUTCAR $PAT/OUTCAR_{i}
python {mybin}/convert-VASP_OUTCAR_AU.py $PAT/OUTCAR_{i} $PAT/OUTCAR_{i}.inp

"""
        if self.type == "nnp-train":
            out = f"""
cd $PAT

$MPIBIN -np {int(self.nproc//2)} --bind-to core $N2P2DIR/bin/nnp-train
"""
        if self.type == "nnp-scaling":
            out = f"""
cd $PAT

$MPIBIN -np {self.nproc} $N2P2DIR/bin/nnp-scaling 5
rm -f sf*.histo
"""
        return(out)
    
    def write_job(self):
        """Write job script to `path/jobname` and make executable"""
        out = self.write_header() + self.write_command()
        with open(f"{self.path}/{self.jobname}", "w") as f:
            f.write(out)
        subprocess.run(f"chmod u+x {self.path}/{self.jobname}", shell=True)
        time.sleep(2)
    
    def submit(self):
        """Submit job to slurm's queue and retrieve job number ID"""
        jobnumber = []
        Ntries = 0
        midcount=0
        if self.node == '':
            self.assign_node()
        if not os.path.exists(f"{self.path}/{self.jobname}"):
            self.write_job()
        while Ntries < self.maxtries:
            Ntries += 1
            msg = subprocess.run(f"sbatch {self.path}/{self.jobname}", shell=True, 
                    capture_output=True).stdout.decode('utf-8').split('\n')[0]
            jobnumber = re.findall(r'\d+', msg)
            if len(jobnumber)==0: # if job submission failed for some reason, try again
                time.sleep(2)
                self.assign_node()
                self.write_job()
                time.sleep(2)
                continue
            self.jobnumber = jobnumber[0]
            print(f"├─ Launched {self.jobname} with ID {self.jobnumber} on {self.node} with {self.nproc} cores...", flush=True, end="")
            time.sleep(self.wait)
            msg = subprocess.run(f"squeue -u $USER", shell=True, 
                capture_output=True).stdout.decode('utf-8')
            if self.jobnumber in msg:
                self.running = True
                print(f"  ──▶︎ OK.", flush=True)
                break
            else:
                midcount += 1
                print(f"  ──▶︎ Failed.", flush=True)
                if midcount>6:
                    print(f"│   └──▶︎ Again? OK, {self.node}, I'm adding your ass to my shit list.", flush=True)
                    self.cluster.update_forbid_nodes([self.node])
                    midcount=0
                print(f"│   └──▶︎ Retrying {Ntries}/{self.maxtries}...", flush=True)
                self.assign_node()
                self.write_job()
        else:
            raise TypeError(f"Could not submit job {self.path}/{self.jobname} after {Ntries} tries.")
    
    def is_running(self):
        """Check whether the job is still in slurm's queue or not"""
        msg = subprocess.run(f"squeue -u $USER", shell=True, 
                capture_output=True).stdout.decode('utf-8')
        if len(self.jobnumber)>0:
            self.running = self.jobnumber in msg
            return(self.running)
        else:
            return(False)
    
    def assign_node(self):
        """
        Get state of lynx and find available nodes. 
        If no node available, queue the job to `self.preferred_node`.
        """
        msg = subprocess.run(f"squeue -u $USER", shell=True, 
                capture_output=True).stdout.decode('utf-8')
        if ' PD ' in msg:
            time.sleep(5) # wait for previouly submitted job to maybe be able to run
        freenodes = self.cluster.get_free_nodes()
        cluster_state = self.cluster.get_state()
        if len(freenodes)>0:
            self.node  = freenodes.NodeName.tolist()[0]
            self.queue = freenodes.Partitions.tolist()[0]
            self.nproc = freenodes.CPUFree.tolist()[0]
        else:
            self.node  = self.cluster.preferred_node
            self.nproc = cluster_state[cluster_state.NodeName.isin([self.node])].CPUTot.tolist()[0]
            self.queue = cluster_state[cluster_state.NodeName.isin([self.node])].Partitions.tolist()[0]
        if self.type == 'nnp-scaling':
            self.nproc = self.nproc//2 #4

