import numpy as np
import pandas as pd
import time
import subprocess
from tabulate import tabulate

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def pprint_df(dframe):
    return(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))


class Cluster:
    """
    A class to access cluster state
    
    ## Parameters
    `preferred_node`: str. Default = 'l-node04'
        Preferred node in case no available node is found.
    `forbid_nodes`: str list, optional. Default = `['l-node01', 'l-node02', 'l-node03']`
        List of nodes the user has not access to.
    `forbid_queues`: str list, optional. Default = `['tc1']`
        List of queues to forbid launching to.
    
    ## Methods
    `get_state(self)`
        Get state of cluster as a DataFrame
    `get_free_nodes(self)`
        Get state of cluster and find available nodes as a DataFrame.
    `update_forbid_nodes(self, nodes)`
        Add nodes to the list of forbidden nodes.
    
    ## Usage
    >>> from adaptive_learning import *
    >>> lynx = Cluster()
    >>> lynx.get_free_nodes()
    NodeName  CPUTot  CPUAlloc  CPUFree Partitions  State
    node33      96         0       96        tc3     IDLE
    node32      40         0       40        tc2     IDLE
    node38     128       102       26        tc5    MIXED
    """
    
    def __init__(self, 
                 preferred_node = 'l-node04',
                 forbid_nodes = ['l-node01', 'l-node02', 'l-node03'],
                 forbid_queues = ['tc1']
                 ):
        self.forbid_nodes   = forbid_nodes
        """List of nodes the user has not access to."""
        self.forbid_queues  = forbid_queues
        """List of queues to forbid launching to."""
        self.preferred_node = preferred_node
        """Preferred node in case no available node is found."""
        self.state          = self.get_state()
        """Cluster state (result of self.get_state())"""
        self.free           = self.get_free_nodes()
        """Cluster free nodes (result of self.get_free_nodes())"""


    def __repr__(self):
        return(self.__str__())
    
    def __str__(self):
        out = f"""
CLUSTER INFORMATION ───────────────────────────────────────────────────

Preferred node   ├ {", ".join(self.preferred_node)}
Forbidden nodes  ├ {", ".join(self.forbid_nodes)}
Forbidden queues ├ {", ".join(self.forbid_queues)}

AVAILABLE NODES ───────────────────────────────────────────────────────

{pprint_df(self.free)}

CLUSTER STATE ─────────────────────────────────────────────────────────

{pprint_df(self.state)}
"""
        return(out)
    
    def get_state(self):
        """
        Get state of cluster as a DataFrame
        """
        msg = subprocess.run(f"scontrol show node", shell=True, 
                capture_output=True).stdout.decode('utf-8').split('\n')
        NodeName   = np.array([])
        CPUAlloc   = np.array([])
        CPUTot     = np.array([])
        Partitions = np.array([])
        State      = np.array([]) # want idle or mixed
        for line in msg:
            if 'NodeName' in line:
                NodeName = np.append(NodeName, line.split('NodeName=')[1].split(' ')[0])
            if 'CPUAlloc' in line:
                CPUAlloc = np.append(CPUAlloc, line.split('CPUAlloc=')[1].split(' ')[0])
            if 'CPUTot' in line:
                CPUTot = np.append(CPUTot, line.split('CPUTot=')[1].split(' ')[0])
            if 'Partitions' in line:
                Partitions = np.append(Partitions, line.split('Partitions=')[1].split(' ')[0])
            if 'State' in line:
                State = np.append(State, line.split('State=')[1].split(' ')[0])
        CPUTot = CPUTot.astype(int)
        CPUAlloc = CPUAlloc.astype(int)
        CPUFree = CPUTot - CPUAlloc
        return(pd.DataFrame({"NodeName"  : NodeName,  "CPUTot"  : CPUTot,
                             "CPUAlloc"  : CPUAlloc,  "CPUFree" : CPUFree,
                             "Partitions": Partitions,"State"   : State}))
    
    def get_free_nodes(self):
        """
        Get state of cluster and find available nodes. 
        """
        msg = subprocess.run(f"squeue -u $USER", shell=True, 
                capture_output=True).stdout.decode('utf-8')
        if ' PD ' in msg:
            time.sleep(5) # wait for previouly submitted job to maybe be able to run
        self.state = state = self.get_state()
        oknodes = ['MIXED', 'IDLE']
        state = state[~state.NodeName.isin(self.forbid_nodes)] # remove forbidden nodes
        state = state[~state.Partitions.isin(self.forbid_queues)] # remove forbidden queues
        statefree = state[state.State.isin(oknodes)].sort_values( # sort by free CPUs
                            by = ['CPUFree'], ascending = False)
        return(statefree)
    
    def update_forbid_nodes(self, nodes):
        """
        Add nodes to the list of forbidden nodes. `nodes` MUST be a list.
        """
        self.forbid_nodes += nodes

