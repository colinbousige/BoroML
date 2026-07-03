#!/usr/local/bin/python3

from adaptive_learning import *

# Create a Cluster object:
lynxvasp = Cluster(preferred_node = 'l-node04',
                   forbid_nodes   = ['node31', 'l-gnode01'],
                   forbid_queues  = ['tc1', 'tc5', 'cl24'])
lynxnnp  = Cluster(preferred_node = 'l-node04',
                   forbid_nodes   = ['node31', 'l-gnode01'],
                   forbid_queues  = ['tc1', 'tc5', 'cl24'])

# Initialize an AdaptiveTraining object:
AT = AdaptiveTraining(Nadd        = 50, 
                      Nepoch      = 20,
                      restart     = False,
                      clusterNNP  = lynxnnp,
                      clusterVASP = lynxvasp,
                      vasp_Nnodes = 4,
                      Kpoints     = [7,7,1],
                      ENCUT       = 700,
                      minimize    = 'F')
# /!\ Set to restart=True if restarting after stopping job, 
# /!\ otherwise it will restart from scratch!

# Run the AdaptiveTraining object:
AT.run()
