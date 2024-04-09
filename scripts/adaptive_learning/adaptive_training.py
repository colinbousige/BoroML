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
AT = AdaptiveTraining(Nadd        = 20, 
                      restart     = False, 
                    #   restart     = True, 
                      Nepoch      = 7,
                      clusterNNP  = lynxnnp,
                      clusterVASP = lynxvasp,
                      vasp_Nnodes = 5,
                      atoms       = "BAg")
# /!\ Set to restart=True if restarting after stopping job, 
# /!\ otherwise it will restart from scratch!

for i in range(1, AT.Niter+1):
    Nstruct = AT.get_Nstruct(copy=True)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    print(f"""├─────▶︎ ITERATION {i} / {AT.Niter} ◀︎──────
│
├─▶︎ {Nstruct} structures in input.data
├─▶︎ {AT.get_Nstruct(stock=True)} structures in stock
│""", flush=True)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    print(f"├───▶︎ Training {AT.Nnnp} NNPs\n│", flush=True)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Perform the, scaling, training and prediction of both NNPs and wait until it's done
    # AT.do_NNPs() # sometimes fails for whatever reason...
    AT.do_scaling()
    AT.do_training()
    AT.do_predict()
    # Get RMSEs
    RMSE = AT.get_last_rmse()
    # Compare energies
    dEmean, dEstd, numbers, energies = AT.compare_energies()
    # Output to log
    AT.write2log(Nstruct, RMSE, dEmean, dEstd)
    # Make plots
    AT.plot_RMSE_training(i)
    AT.plot_distrib(energies, i)
    if i > 1:
        AT.plot_convergence()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    print(f"│\n├───▶︎ Making VASP calculations on the {AT.Nadd} structures with largest E difference\n│", flush=True)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Compute the AT.Nadd first structures for each combination of NNP
    nums = np.unique(np.concatenate([nu[:AT.Nadd] for nu in numbers]).ravel())
    indexes = AT.get_POSCARs(nums)
    AT.do_vasp(indexes)
    # Structures up to numbers[:to_add] have been added, remove them from stock:
    to_add = AT.Nadd
    struct_done = np.unique(np.concatenate([nu[:to_add] for nu in numbers]).ravel())
    AT.update_stock(struct_done)

# It's done!
print_success_message()


