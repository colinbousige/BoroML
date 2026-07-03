#!/home/pmignon/bin/miniconda3/bin/python

from ActiveLearning import * # type: ignore
import os
import glob
import time
import subprocess
###############################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # #        DEFINE ARGUMENTS         # # # # # # # # #
###############################################################################################################
args = getArguments()
Nadd = args.Nadd                           # nb of structures to added to the dataset every iteration f the adptive
Nepoch = args.Nepoch                       # nb of training epochs for the NNPs
request = args.request                     # action requested to the active
stepNB = args.stepNB                       # step number of the active learning, used to name the output files
nnodes = args.nnodes                       # nb of nodes used for this job 
minimize = args.minimize                   # which quantity to be minimized between the NNPs 
minimize = check_minimize(minimize)     
GammaPoint = args.GammaPoint
ds_increment = args.ds_increment
path = os.getcwd()                         # directory where every files and diretories will be created
nNNP = len(glob.glob1(path,"input?.nn"))   # nb of NNPS to check if potenetial surface is well sampled
Nstock = len(glob.glob1(path,"stock[1-9]*.data"))
logfile = stepNB + "_status.log"
p_logfile = path+"/"+logfile

###############################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # #      INITIALIZE OPTION , CREATE DIRECTORIES      # # # # # # # # #
    # # # # # # #        & LAUNCH INITIAL SCAL-TRAIN-PREDICT        # # # # # # # # #
###############################################################################################################
if request == 'init':
    ITERATION = 0
    AT = ActiveTraining(request     = request,
                          stepNB      = stepNB,
                          Nadd        = Nadd,
                          Nepoch      = Nepoch,
                          minimize    = minimize,
                          path        = path,
                          nNNP        = nNNP,
                          nnodes      = nnodes,
                          Nstock      = Nstock,
                          logfile     = logfile,
                          ITERATION   = ITERATION,
                          GammaPoint  = GammaPoint)

    out = f"""├──────────────────────────────────────────────────────────────────────────────────
│        Starting Active Learning Calculation on {time.strftime('%Y-%m-%d', time.gmtime())}            [{time.strftime('%H:%M:%S', time.gmtime())}]
├──────────────────────────────────────────────────────────────────────────────────
├──────────────────────▶︎        ADAPTIVE   STEP   NB    {stepNB}     
├──────────────────────────────────────────────────────────────────────────────────
├──────────────────────▶︎          ITERATION    {ITERATION}
├───────▶︎       INITIAL STEP : Create directories and copy necessary files
├──────────────────────────────────────────────────────────────────────────────────
"""
    out += AT.__str__()
    with open(f"{path}/{logfile}", "w") as f:
        f.write(out)
    print(out)

else:

###############################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #       CHECK logfile EXISTS, READ INFORMATIONS        # # # # # # # # #
# # # # # # #                   Create AT object                   # # # # # # # # #
###############################################################################################################
    if not Path(f"{logfile}").is_file():
        print(f""" !!!!    Make sure the following file exits     !!!!
    {logfile}
    !!!!   <---------------------------------->    !!!!
        """, flush=True)
        exit()
    Nadd, Nepoch, minimize, ITERATION, StokIndPairs = readlogfile(logfile = logfile)  # indexes necessary for the "updates" option
    AT = ActiveTraining(request     = request,
                          stepNB     = stepNB,
                          Nadd        = Nadd,
                          Nepoch      = Nepoch,
                          minimize    = minimize,
                          path        = path,
                          nNNP        = nNNP,
                          nnodes      = nnodes,
                          Nstock      = Nstock,
                          logfile     = logfile,
                          ITERATION   = ITERATION,
                          GammaPoint  = GammaPoint)

###############################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #       PERFORM A SCALING - TRAINING - PREDICT          # # # # # # # # #
###############################################################################################################
if request == 'ScalTrainPred':
    AT.ITERATION += 1
    out = f"""│  [{time.strftime('%H:%M:%S', time.gmtime())}]
├──────────────────────▶︎          ITERATION    {AT.ITERATION}
├──────────────────────────────────────────────────────────────────────────────────
"""
    out += AT.__str__()
    out += f"""├──────────────────────────────────────────────────────────────────────────────────
│
├───────▶︎       Perform  Scaling - Training - Predict  from n2p2        [{time.strftime('%H:%M:%S', time.gmtime())}]
├──────────────────────────────────────────────────────────────────────────────────
"""
    with open(f"{path}/{logfile}", "a") as f:
        f.write(out)
    AT.do_ScalTrainPred()

###############################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #     COMPARE ENERGIES OPTION, SELECT NEW STRUCTURES    # # # # # # # # #
# # # # # # #             & PREPARE DFT CALCULATIONS                # # # # # # # # #
###############################################################################################################
if request == 'CompEfSSDFT':
    out = f"""├───────▶︎       Compare NNPs E/forcs - get structrs from stock - perform DFT calcs          [{time.strftime('%H:%M:%S', time.gmtime())}]
├──────────────────────────────────────────────────────────────────────────────────
"""
    out += AT.__str__()
    with open(f"{path}/{logfile}", "a") as f:
        f.write(out)
# Get RMSEs
    with open(f"{path}/{logfile}", "a") as f:
        f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Compute RMSEs\n")
    RMSE = AT.get_last_rmse()
# Compare energies/forces
    if minimize == "energy":
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Compare energies\n")
        dmean, dstd, StokIndPairs, quantities = AT.do_compare_energies()
        indexi = ds_increment * Nadd
        indexj = indexi + Nadd
        StokIndPairs = StokIndPairs[indexi:indexj]
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Selected structures indices : Stock[{indexi}:{indexj}].\n")

    if minimize == "forces":
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Compare forces\n")
        dmean, dstd, StokIndPairs, quantities = AT.do_compare_forces()
# Make plots
    with open(f"{path}/{logfile}", "a") as f:
        f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Make plots\n")
    AT.plot_RMSE_training(AT.ITERATION, forces=True)
    AT.plot_distrib(quantities, AT.ITERATION)
# write training.csv
    with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Write training.csv\n")
    AT.write2log(sum(AT.get_Nstruct()), RMSE, dmean, dstd)
    if AT.ITERATION > 1:
        AT.plot_convergence()
# select the self.Nadd structures with largest E/F differences computd from NNPis from stock.data
    vasp_StokIndPairs = [item for item in StokIndPairs if item not in AT.compids]
    comp_StokIndPairs = [item for item in StokIndPairs if item not in vasp_StokIndPairs]
# print informations on logfile
    with open(f"{path}/{logfile}", "a") as f:
        f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Indices of the {len(StokIndPairs)} selected structures :{StokIndPairs}.\n")
        if len(vasp_StokIndPairs) != 0:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Indices of the {len(vasp_StokIndPairs)} structures to be computed with DFT :{vasp_StokIndPairs}.\n")
        if len(comp_StokIndPairs) != 0:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Indices of the {len(comp_StokIndPairs)} structures with DFT E/forces already in stock :{comp_StokIndPairs}.\n")
# Perform VASP jobs to compute the selected structures (not yet DFT computed) 
    if len(vasp_StokIndPairs) == 0:
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ No Vasp job !\n")
    else:
        AT.do_vasp(vasp_StokIndPairs)
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ {len(vasp_StokIndPairs)} POSCARS created being computed with VASP\n")
    if len(comp_StokIndPairs) != 0:
        subprocess.run(f"touch {AT.path}/{stepNB}_vasp1/OUTCAR_{AT.ITERATION}.inp", shell=True)
        for pair in comp_StokIndPairs:
            subprocess.run(f"{mybin}/SelStructFromDat.sh {pair[0]}.data {pair[1]}", shell=True)
            subprocess.run(f"cat {AT.path}/SelStruct.data >> {AT.path}/{stepNB}_vasp1/OUTCAR_{AT.ITERATION}.inp", shell=True)
            subprocess.run(f"rm -f {AT.path}/SelStruct.data", shell=True)
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ {len(comp_StokIndPairs)} structures from stockx.data files copied to {stepNB}_vasp1/OUTCAR_{AT.ITERATION}.inp\n")

###############################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #           UPDATE STOCK AND DATASET FILES            # # # # # # # # #
###############################################################################################################
if request == 'updates':
#
# # # Add a warning if OUTCARS.inp sont vides (cela veut dire que forces > 5 eV/Angstrom)
#
    N_OUTCARS = subprocess.run(f"ls -l {stepNB}_vasp?/OUTCAR_{AT.ITERATION}*inp|wc -l", 
                                shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
    N_OUTCARS = int(N_OUTCARS)
    N_empty_OUT = subprocess.run(f"find {stepNB}_vasp? -maxdepth 1 -name \"OUTCAR_{AT.ITERATION}*inp\" -type f -size 0|wc -l", 
                                 shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[0]
    N_empty_OUT = int(N_empty_OUT)
    out = f"""├───────▶︎       Update Stock and dataset files                          [{time.strftime('%H:%M:%S', time.gmtime())}]
├──────────────────────────────────────────────────────────────────────────────────
"""
    with open(f"{path}/{logfile}", "a") as f:
        f.write(out)
    if N_OUTCARS == N_empty_OUT:
        out = f"""│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎  !!! WARNING !!!  Forces certainly > 25eV/A, no structures added to dataset.
│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎  All OUTCAR_{AT.ITERATION}*.inp files are empty ...
│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎  Redo a new : Compare NNPs - get structrs from stock - perform DFT calcs\n"""
        with open(f"{path}/{logfile}", "a") as f:
            f.write(out)
    else:
        N_good_OUT = N_OUTCARS - N_empty_OUT
        out = f"""│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ {N_OUTCARS} OUTCAR_{AT.ITERATION}*.inp files found, 
│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ {N_empty_OUT} empty OUTCAR_{AT.ITERATION}*.inp files found.
│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ {N_good_OUT} OUTCAR_{AT.ITERATION}*.inp structures will be added to the dataset.\n"""
        with open(f"{path}/{logfile}", "a") as f:
            f.write(out)
# Adding a OUTCARS.inp to input.data
        for j in range(1,nnodes+1):
            subprocess.run(f"cat {path}/{stepNB}_vasp{j}/OUTCAR_{AT.ITERATION}*.inp >> {path}/input.data", shell=True)
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Now there are {sum(AT.get_Nstruct())} structures in the dataset.\n")
# Structures up to numbers[:to_add] have been added, remove them from stock:
    for pair in StokIndPairs:
        subprocess.run(f"{mybin}/StructDelFromDat.sh {pair[0]}.data {pair[1]}", shell=True)
    with open(f"{path}/{logfile}", "a") as f:
        f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ Now there are {sum(AT.get_Nstruct(stock = True))} structures in the stock.\n")
    new_Nstock = len(glob.glob1(path,"stock[1-9]*.data"))
    if new_Nstock == AT.Nstock+1:
        # mkdir predictN directories and copy needed files done if ITERATION 0 but also if Nstock increased and the number of predict
        for i in range(1, AT.nNNP+1):
            Path(f"{AT.path}/NNP{i}/predict{new_Nstock+1}").mkdir(parents=True, exist_ok=True)
            subprocess.run(f"cp {AT.path}/input{i}.nn {AT.path}/NNP{i}/predict{new_Nstock+1}/input.nn", shell=True)
            subprocess.run(f"ln -s {AT.path}/stock{new_Nstock+1}.data {AT.path}/NNP{i}/predict{new_Nstock+1}/input.data", shell=True)
        with open(f"{path}/{logfile}", "a") as f:
            f.write(f"│  [{time.strftime('%H:%M:%S', time.gmtime())}]   ├─▶︎ !!!  creation of a new stock{new_Nstock+1}.data file  !!! \n")
        

