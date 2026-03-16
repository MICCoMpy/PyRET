"""
Use this python code to extract wavefunction data from hdf5 files outputed by Quantum Espresso to custom json files
that can be used for matrix element computation using py-RET
"""


import os, sys
cwd = os.getcwd()
path1 = os.path.dirname(cwd)
path2 = os.path.dirname(path1)
sys.path.append(path2)

import numpy as np
from mpi4py import MPI
import h5py
from pyRET import *
import time
import json

consts = Consts()



#MPI World:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# print(rank)
# if rank == 0: print(f"MPI world has a size {size}")

#WFunction Data
if rank==0: print(f"============================================================")
if rank==0: print(f"WAVEFUNCTIONS- DATA \n\n")
start_time = MPI.Wtime()

Nxyz = 40
states = ["VOs", "cbm", "VOpx","VOpy","VOpz"]
iKSs = [252, 253, 254, 255, 256]

wfpath = os.path.join(os.path.dirname(os.getcwd()), "pyRET-Data", "Data", "VO_MgO", "up.hdf5")
pwoutpath = os.path.join(os.path.dirname(os.getcwd()), "pyRET-Data", "Data", "VO_MgO", "pw.out")

if rank == 0: print(f"\n\n Importing wavefunctions from {wfpath} with gridsize {Nxyz}x{Nxyz}x{Nxyz}\n")

#Wavefunction object
aA = 8.43202
cA = [[aA, 0, 0], [0, aA, 0],[0, 0, aA]]

with open(pwoutpath, "r") as f:
    lines = f.readlines()
    pwtxt = " ".join([l.strip() for l in lines])

    
wfs = [WFunction() for _ in states]

for iwf, wf in enumerate(wfs):
    wf.readfromhdf5(wfpath,iKSs[iwf],cA)
    N, rgridwf, wfr, kwfr = get_wfr_and_kwrf_parallel(wf, iKSs[iwf], 4.216, Nxyz=40)
    
    
    wfd = WFunctions_data(emitterTypes = ["B"], states = [[states[iwf]]],
                 iKSs = [[iKSs[iwf]]],
                 rgridwf = [rgridwf],
                 Ns = [{states[iwf]: N}],
                 pwout = [pwtxt],
                 wfs = [{states[iwf]: wf}],
                 kwfs = [],
                 wfrs = [{states[iwf]: wfr}],
                 kwfrs = [{states[iwf]: kwfr}])
    
    
    if rank == 0: 
        print(f"Done for {iwf=}")    
    
        newpath = f"./Extracted_Wavefunctions/VO_MgO"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        with open(f"{newpath}/{states[iwf]}.json" , "w") as f:
            json.dump(wfd.Encode(), f)
    
    comm.Barrier()
    if rank == 0:
        print(f"Elapsed time for wavefunction extraction = {MPI.Wtime()-start_time}s")
        print(f"\n\n============================================================")
        