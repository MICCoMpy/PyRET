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
# from mpi4py import MPI
# import h5py
from pyRET import *
import time
import json

consts = Consts()



# #MPI World:
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# print(rank)
# if rank == 0: print(f"MPI world has a size {size}")

#WFunction Data
# if rank==0: print(f"============================================================")
# if rank==0: print(f"WAVEFUNCTIONS- DATA \n\n")
# start_time = MPI.Wtime()

aA = 7.136012
cA = [[aA, 0, 0], [0, aA, 0],[0, 0, aA]]
wfpath = os.path.join(os.path.dirname(os.getcwd()), "pyRET-Data", "Data", "NV", "wfc1.dat")
pwoutpath = os.path.join(os.path.dirname(os.getcwd()), "pyRET-Data", "Data",  "NV", "pw.out")


#Import the G and EVC array from the wavefunction file
with open(wfpath, 'rb') as f:
    # Moves the cursor 4 bytes to the right
    f.seek(4)

    ik = np.fromfile(f, dtype='int32', count=1)[0]
    xk = np.fromfile(f, dtype='float64', count=3)
    ispin = np.fromfile(f, dtype='int32', count=1)[0]
    gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
    scalef = np.fromfile(f, dtype='float64', count=1)[0]

    # Move the cursor 8 byte to the right
    f.seek(8, 1)

    ngw = np.fromfile(f, dtype='int32', count=1)[0]
    igwx = np.fromfile(f, dtype='int32', count=1)[0]
    npol = np.fromfile(f, dtype='int32', count=1)[0]
    nbnd = np.fromfile(f, dtype='int32', count=1)[0]

    # Move the cursor 8 byte to the right
    f.seek(8, 1)

    b1 = np.fromfile(f, dtype='float64', count=3)
    b2 = np.fromfile(f, dtype='float64', count=3)
    b3 = np.fromfile(f, dtype='float64', count=3)

    f.seek(8,1)
    
    mill = np.fromfile(f, dtype='int32', count=3*igwx)
    mill = mill.reshape( (igwx, 3) ) 

    evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")

    f.seek(8,1)
    for i in range(nbnd):
        evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
        f.seek(8, 1)

bcrystal = 2*np.pi /aA * 1e10
Gs = bcrystal * mill.T

#Read pw.out file
with open(pwoutpath, "r") as f:
    lines = f.readlines()
    pwtxt = " ".join([l.strip() for l in lines])


Nxyz = 40
iKSs = [ 87, 122, 123, 126, 127, 128] 
states = [f"{iKS}" for iKS in iKSs]

# if rank == 0: 
print(f"\n\n Importing wavefunctions from {wfpath} with gridsize {Nxyz}x{Nxyz}x{Nxyz}\n")

    
wfs = [WFunction() for _ in states]
for iiKS, (iKS, wf) in enumerate(zip(iKSs, wfs)):
    wf.Gs = Gs
    wf.evcCs = evc[iKS-1,:]
    wf.isGamma = True
    wf.NG = Gs.shape[1]
    wf.isplanewave = True

#comm.Barrier()
print(f"wf objects created {len(wfs)}")




for iwf, wf in enumerate(wfs):
    N, rgridwf, wfr, kwfr = get_wfr_and_kwrf(wf, iKSs[iwf], aA, Nxyz=Nxyz)

    for axis in [0, 1, 2]:
        wfr = np.roll(wfr, int(Nxyz/2), axis = axis)
        kwfr = np.roll(kwfr, int(Nxyz/2), axis = axis+1)
    
    
    wfd = WFunctions_data(emitterTypes = ["NV"], states = [[states[iwf]]],
                 iKSs = [[iKSs[iwf]]],
                 rgridwf = [rgridwf],
                 Ns = [{states[iwf]: N}],
                 pwout = [pwtxt],
                 wfs = [{states[iwf]: wf}],
                 kwfs = [],
                 wfrs = [{states[iwf]: wfr}],
                 kwfrs = [{states[iwf]: kwfr}])
    
    
    print(f"Done for {iwf=}")    

    newpath = f"./Extracted_Wavefunctions/NV"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open(f"{newpath}/{states[iwf]}.json" , "w") as f:
        json.dump(wfd.Encode(), f)

    print(f"============================================================\n\n")
        