
import sys
import argparse
from pyRET.wavefunctions import *
from pyRET.data_classes import *
import numpy as np

import h5py
import time
import json
import os
from pyRET.inputparser import WFin_QE, wfreadinput
from pyRET.WFtools import get_wfr_and_kwrf_parallel
import itertools
import logging

def main(**kwargs):
    
    
    
    consts = Consts()
    from mpi4py import MPI
    #MPI World:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #WFunction Data
    if rank==0: print(f"============================================================")
    if rank==0: print(f"CALCULATING WAVEFUNCTIONS- DATA - from QE output \n\n")
    start_timew = MPI.Wtime()
    comm.Barrier()
    
    
    infile= kwargs.get("in")
    wfin = wfreadinput(infile)
    
    EmType = wfin.EmType
    Nxyz = wfin.Nxyz
    aA = wfin.aA
    cA = [[aA, 0, 0], [0, aA, 0],[0, 0, aA]]
    
    savepath = wfin.savepath
    savefile = wfin.savefile
    
    
    if rank == 0: logging.debug(f"\n\n Calculating atomic wfs with gridsize {Nxyz}x{Nxyz}x{Nxyz}\n")
    
    #Combined set for all the iKSs for which wavefunctions need to be imported
    
    iKSsunq = set(itertools.chain(*wfin.iKSs))
    wfstemp = {}
    wfrstemp = {}
    kwfrstemp = {}
    
    for iKS in iKSsunq:

        if rank == 0: logging.debug(f"Now extracting for {iKS=}")
        
        wf= WFunction()
        

        #if file is hdf5, use the hdf5 reader, else use the dat reader
        if wfin.wfloadfile.endswith(".hdf5") or wfin.wfloadfile.endswith(".h5"):
            wf.readfromhdf5(wfin.wfloadfile, iKS, cA)
        elif wfin.wfloadfile.endswith(".dat"):
            wf.readfromdat(wfin.wfloadfile, iKS, cA)
        comm.Barrier()
        _, rgridwf, wfrstemp[iKS], kwfrstemp[iKS] = get_wfr_and_kwrf_parallel(wf, iKS, aA, Nxyz)
        wfstemp[iKS]= wf

    Ns = {}
    wfrs = {}
    kwfrs = {}
    wfs = {}
    
    dV = (aA/Nxyz)**3 *1e-30
    comm.Barrier()    
    
    if rank == 0:
        for istate, (state, iKSs, Ws) in enumerate(zip(wfin.states, wfin.iKSs, wfin.Ws)):

            for i, (iKS, W) in enumerate(zip(iKSs, Ws)):
                if i == 0:
                    if rank == 0: logging.debug("iKS=",iKS)
                    if rank == 0: logging.debug("W=",W)

                    logging.debug()
                    wfs[state] = wfstemp[iKS].scale(W)
                    wfrs[state] = W*wfrstemp[iKS]
                    kwfrs[state] = W*kwfrstemp[iKS]

                else:
                    wfs[state] = wfs[state] + wfstemp[iKS].scale(W)
                    wfrs[state] = wfrs[state] + W*wfrstemp[iKS]
                    kwfrs[state] = kwfrs[state] + W*kwfrstemp[iKS]

                if wfin.centering:
                    for axis in [0, 1, 2]:
                        wfrs[state] = np.roll(wfrs[state], int(Nxyz/2), axis = axis)
                        kwfrs[state] = np.roll(kwfrs[state], int(Nxyz/2), axis = axis+1)

            Ns[state] = 1/np.sqrt(np.sum(np.conj(wfrs[state])*wfrs[state])*dV)


        if os.path.exists(wfin.qeoutfile):
            with open(wfin.qeoutfile, "r") as f:
                lines = [line for line in f]
            qeouttext = " ".join(lines)
        else:
            qeouttext = None
            
        WF_data = WFunctions_data([EmType],[wfin.states],[None], [rgridwf], [Ns], [qeouttext], [wfs], [], [wfrs], [kwfrs])

        with open(savepath+savefile , "w") as f:
            json.dump(WF_data.Encode(), f)

        print(f"\n\n============================================================")
        print(f"Wavefunctions extracted and data saved in {savepath}{savefile}")
        print(f"Elapsed time for wavefunction extraction = {MPI.Wtime()-start_timew}s")
    
    
    comm.Barrier()        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "-in", type = str, help = "Input filename containing which wavefunctions to calculate. For details please check the inputparser.py file")
    
    args  = parser.parse_args()
    
    
    mydict = {"in": args.i}
    
    main(**mydict)
    