
import sys
import argparse
from pyRET.wavefunctions import *
from pyRET.data_classes import *
import numpy as np
import h5py
import time
import json
import os
from pyRET.inputparser import WFin_atomic, wfreadinput
from pyRET.WFtools import get_wfr_and_kwrf_parallel
import logging

def main(**kwargs):
    
    consts = Consts()

    #MPI World:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #print(kwargs)
    #Retrieve arguments:
    
    
    infile= kwargs.get("in")
    wfin = wfreadinput(infile)
    
    EmType = wfin.EmType
    Z = wfin.Z
    Nxyz = wfin.Nxyz
    abya0 = wfin.abya0
    savepath = wfin.savepath
    savefile = wfin.savefile
    

    
    
    
    #WFunction Data
    if rank==0: print(f"============================================================")
    if rank==0: print(f"CALCULATING WAVEFUNCTIONS- DATA \n\n")
    start_timew = MPI.Wtime()
    comm.Barrier()
    

    
    aA = abya0*consts.abohr/Z *1e10
    
    
    if rank == 0: logging.debug(f"\n\n Calculating atomic wfs with gridsize {Nxyz}x{Nxyz}x{Nxyz}\n")
    
    cA = [[aA, 0, 0], [0, aA, 0],[0, 0, aA]]
    
    Ns = {}
    wfrs = {}
    kwfrs = {}
    wfs = {}
    
    for istate, (state, n, L, M, W) in enumerate(zip(wfin.states, wfin.ns, wfin.Ls, wfin.Ms, wfin.Ws)):
        if rank == 0: logging.debug(f"Now extracting for {state=}")
        
        wf= WFunction()
        wf.MPField.add_component(1, max(L)+1 , {"type":"Atomic", "n": n, "Z": Z})
        for Lval, Mval, Wval in zip(L, M, W):
            wf.MPField.mps[-1].C[Lval][Mval] = Wval
            
            

        comm.Barrier()
        Ns[state], rgridwf, wfrs[state], kwfrs[state] = get_wfr_and_kwrf_parallel(wf, istate, aA, Nxyz)
        wfs[state]= wf

    if rank == 0:
        WF_data = WFunctions_data([EmType],[wfin.states],[None], [rgridwf], [Ns], [None], [wfs], [], [wfrs], [kwfrs])

        with open(savepath+savefile , "w") as f:
            json.dump(WF_data.Encode(), f)

        print(f"\n\n============================================================")
        print(f"Wavefunctions extracted and data saved in {savepath}{savefile}")
        print(f"Elapsed time for wavefunction extraction = {MPI.Wtime()-start_timew}s")
        print(f"JOB DONE")
    
    
    comm.Barrier()        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "-in", type = str, help = "Input filename containing which wavefunctions to calculate. For details please check the inputparser.py file")
    
    args  = parser.parse_args()
    
    
    mydict = {"in": args.i}
    
    main(**mydict)
    