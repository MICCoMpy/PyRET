
#import sys
import argparse
from .wavefunctions import *
from .data_classes import *
import numpy as np
#import h5py
#import time
import json
#import os

from .inputparser import Vin, vreadinput
from .Vtools import computeV
#import itertools
import logging

def main(**kwargs):
    
    
    consts = Consts()

    #MPI World:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #print(comm, rank, size)
    
    #WFunction Data
    if rank==0: print(f"============================================================")
    if rank==0: print(f"CALCULATING V \n\n")
    start_timev = MPI.Wtime()
    comm.Barrier()
    
    
    infile= kwargs.get("in")
    vin = vreadinput(infile)
    logging.debug(vin.emdatafile)
    with open(vin.emdatafile, "r") as f:
        Emdata = Emitters_data().Decode(json.load(f))
    

    

    with open(vin.posdatafile, "r") as f:
        jsondata = json.load(f)
        if jsondata["Class"] == "Positions_data":
            Posdata = Positions_data().Decode(jsondata)
        elif jsondata["Class"] == "Positions_data_1config":
            Posdata = Positions_data_1config().Decode(jsondata)
            
    WFdata = []
    
    #Find out all states involved in the transition:
    allstates = []
    for item in vin.absorber_transitions:
        allstates +=item.split("-to-")
    allstates = set(allstates)
    
    #Find out what is the type of the absorber
    absorber_type = Emdata.EmType[vin.absorber_id]
    
    if rank == 0: logging.debug(f"Searching for wavefunctions corresponding to the states {allstates} in for the absorber id = {vin.absorber_id}, type = {absorber_type} ")
    for wfdatafile in vin.wfdatafiles:
        with open(wfdatafile,"r") as f:
            data = json.load(f)
        if absorber_type in data["emitterTypes"]:
            index = data["emitterTypes"].index(absorber_type)
            commonstates = list(set(data["states"][index]) & allstates)
            if commonstates:
                if rank == 0: logging.debug(f"{commonstates} found in wavefunction file {wfdatafile} \n")
                WFdata.append(WFunctions_data().Decode(data))
        
        del data
        
    #ws:
    
    ws = np.array(vin.ws, dtype = complex) 
    """This variable can contain the photon frequencies in the case of a 
    photon spectral analysis
    The value [0] indicates that the 2nd order perturbation based 
    contour sum is already taken into account"""
    Nw = np.size(ws)
    
    V_data = computeV(vin.absorber_id, vin.emitter_id, vin.absorber_transitions,
                     ws, vin.Lmax, Emdata, Posdata, WFdata, radtype_abs = vin.radtype_abs,
                     Hint = vin.Hint)

    if rank == 0:
    
        with open(vin.savepath+vin.savefile, "w") as f:
            json.dump(V_data.Encode(), f)
        print(f"\n\n============================================================")
        print(f"Saved json file {vin.savepath+vin.savefile} containing the V data")
        print(f"Elapsed time on Vdata = {MPI.Wtime()-start_timev}s")
        print("JOB DONE")

    
    comm.Barrier()        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "-in", type = str, help = "Input filename containing the configuration and emitter information for which V are to be calculated. For details please check the inputparser.py file")
    
    args  = parser.parse_args()
    
    
    mydict = {"in": args.i}
    #print("Running main now")
    #print(args)
    main(**mydict)
    