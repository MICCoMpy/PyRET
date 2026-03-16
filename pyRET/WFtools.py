"""
Helper functions for wavefunction processing.
"""


import numpy as np
import h5py
from .wavefunctions import *
import time
import json
import logging


#========================================================================
#Functions:
def Split1D(Nf, Np):
    #Find way to split an array of size Nf into Np processes
    
    Nsplit = Nf//Np
    #print(Nsplit)
    
    Nrem =Nf -  Nsplit*(Np-1)
    #print(Nrem)
    if Nrem<=Nsplit:
        sendNs = (Nsplit,)*(Np-1)+ (Nrem,)
    else:
        Nextra = Nrem-Nsplit
        sendNs = (Nsplit+1,)*(Nextra)+ (Nsplit,)*(Np-Nextra)
        
    temp = np.cumsum(sendNs)
    istarts = (0,)+ tuple(temp[:-1])
    
    return sendNs, istarts

def get_wfr_and_kwrf_parallel(wf, iKS, aA, Nxyz):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()

    #=============================================================================
    #Position Grid on the root:
    if rank==0:
        xmatA = np.linspace(-1, 1, Nxyz)*aA/2
        ymatA = np.linspace(-1, 1, Nxyz)*aA/2
        zmatA = np.linspace(-1, 1, Nxyz)*aA/2
        xgrid, ygrid, zgrid = np.meshgrid(xmatA*1e-10, ymatA*1e-10, zmatA*1e-10)
        xf, yf, zf = np.ravel(xgrid),np.ravel(ygrid),np.ravel(zgrid)
        Nf = np.size(xf)
        
        logging.debug(f"{Nf=}")
        #Also, defining the way the numpy matrix will be sliced:
        sendNs, istarts = Split1D(Nf, size)
        
    else:
        xf, yf, zf = None, None, None
        sendNs, istarts = None, None
    
    #Broadcase the parallelization indices to all the processes:
    sendNs = comm.bcast(sendNs, root = 0)
    istarts = comm.bcast(istarts, root = 0)


    
    #=========================================================================================
    #Start of the parallel computation
    #Define the local position grid:
    xfloc = np.ascontiguousarray(np.zeros(sendNs[rank]))
    yfloc = np.ascontiguousarray(np.zeros(sendNs[rank]))
    zfloc = np.ascontiguousarray(np.zeros(sendNs[rank]))
    #comm.Barrier()
    comm.Scatterv([xf, sendNs, istarts, MPI.DOUBLE], xfloc, root = 0)
    #comm.Barrier()
    comm.Scatterv([yf, sendNs, istarts, MPI.DOUBLE], yfloc, root = 0)
    #comm.Barrier()
    comm.Scatterv([zf, sendNs, istarts, MPI.DOUBLE], zfloc, root = 0)
    logging.debug(f"\tprocess {rank} recieved a position grid of dimension Nx = {np.shape(xfloc)} Ny = {np.shape(yfloc)} Nz = {np.shape(zfloc)}")
    
    
    
    comm.Barrier()
    
    #==========================================================================================
    #Computation
    wfrloc = wf.torgrid(np.array([xfloc, yfloc, zfloc])) 
    kwfrloc = wf.kpsitorgrid(np.array([xfloc, yfloc, zfloc]))
    
    wfrloc_real = np.ascontiguousarray(np.real(wfrloc))
    wfrloc_imag = np.ascontiguousarray(np.imag(wfrloc))
    kwfrlocx_real = np.ascontiguousarray(np.real(kwfrloc[0,:]))
    kwfrlocx_imag = np.ascontiguousarray(np.imag(kwfrloc[0,:]))
    kwfrlocy_real = np.ascontiguousarray(np.real(kwfrloc[1,:]))
    kwfrlocy_imag = np.ascontiguousarray(np.imag(kwfrloc[1,:]))
    kwfrlocz_real = np.ascontiguousarray(np.real(kwfrloc[2,:]))
    kwfrlocz_imag = np.ascontiguousarray(np.imag(kwfrloc[2,:]))
    
    logging.debug(f"\tComputation done -- Elapsed time on processor {rank} = {MPI.Wtime()-start_time}s")
    
    
    
    comm.Barrier()
    #==========================================================================================
    #Now, gather:
    if rank == 0:
        logging.debug("\tGathering")
        wfr_real = np.ascontiguousarray(np.zeros(Nf))
        kwfrx_real = np.ascontiguousarray(np.zeros(Nf))
        kwfry_real = np.ascontiguousarray(np.zeros(Nf))
        kwfrz_real = np.ascontiguousarray(np.zeros(Nf))
        wfr_imag = np.ascontiguousarray(np.zeros(Nf))
        kwfrx_imag = np.ascontiguousarray(np.zeros(Nf))
        kwfry_imag = np.ascontiguousarray(np.zeros(Nf))
        kwfrz_imag = np.ascontiguousarray(np.zeros(Nf))
        
    else:
        wfr_real = None
        kwfrx_real = None
        kwfry_real = None
        kwfrz_real = None
        wfr_imag = None
        kwfrx_imag = None
        kwfry_imag = None
        kwfrz_imag = None
        
    comm.Gatherv(wfrloc_real, [wfr_real, sendNs, istarts, MPI.DOUBLE], root = 0)
    comm.Gatherv(wfrloc_imag, [wfr_imag, sendNs, istarts, MPI.DOUBLE], root = 0)
    logging.debug(f"\tprocess {rank} gathered into wfr arrays")
    
    comm.Gatherv(kwfrlocx_real, [kwfrx_real, sendNs, istarts, MPI.DOUBLE], root = 0)
    comm.Gatherv(kwfrlocx_imag, [kwfrx_imag, sendNs, istarts, MPI.DOUBLE], root = 0)
    logging.debug(f"\tprocess {rank} gathered into kwfrx arrays")
    
    comm.Gatherv(kwfrlocy_real, [kwfry_real, sendNs, istarts, MPI.DOUBLE], root = 0)
    comm.Gatherv(kwfrlocy_imag, [kwfry_imag, sendNs, istarts, MPI.DOUBLE], root = 0)
    logging.debug(f"\tprocess {rank} gathered into kwfry arrays")
    
    comm.Gatherv(kwfrlocz_real, [kwfrz_real, sendNs, istarts, MPI.DOUBLE], root = 0)
    comm.Gatherv(kwfrlocz_imag, [kwfrz_imag, sendNs, istarts, MPI.DOUBLE], root = 0)
    logging.debug(f"\tprocess {rank} gathered into kwfrz arrays")

    #============================================================================
    
    #Saving the total wavefunction at the node 0:
    #Using the same data format as the WFunctions_data class:
    
        
    if rank  == 0:
        rgridwf =np.array([xgrid, ygrid, zgrid])
        wfr = np.reshape(wfr_real+1j*wfr_imag, (Nxyz, Nxyz, Nxyz))
        kwfr = np.reshape(np.array([kwfrx_real+1j*kwfrx_imag, 
                                                      kwfry_real+1j*kwfry_imag, 
                                                      kwfrz_real+1j*kwfrz_imag]), (-1,Nxyz, Nxyz, Nxyz))
        
        dxA = xmatA[1]-xmatA[0]
        dyA = ymatA[1]-ymatA[0]
        dzA = zmatA[1]-zmatA[0]
        dV = dxA*dyA*dzA*1e-30
        
        N = 1/np.sqrt(np.sum(np.conj(wfr)*wfr)*dV)
        return N, rgridwf, wfr, kwfr
    else:
        return None, None, None, None

def get_wfr_and_kwrf(wf, iKS = 0, aA = 10, Nxyz = 40):
    xmatA = np.linspace(-1, 1, Nxyz)*aA/2
    ymatA = np.linspace(-1, 1, Nxyz)*aA/2
    zmatA = np.linspace(-1, 1, Nxyz)*aA/2
    xgrid, ygrid, zgrid = np.meshgrid(xmatA*1e-10, ymatA*1e-10, zmatA*1e-10)
    rgridwf =np.array([xgrid, ygrid, zgrid])
    
    
    wfr = wf.torgrid(rgridwf) 
    kwfr = wf.kpsitorgrid(rgridwf)
    dxA = xmatA[1]-xmatA[0]
    dyA = ymatA[1]-ymatA[0]
    dzA = zmatA[1]-zmatA[0]
    dV = dxA*dyA*dzA*1e-30
    
    N = 1/np.sqrt(np.sum(np.conj(wfr)*wfr)*dV)
    
    
    return N, rgridwf, wfr, kwfr
    
