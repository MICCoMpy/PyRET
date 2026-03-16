"""
Helper script to perform parallel dot products of arrays of shape (..., Nx, Ny, Nz)
where the last three dimensions are the same for all arrays.  The dot product is performed by splitting the last three dimensions into chunks and distributing them across the available processors. Each processor computes a local dot product for its assigned chunk, and then the results are gathered and summed to obtain the final dot product. This approach allows for efficient parallelization of the dot product operation, especially for large arrays.
"""


import numpy as np

import h5py
import time
import json

def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()
    #print(f"MPI world has a size {size}; This process's rank is {rank}")
    comm.Barrier()
    
    A1 = np.random.rand(40,40,40)+ 1j*np.random.rand(40,40,40)
    A2 = np.random.rand(40,40,40)
    A3 = np.random.rand(2,2,3,40,40,40)
    
    
    s= dot_n_par([A1,A2,A3])
    if rank == 0: print(f"The dot product received in process {rank} is {s=}")
    if rank == 0: print(f"Elapsed time = {MPI.Wtime()-start_time}s")
    return

def dot_3(A1, A2, A3):
    Ndimr = 3
    axes = ()
    for i in range(Ndimr): axes+=(-(i+1),)
    return np.sum(A1*A2*A3, axes)

def dot_4(A1, A2, A3, A4):
    Ndimr = 3
    axes = ()
    for i in range(Ndimr): axes+=(-(i+1),)
    return np.sum(A1*A2*A3, axes)

def dot_n(As):
    Ndimr = 3
    axes = ()
    for i in range(Ndimr): axes+=(-(i+1),)
    prod = 1
    for A in As:
        prod = prod*A
    #print(f"{np.shape(prod)=} {axes= }")
    print()
    return np.sum(prod, axis = axes)

def dot_n_par(As):
    Ndimr = 3
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()
    
    
    comm.Barrier()
    #For all the ranks:
    for iA, A in enumerate(As):
        if rank == 0: pass#print(f"\t\t Parallel Dot: Input array no. {iA} has {(np.shape(A))=}")
        Nx, Ny, Nz = (np.shape(A))[-3:]
        break
    Nf = Nx*Ny*Nz
    
    NA = len(As)
    shapes = []
    for A in As:
        shapes.append(np.shape(A)[:-Ndimr])
    
    comm.Barrier()
    #Operation for each rank:
    Aflocs = []
    for shape, A in zip(shapes,As):
        #Reshape each arrays for distribution
        Af = np.reshape(A, shape+(-1,))
        Afloc = np.ascontiguousarray(np.take(Af, range(rank, Nf, size), axis = -1), dtype = complex)
        #print(f"Processor {rank} has subarray of shape {np.shape(Afloc)}")
        Aflocs.append(Afloc)
    
    del Afloc, Af, As
    
    
    #Local dot product:
    prod = 1
    for Afloc in Aflocs:
        prod = prod*Afloc
    locdot = np.sum(prod, axis = -1)
    
    del prod
    
    #Now we can use MPI_Gather to collect the arrays
    #Let us split the real and imaginary parts for this:
    rlocdot = np.ascontiguousarray(np.real(locdot))
    ilocdot = np.ascontiguousarray(np.imag(locdot))
    #print(f"shape of dot products locally = {np.shape(rlocdot)}")
    
    #In rank 0, create a blank variable to be able to collect:
    if rank == 0:
        rdot = np.ascontiguousarray(np.zeros_like(rlocdot))
        idot = np.ascontiguousarray(np.zeros_like(ilocdot))
    else:
        rdot = None
        idot = None
    
    #Gather:
    if rank == 0: pass#print("\t\t Parallel Dot: Gathering")
    
    comm.Barrier()
    comm.Reduce(rlocdot, rdot, op = MPI.SUM)
    comm.Reduce(ilocdot, idot, op = MPI.SUM)
    
    if rank == 0:
        return rdot + 1j*idot
    else:
        return None
    
def dot_n_SF_par(rgrids, As, SF, parallel = True):
    
    if not parallel:
        Sr = SF.torgrid(rgrids)
        prod = 1.
        for A in As:
            prod = prod*A
        locdot = np.sum(prod*Sr, axis = (-1, -2, -3))
        return locdot

    
    Ndimr = 3
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()
    
    
    comm.Barrier()
    #For all the ranks:
    for iA, A in enumerate(As):
        if rank == 0: pass#print(f"\t\t Parallel Dot: Input array no. {iA} has {(np.shape(A))=}")
        Nx, Ny, Nz = (np.shape(A))[-3:]
        break
    Nf = Nx*Ny*Nz
    
    NA = len(As)
    shapes = []
    for A in As:
        shapes.append(np.shape(A)[:-Ndimr])
    
    comm.Barrier()
    #Operation for each rank:
    Aflocs = []
    for shape, A in zip(shapes,As):
        #Reshape each arrays for distribution
        Af = np.reshape(A, shape+(-1,))
        Afloc = np.ascontiguousarray(np.take(Af, range(rank, Nf, size), axis = -1), dtype = complex)
        #print(f"Processor {rank} has subarray of shape {np.shape(Afloc)}")
        Aflocs.append(Afloc)
    
    del Afloc, Af, As
    
    #splicing rgrids:
    rshape = np.shape(rgrids)[:-3]
    rgridsf = np.reshape(rgrids, rshape+(-1,))
    rgridlocf = np.take(rgridsf, range(rank, Nf, size), axis = -1)
    del rgrids, rgridsf
    
    SFloc = np.ascontiguousarray(SF.torgrid(rgridlocf))
    del rgridlocf
    
    
    #Local dot product:
    prod = 1
    for Afloc in Aflocs:
        prod = prod*Afloc
    prod = prod*SFloc
    locdot = np.sum(prod, axis = -1)
    
    del prod
    
    #Now we can use MPI_Gather to collect the arrays
    #Let us split the real and imaginary parts for this:
    rlocdot = np.ascontiguousarray(np.real(locdot))
    ilocdot = np.ascontiguousarray(np.imag(locdot))
    #print(f"shape of dot products locally = {np.shape(rlocdot)}")
    
    #In rank 0, create a blank variable to be able to collect:
    if rank == 0:
        rdot = np.ascontiguousarray(np.zeros_like(rlocdot))
        idot = np.ascontiguousarray(np.zeros_like(ilocdot))
    else:
        rdot = None
        idot = None
    
    #Gather:
    if rank == 0: pass#print("\t\t Parallel Dot: Gathering")
    
    comm.Barrier()
    comm.Reduce(rlocdot, rdot, op = MPI.SUM)
    comm.Reduce(ilocdot, idot, op = MPI.SUM)
    
    if rank == 0:
        return rdot + 1j*idot
    else:
        return None
    
def dot_n_VF_par(rgrids, As, VF, icomp, parallel = True):
    
    if not parallel:
        Vr = VF.torgrid(rgrids)[icomp]
        prod = 1.
        for A in As:
            prod = prod*A
        locdot = np.sum(prod*Vr, axis = (-1, -2, -3))
        return locdot
    
    
    
    
    Ndimr = 3
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()
    
    
    comm.Barrier()
    #For all the ranks:
    for iA, A in enumerate(As):
        if rank == 0: 
            pass
            #print(f"\t\t Parallel Dot: Input array no. {iA} has {(np.shape(A))=}")
        Nx, Ny, Nz = (np.shape(A))[-3:]
        break
        
    Nf = Nx*Ny*Nz
    
    NA = len(As)
    shapes = []
    
    for A in As:
        shapes.append(np.shape(A)[:-Ndimr])
    
    comm.Barrier()
    
    #Operation for each rank:
    Aflocs = []
    
    for shape, A in zip(shapes,As):
        #Reshape each arrays for distribution
        Af = np.reshape(A, shape+(-1,))
        
        if rank == 0: 
            pass
            #print(f"Shape of A:{np.shape(A)},"
            #      +f"Shape of Af:{np.shape(Af)}, poolsize = {size}")
                            
        Afloc = np.ascontiguousarray(np.take(Af, 
                                             range(rank, Nf, size),
                                             axis = -1),
                                     dtype = complex)
        #print(f"Processor {rank} has subarray of shape {np.shape(Afloc)}")
        Aflocs.append(Afloc)
    
    del Afloc, Af, As
    
    #splicing rgrids:
    rshape = np.shape(rgrids)[:-3]
    rgridsf = np.reshape(rgrids, rshape+(-1,))
    rgridlocf = np.take(rgridsf,
                        range(rank, Nf, size),
                        axis = -1)
    
    del rgrids, rgridsf
    
    VFloc = np.ascontiguousarray(VF.torgrid(rgridlocf)[icomp])
    
    
    
    del rgridlocf
    
    
    #Local dot product:
    prod = 1
    for Afloc in Aflocs:
        prod = prod*Afloc
    prod = prod*VFloc
    locdot = np.sum(prod, axis = -1)
    
    del prod
    
    
    
    #Now we can use MPI_Gather to collect the arrays
    #Let us split the real and imaginary parts for this:
    rlocdot = np.ascontiguousarray(np.real(locdot))
    ilocdot = np.ascontiguousarray(np.imag(locdot))
    #print(f"shape of dot products locally = {np.shape(rlocdot)}")
    
    #In rank 0, create a blank variable to be able to collect:
    if rank == 0:
        rdot = np.ascontiguousarray(np.zeros_like(rlocdot))
        idot = np.ascontiguousarray(np.zeros_like(ilocdot))
    else:
        rdot = None
        idot = None
    
    #Gather:
    if rank == 0: pass #print("\t\t Parallel Dot: Gathering")
    
    comm.Barrier()
    comm.Reduce(rlocdot, rdot, op = MPI.SUM)
    comm.Reduce(ilocdot, idot, op = MPI.SUM)
    
    if rank == 0:
        return rdot + 1j*idot
    else:
        return None

    
if __name__=="__main__":
    main()