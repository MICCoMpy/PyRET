import numpy as np

#import h5py
from .constants import Consts
from .wavefunctions import *
from .data_classes import *
from .emfieldtools import *
import time
import logging
#import json
from .parallel_dot import dot_n_VF_par
from .parallel_dot import dot_n_SF_par

def hasWF(em, WF_data):
    """
    Check if the WF_data contains the wavefunction data for the emitter type em

    Args:
        em (str): Emitter type
        WF_data (list): List of wavefunction data objects
    Returns:
        bool: True if the wavefunction data for the emitter type em is found in WF_data, False otherwise
    """
    if WF_data is None:
        return False
    
    for item in WF_data:
        if em in item.emitterTypes:
            return True
    
    return False

def whichWF(em, statei, statef, WF_data):
    """
    Find the indices of the Wavefunction data that corresponds to the states of a specific transition.

    Args:
        em (str): Emitter type
        statei (str): Initial state of the transition
        statef (str): Final state of the transition
        WF_data (list): List of wavefunction data objects
    Returns:
        iwfdatai (int): Index of the WF_data object that contains the wavefunction data for the initial state
        iemi (int): Index of the emitter type in the WF_data object that contains the wavefunction data for the initial state
        iwfdataf (int): Index of the WF_data object that contains the wavefunction data for the final state
        iemf (int): Index of the emitter type in the WF_data object that contains the wavefunction data for the final state
    """    
    for iwf, wfd in enumerate(WF_data):
        if em in wfd.emitterTypes:
            index = wfd.emitterTypes.index(em)
            if statei in wfd.states[index]:
                iwfdatai = iwf
                iemi = index
            if statef in wfd.states[index]:
                iwfdataf = iwf
                iemf = index
            
    
    return iwfdatai, iemi, iwfdataf, iemf

def computeV(
        absorber_id,
        emitter_id,
        absorber_transitions,
        ws,
        Lmax,
        Em_data,
        Pos_data,
        WF_data,
        radtype_abs = "H1",
        parallel = True,
        Hint = "Adotp",
        ):
    
    """
    This function computes the V matrix elements for a given set of transitions and photon energies. It checks if the wavefunction data for the absorber is available in the WF_data, and if so, it uses the wavefunction data to compute the V matrix elements. If not, it uses known multipole moments to compute the V matrix elements. The function can also handle both self-interaction (where the emitter and absorber are the same) and distant interaction (where the emitter and absorber are different) cases. The computed V matrix elements are stored in a V_data object and returned at the end of the function.

    Args:
        absorber_id (str): Identifier for the absorber
        emitter_id (str): Identifier for the emitter
        absorber_transitions (list): List of transitions for the absorber
        ws (numpy array): Array of photon angular frequencies (rad/s)
        Lmax (int): Maximum multipole order to consider for the calculation of V
        Em_data (Emitters_data): Object of the Emitters_data class
        Pos_data (Positions_data or Positions_data_1config): Object of the Positions_data or Positions_data_1config class
        WF_data (list of WFunctions_data): List of objects of WFunctions_data class
        radtype_abs (str, optional): Type of radial function to use for the absorber. Options are "J" or "H1". Defaults to "H1".
        parallel (bool, optional): Whether to use parallel processing. Defaults to True.
        Hint (str, optional): Type of interaction Hamiltonian to use. Options are "Adotp" or "Edotr". Defaults to "Adotp".

    
    Returns:
        Vdata (V_data): Object of the V_data class containing the computed matrix elements.
    """


    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        start_time = MPI.Wtime()
    else:
        rank = 0
        size = 1
        start_time = time.time()
        
    if rank == 0: 
        logging.info("######################################### \n\n\n\n"+
          f"Computing V for energy transfer to {absorber_id} from {emitter_id}\n"+
          f"{absorber_id} has transitions {absorber_transitions} \n",
          f"{emitter_id} Emits at omega =  {ws} \n")
    
    
    ems = [absorber_id, emitter_id]
    
    
    consts = Consts()
    #Initialize data structure
    Vdata = V_data()
    Vdata.ws[ems[0]] = {}
    Vdata.ws[ems[0]][ems[1]] = ws

    Vdata.V[ems[0]] = {}
    Vdata.V[ems[0]][ems[1]] = {}
    
    
    Nw = np.size(ws)
    nIndex = Em_data.nIndex[ems[1]]
    ks = ws*nIndex/consts.c
    #kem1 = Em_data.k[ems[1]]
    #Self Interaction (Multipole center same as the absorber)
    if ems[0] == ems[1]:
        if rank == 0: logging.debug("\n\n Multipole center same as the absorber. Using J-Bessel functions.")
        
        #Checking if an emitter is in the WFdata array
        absorber_type = Em_data.EmType[ems[0]]
        #If wavefunction exists:
        if hasWF(absorber_type, WF_data):#ems[0] in WF_data.emitters:
            if rank == 0: logging.debug("\n\nWavefunction data exist for absorber")
            for trans in absorber_transitions:
                Vdata.V[ems[0]][ems[1]][trans] = {}
                #Get the wavefunction for the corresponding transition
                statei = trans.split("-to-")[0]
                statef = trans.split("-to-")[1]
                if rank == 0: logging.debug(f"\n\nNow calculating for transition {trans}. {statei=} {statef=}")
                
                #Find the correspinding WFdata files and emitter indices for statei and statef
                iwfdatai,iemi,iwfdataf, iemf = whichWF(absorber_type, statei, statef, WF_data) 
                wfri = WF_data[iwfdatai].Ns[iemi][statei]*WF_data[iwfdatai].wfrs[iemi][statei]
                wfrf = WF_data[iwfdataf].Ns[iemf][statef]*WF_data[iwfdataf].wfrs[iemf][statef]
                kwfri = WF_data[iwfdatai].Ns[iemi][statei]*WF_data[iwfdatai].kwfrs[iemi][statei]
                kwfrf = WF_data[iwfdataf].Ns[iemf][statef]*WF_data[iwfdataf].kwfrs[iemf][statef]
                rgridwf =  WF_data[iwfdatai].rgridwf[iemi]
                dV = (rgridwf[2][1,1,1] -rgridwf[2][0,0,0])*(rgridwf[1][1,1,1] -rgridwf[1][0,0,0])*(rgridwf[2][1,1,1] -rgridwf[2][0,0,0]) 
                
                
                if rank == 0: logging.debug(f"Wavefunctions found in WFdata.")
                    
                for L in range(1,Lmax+1):
                    Vdata.V[ems[0]][ems[1]][trans][L] = {}
                    for M in range(-L, L+1):
                        Vdata.V[ems[0]][ems[1]][trans][L][M] = {}
                        for P in [-1, 1]:
                            
                            Vdata.V[ems[0]][ems[1]][trans][L][M][P] = np.zeros((2,2, Nw), dtype = complex)
                            if rank == 0: logging.debug(f"Calculating V for {L=} {M=} {P=}. Dimension {(2,2, Nw)}")
                           
                            if rank == 0: logging.debug(f"Spectral response for {len(ks)} k-points for the photon")
                            for ik, k in enumerate(ks):

                                Vdata.V[ems[0]][ems[1]][trans][L][M][P][:,:,ik] = \
                                    V12_par(wfri,
                                            kwfri,
                                            wfrf,
                                            kwfrf,
                                            rgridwf,
                                            k,
                                            L,
                                            M,
                                            P,
                                            nIndex,
                                            Cgauge = 0,
                                            Hint = Hint,
                                            rshift=np.zeros(3),
                                            radtype = "J",
                                            dV = dV,
                                            parallel = parallel,)

                    
        #If wavefunction does not exists:
        else:
            if rank == 0: logging.debug("\n\nWavefunction data does not exist for absorber- using known multipoles instead")
            for trans in absorber_transitions:
                if rank == 0: logging.debug(f"\n\nNow calculating for transition {trans}")
                Vdata.V[ems[0]][ems[1]][trans] = {}
                for L in range(1,Lmax+1):
                    Vdata.V[ems[0]][ems[1]][trans][L] = {}
                    for M in range(-L, L+1):
                        Vdata.V[ems[0]][ems[1]][trans][L][M] = {}
                        for P in [-1, 1]:
                            Vdata.V[ems[0]][ems[1]][trans][L][M][P] = np.zeros((2,2, Nw), dtype = complex)
                            if rank == 0: logging.debug(f"Calculating V for {L=} {M=} {P=}. Dimension {(2,2, Nw)}")
                            
                            if rank == 0: logging.debug(f"Spectral response for {len(ks)} k-points for the photon")
                            for ik, k in enumerate(ks):

                                if "ED" in trans:
                                    Vdata.V[ems[0]][ems[1]][trans][L][M][P][:,:,ik] = \
                                        V12_ED(Em_data.pMPs[ems[0]][trans][1][-1],np.zeros(3) ,k, L, M, P,nIndex,  "J", Cgauge = 0)
                                else:
                                    Vdata.V[ems[0]][ems[1]][trans][L][M][P][:,:,ik] = \
                                        V12_MD(Em_data.pMPs[ems[0]][trans][1][1],np.zeros(3) ,k, L, M, P,nIndex,  "J", Cgauge = 0)


    
    #Distant Interaction (Multipole center NOT same as the absorber)
    else:
        
        if rank == 0: logging.debug("\n\nMultipole center not same as the absorber. Using Hankel functions. Relative positions defined in Pos_data dataclass")
        
        #if the position data is in the classical notation where each pair have multiple configurations
        if isinstance(Pos_data, Positions_data):
            #Creating the full rvect matrix with the rgrid:
            if ems[0] in Pos_data.Rvects.keys():
                if ems[1] in Pos_data.Rvects[ems[0]].keys():
                    Rvects = Pos_data.Rvects[ems[0]][ems[1]]

            elif ems[1] in Pos_data.Rvects.keys():
                if ems[0] in Pos_data.Rvects[ems[1]].keys():
                    Rvects = - Pos_data.Rvects[ems[1]][ems[0]]
        elif isinstance(Pos_data, Positions_data_1config):
            iabs = list(Em_data.EmType).index(ems[0])
            iemit = list(Em_data.EmType).index(ems[1])
            Rvects = np.reshape(Pos_data.Rvects[:,iabs, iemit], (3,1,1,1))
            
            
            
        #Checking if an emitter is in the WFdata array
        absorber_type = Em_data.EmType[ems[0]]
        
        #If wavefunction exists:
        if hasWF(absorber_type, WF_data):#ems[0] in WF_data.emitters:
            if rank == 0: logging.debug("\n\nWavefunction data exist for absorber")
            for trans in absorber_transitions:
                Vdata.V[ems[0]][ems[1]][trans] = {}
                #Get the wavefunction for the corresponding transition
                statei = trans.split("-to-")[0]
                statef = trans.split("-to-")[1]
                if rank == 0: logging.debug(f"\n\nNow calculating for transition {trans}. {statei=} {statef=}")
                
                #Find the correspinding WFdata files and emitter indices for statei and statef
                iwfdatai,iemi,iwfdataf, iemf = whichWF(absorber_type, statei, statef, WF_data) 
                wfri = WF_data[iwfdatai].Ns[iemi][statei]*WF_data[iwfdatai].wfrs[iemi][statei]
                wfrf = WF_data[iwfdataf].Ns[iemf][statef]*WF_data[iwfdataf].wfrs[iemf][statef]
                kwfri = WF_data[iwfdatai].Ns[iemi][statei]*WF_data[iwfdatai].kwfrs[iemi][statei]
                kwfrf = WF_data[iwfdataf].Ns[iemf][statef]*WF_data[iwfdataf].kwfrs[iemf][statef]
                rgridwf =  WF_data[iwfdatai].rgridwf[iemi]
                dV = (rgridwf[2][1,1,1] -rgridwf[2][0,0,0])*(rgridwf[1][1,1,1] -rgridwf[1][0,0,0])*(rgridwf[2][1,1,1] -rgridwf[2][0,0,0]) 
                
                if rank == 0: logging.debug(f"Wavefunctions found in WFdata. ")
                
                rgrids = np.zeros(np.shape(Rvects)+np.shape(rgridwf[0]))       
                for idir in [0, 1, 2]: #x, y, z directions
                    for i0 in range(np.size(Rvects, 1)):
                        for i1 in range(np.size(Rvects, 2)):
                            for i2 in range(np.size(Rvects, 3)):
                                rgrids[idir, i0, i1, i2] = Rvects[idir, i0, i1, i2]+rgridwf[idir]
                            
                
                    
                for L in range(1,Lmax+1):
                    Vdata.V[ems[0]][ems[1]][trans][L] = {}
                    for M in range(-L, L+1):
                        Vdata.V[ems[0]][ems[1]][trans][L][M] = {}
                        for P in [-1, 1]:
                            Vdata.V[ems[0]][ems[1]][trans][L][M][P] = np.zeros((2,2, Nw)+np.shape(Rvects)[1:], dtype = complex)
                            if rank == 0: logging.debug(f"Calculating V for {L=} {M=} {P=}. Dimension {(2,2, Nw)+np.shape(Rvects)[1:]}")
                            
                            if rank == 0: logging.debug(f"Spectral response for {len(ks)} k-points for the photon")
                            for ik, k in enumerate(ks):

                                Vdata.V[ems[0]][ems[1]][trans][L][M][P][:,:,ik] = \
                                    V12_par(wfri, kwfri, wfrf, kwfrf,rgrids,k, L, M, P,
                                            nIndex, Cgauge = 0, Hint = Hint, rshift=np.zeros(3),
                                            radtype = radtype_abs, dV = dV, parallel = parallel)

                    
        #If wavefunction does not exists:
        else:
            if rank == 0: logging.debug("\n\nWavefunction data does not exist for absorber- using known multipoles instead")
            for trans in absorber_transitions:
                if rank == 0: logging.debug(f"\n\nNow calculating for transition {trans}")
                Vdata.V[ems[0]][ems[1]][trans] = {}
                for L in range(1,Lmax+1):
                    Vdata.V[ems[0]][ems[1]][trans][L] = {}
                    for M in range(-L, L+1):
                        Vdata.V[ems[0]][ems[1]][trans][L][M] = {}
                        for P in [-1, 1]:
                            Vdata.V[ems[0]][ems[1]][trans][L][M][P] = np.zeros((2,2, Nw)+np.shape(Rvects)[1:], dtype = complex)
                            if rank == 0: logging.debug(f"Calculating V for {L=} {M=} {P=}. Dimension {(2,2, Nw)+np.shape(Rvects)[1:]}")
                            
                            if rank == 0: logging.debug(f"Spectral response for {len(ks)} k-points for the photon")
                            for ik, k in enumerate(ks):


                                if "ED" in trans:
                                    Vdata.V[ems[0]][ems[1]][trans][L][M][P][:,:,ik] = \
                                        V12_ED(Em_data.pMPs[ems[0]][trans][1][-1], Rvects, k, L, M, P,nIndex, radtype_abs, Cgauge = 0)
                                else:
                                    Vdata.V[ems[0]][ems[1]][trans][L][M][P][:,:,ik] = \
                                        V12_MD(Em_data.pMPs[ems[0]][trans][1][1], Rvects, k, L, M, P,nIndex, radtype_abs, Cgauge = 0)

    
    
    if rank == 0:
        return Vdata
    else:
        return None
    
############################################
# Methods to compute matrix elements     ##
############################################
def V12(
        wfr1,
        kwfr1,
        wfr2,
        kwfr2,
        rgrids,
        k,
        L,
        M,
        P,
        nIndex = 1.,
        Cgauge = 0,
        Hint = "Adotp",
        rshift=np.zeros(3),
        radtype = "J",
        dV = None,
        etaeff = 1.,
        ):
    
    """
    Computes the matrix elements with specific positions and wavefunctions. This is the core function that computes the V matrix elements for a given set of wavefunctions, multipole parameters, and positions. It can compute the matrix elements for both the "Adotp" and "Edotr" interaction Hamiltonians, and it can also handle different gauges and radial function types. The computed matrix elements are returned as a 2x2 matrix corresponding to the different spin configurations.
    
    Args:
        wfr1 (numpy array of shape (N, N, N)): Wavefunction of the initial state
        kwfr1 (numpy array of shape (3, N, N, N)): Gradient of the wavefunction of the initial state
        wfr2 (numpy array of shape (N, N, N)): Wavefunction of the final state
        kwfr2 (numpy array of shape (3, N, N, N)): Gradient of the wavefunction of the final state
        rgrids (numpy array of shape (3,n1,n2,..nM, N, N, N)): Radial grid for the computation of the matrix elements
        k (float): Wavevector of the emitted photon
        L (int): Orbital angular momentum
        M (int): projected total angular momentum
        P (int): Parity of the multipole
        nIndex (float): Refractive index of the medium. Defaults to 1.
        Cgauge (int): Gauge parameter. 0 for Coulomb gauge.
        Hint (str): Type of interaction Hamiltonian to use. Options are "Adotp" or "Edotr". Defaults to "Adotp".
        rshift (numpy array of shape (3,)): Additional shift in position. Optional.
        radtype (str): Type of radial function to use. Options are "J" for Bessel functions and "H1" for Hankel functions. Defaults to "J".
        dV (float): Volume element for the integration. If None, it will be computed from the rgrids. Defaults to None.
        etaeff (float): Effective mass of electron in the Pauli-Fierz hamiltonian normalized to the free electron mass.    
    
    
    Returns:
        V12 (numpy array of shape (2, 2, n1, n2, ... nM)): matrix elements. The first two dimensions correspond to the spin configurations (spin flip tag on initial and final states), and the remaining dimensions correspond to the spatial grid defined by rgrids.
    
    """
    
    #First, determine the dimension of rgrid:
    Ndim = len(np.shape(rgrids))
    Ndimwf = len(np.shape(wfr1))
    
    Ndimextra = Ndim-1-Ndimwf
    shape = list(np.shape(rgrids))
    outshape = ()
    for i in range(1, Ndim-Ndimwf):
        outshape+=(shape[i],)
    #print(f"Computing V {Ndim=} {Ndimwf=} {outshape = }")
    
    
    #Defining the constants:
    consts = Consts()
    mu0 = consts.mu0
    eps0 = consts.eps0
    lam = 2*np.pi*nIndex/k
    omega = 2*np.pi*consts.c/lam
    e = consts.qe
    hbar= consts.hplank/(2*np.pi)
    hplank = consts.hplank
    me = consts.me*etaeff
    muB = e*consts.hbar/(2*me)
    
    if dV == None:
        #Volume element:
        if Ndimextra == 0:
            dV = (rgrids[0][1,1,1] -rgrids[0][0,0,0])*(rgrids[1][1,1,1] -rgrids[1][0,0,0])*(rgrids[2][1,1,1] -rgrids[2][0,0,0]) 
        elif Ndimextra == 1:
            dV = (rgrids[0][0][1,1,1] -rgrids[0][0][0,0,0])*(rgrids[1][0][1,1,1] -rgrids[1][0][0,0,0])*(rgrids[2][0][1,1,1] -rgrids[2][0][0,0,0])
        elif Ndimextra == 2:
            dV = (rgrids[0][0,0][1,1,1] -rgrids[0][0,0][0,0,0])*(rgrids[1][0,0][1,1,1] -rgrids[1][0,0][0,0,0])*(rgrids[2][0,0][1,1,1] -rgrids[2][0,0][0,0,0])
        elif Ndimextra == 3:
            dV = (rgrids[0][0,0,0][1,1,1] -rgrids[0][0,0,0][0,0,0])*(rgrids[1][0,0,0][1,1,1] -rgrids[1][0,0,0][0,0,0])*(rgrids[2][0,0,0][1,1,1] -rgrids[2][0,0,0][0,0,0])
        else:
            logging.warning("Number of auxiliary dimension out of bounds. Max 3 allowed.")
        
    V12 = np.zeros((2,2)+ outshape, dtype = complex) 

    if Hint ==  "Adotp":
        
        #Initialize the multipole:
        A, A0, E, H = InitializeMultipole([L], [M], [P], [1.], lam , nIndex, L+2, C = Cgauge, radtype=radtype)
        #Fields computed over the radial grid provided as argument
        #print(np.shape(rgrids))
        Ar = A.torgrid(rgrids)
        A0r = A0.torgrid(rgrids)
        Hr = H.torgrid(rgrids)
        divAr = divergence(A).torgrid(rgrids)


        #Spin conserving components
        adotkpsi = (Ar[0]*kwfr1[0]+Ar[1]*kwfr1[1]+Ar[2]*kwfr1[2])
        v12 = -1j*muB *np.sum(np.conj(wfr2)*wfr1*divAr*dV, (-1, -2, -3)) \
                    +2*muB *np.sum(np.conj(wfr2)*adotkpsi*dV, (-1, -2, -3))

        s12 = -e*np.sum(np.conj(wfr2)*wfr1*A0r*dV, (-1, -2, -3))

        #Spin flip components:
        b00 = muB*np.sum(np.conj(wfr2)*mu0 * Hr[2] *wfr1*dV, (-1, -2, -3))
        b01 = muB*np.sum(np.conj(wfr2)*mu0 * (Hr[0]+1j*Hr[1]) *wfr1*dV, (-1, -2, -3))
        b10 = muB*np.sum(np.conj(wfr2)*mu0 * (Hr[0]-1j*Hr[1]) *wfr1*dV, (-1, -2, -3))

        V12[0,0] = v12+s12+b00
        V12[1,1] = v12+s12-b00
        V12[0,1] = b01
        V12[1,0] = b10

        return V12

    elif Hint == "Edotr":
        
        #Initialize the multipole:
        A, A0, E, H = InitializeMultipole([L], [M], [P], [1.], 2*np.pi*nIndex/k , nIndex, L+1, C = Cgauge, radtype=radtype)
        #Fields computed over the radial grid provided as argument
        #Ar = A.torgrid(rgrid)
        #A0r = A0.torgrid(rgrid)
        Er = E.torgrid(rgrids)
        Hr = H.torgrid(rgrids)
        #divAr = divergence(A).torgrid(rgrid)
        
        #Spin conserving components
        Edotr = (Er[0]*rgrids[0]+Er[1]*rgrids[1]+Er[2]*rgrids[2])\
            +(Er[0]*rshift[0]+Er[1]*rshift[1]+Er[2]*rshift[2]) 

        e12 = e*np.sum(np.conj(wfr2)*wfr1*Edotr*dV, (-1, -2, -3))
        
        #Spin flip components:
        b00 = muB*np.sum(np.conj(wfr2)*mu0 * Hr[2] *wfr1*dV, (-1, -2, -3))
        b01 = muB*np.sum(np.conj(wfr2)*mu0 * (Hr[0]+1j*Hr[1]) *wfr1*dV, (-1, -2, -3))
        b10 = muB*np.sum(np.conj(wfr2)*mu0 * (Hr[0]-1j*Hr[1]) *wfr1*dV, (-1, -2, -3))
        
        V12[0,0] = e12+b00
        V12[1,1] = e12-b00
        V12[0,1] = b01
        V12[1,0] = b10

        return V12
        
    return

def V12_par(
        wfr1,
        kwfr1,
        wfr2,
        kwfr2,
        rgrids,
        k,
        L,
        M,
        P,
        nIndex = 1.,
        Cgauge = 0,
        Hint = "Adotp",
        rshift=np.zeros(3),
        radtype = "J",
        dV = None,
        etaeff = 1.,
        parallel = True,
        ):
    
    """
    Parallely computes, using mpi4py, the matrix elements with specific positions and wavefunctions. This is the core function that computes the V matrix elements for a given set of wavefunctions, multipole parameters, and positions. It can compute the matrix elements for both the "Adotp" and "Edotr" interaction Hamiltonians, and it can also handle different gauges and radial function types. The computed matrix elements are returned as a 2x2 matrix corresponding to the different spin configurations. This function also supports serial implementation if parallel flag is set to False.
    
    Args:
        wfr1 (numpy array of shape (N, N, N)): Wavefunction of the initial state
        kwfr1 (numpy array of shape (3, N, N, N)): Gradient of the wavefunction of the initial state
        wfr2 (numpy array of shape (N, N, N)): Wavefunction of the final state
        kwfr2 (numpy array of shape (3, N, N, N)): Gradient of the wavefunction of the final state
        rgrids (numpy array of shape (3,n1,n2,..nM, N, N, N)): Radial grid for the computation of the matrix elements
        k (float): Wavevector of the emitted photon
        L (int): Orbital angular momentum
        M (int): projected total angular momentum
        P (int): Parity of the multipole
        nIndex (float): Refractive index of the medium. Defaults to 1.
        Cgauge (int): Gauge parameter. 0 for Coulomb gauge.
        Hint (str): Type of interaction Hamiltonian to use. Options are "Adotp" or "Edotr". Defaults to "Adotp".
        rshift (numpy array of shape (3,)): Additional shift in position. Optional.
        radtype (str): Type of radial function to use. Options are "J" for Bessel functions and "H1" for Hankel functions. Defaults to "J".
        dV (float): Volume element for the integration. If None, it will be computed from the rgrids. Defaults to None.
        etaeff (float): Effective mass of electron in the Pauli-Fierz hamiltonian normalized to the free electron mass.    
    
    
    Returns:
        V12 (numpy array of shape (2, 2, n1, n2, ... nM)): matrix elements. The first two dimensions correspond to the spin configurations (spin flip tag on initial and final states), and the remaining dimensions correspond to the spatial grid defined by rgrids.
    
    """
    

    if parallel:
        from mpi4py import MPI  
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        start_time = MPI.Wtime()
    else:
        rank = 0
        size = 1
        start_time = time.time()
    
    
    if rank == 0: logging.debug(f"\t Calculating V with parallel processing: {size} Workers")
    
    
    #First, determine the dimension of rgrid:
    Ndim = len(np.shape(rgrids))
    Ndimwf = len(np.shape(wfr1))
    
    Ndimextra = Ndim-1-Ndimwf
    shape = list(np.shape(rgrids))
    outshape = ()
    for i in range(1, Ndim-Ndimwf):
        outshape+=(shape[i],)
    if rank==0: logging.debug(f"\tComputing V {Ndim=} {Ndimwf=} {outshape = }")
    
    
    #Defining the constants:
    consts = Consts()
    mu0 = consts.mu0
    eps0 = consts.eps0
    lam = 2*np.pi*nIndex/k
    omega = 2*np.pi*consts.c/lam
    e = consts.qe
    hbar= consts.hplank/(2*np.pi)
    hplank = consts.hplank
    me = consts.me*etaeff
    muB = e*consts.hbar/(2*me)
    
    if dV == None:
        #Volume element:
        if Ndimextra == 0:
            dV = (rgrids[0][1,1,1] -rgrids[0][0,0,0])*(rgrids[1][1,1,1] -rgrids[1][0,0,0])*(rgrids[2][1,1,1] -rgrids[2][0,0,0]) 
        elif Ndimextra == 1:
            dV = (rgrids[0][0][1,1,1] -rgrids[0][0][0,0,0])*(rgrids[1][0][1,1,1] -rgrids[1][0][0,0,0])*(rgrids[2][0][1,1,1] -rgrids[2][0][0,0,0])
        elif Ndimextra == 2:
            dV = (rgrids[0][0,0][1,1,1] -rgrids[0][0,0][0,0,0])*(rgrids[1][0,0][1,1,1] -rgrids[1][0,0][0,0,0])*(rgrids[2][0,0][1,1,1] -rgrids[2][0,0][0,0,0])
        elif Ndimextra == 3:
            dV = (rgrids[0][0,0,0][1,1,1] -rgrids[0][0,0,0][0,0,0])*(rgrids[1][0,0,0][1,1,1] -rgrids[1][0,0,0][0,0,0])*(rgrids[2][0,0,0][1,1,1] -rgrids[2][0,0,0][0,0,0])
        else:
            logging.warning("Number of auxiliary dimension out of bounds. Max 3 allowed.")
    
    
    V12 = np.zeros((2,2)+ outshape, dtype = complex) 

    if Hint ==  "Adotp":
        
        #Initialize the multipole:
        A, A0, E, H = InitializeMultipole([L], [M], [P], [1.], lam , nIndex, L+2, C = Cgauge, radtype=radtype)
        #Fields computed over the radial grid provided as argument
        #print(np.shape(rgrids))
        """Ar = A.torgrid(rgrids)
        A0r = A0.torgrid(rgrids)
        Hr = H.torgrid(rgrids)
        divAr = divergence(A).torgrid(rgrids)"""


        #Spin conserving components
        if parallel:
            comm.Barrier()
        temp = dot_n_SF_par(rgrids, [np.conj(wfr2), wfr1], divergence(A), parallel = parallel)
        v12_0 = -1j*muB *dV * temp if rank == 0 else None
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),kwfr1[0]], A, 0, parallel = parallel)
        v12_1 = 2*muB  *dV * temp if rank == 0 else None 
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),kwfr1[1]], A, 1,  parallel = parallel)
        v12_2 = 2*muB  *dV * temp if rank == 0 else None 
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),kwfr1[2]], A, 2, parallel = parallel)
        v12_3 = 2*muB  *dV * temp if rank == 0 else None  
        
        if parallel:
            comm.Barrier()
        temp = dot_n_SF_par(rgrids,[np.conj(wfr2),wfr1], A0, parallel = parallel)
        s12 = -e*dV* temp if rank == 0 else None
        

        #Spin flip components:
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1], H, 2, parallel = parallel) 
        b00 = muB*dV*mu0* temp if rank == 0 else None
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1], H, 0, parallel = parallel)
        temp1 = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1], H, 1, parallel = parallel)
        b01 = muB*dV*mu0* (temp + 1j*temp1) if rank == 0 else None
        b10 = muB*dV*mu0* (temp - 1j*temp1) if rank == 0 else None
        
        if parallel:
            comm.Barrier()
        if rank == 0:

            V12[0,0] = v12_0+v12_1+v12_2+v12_3+s12+b00
            V12[1,1] = v12_0+v12_1+v12_2+v12_3+s12-b00
            V12[0,1] = b01
            V12[1,0] = b10

            return V12
        else:
            return None
        
    elif Hint == "Edotr":
        
        #Initialize the multipole:
        A, A0, E, H = InitializeMultipole([L], [M], [P], [1.], 2*np.pi*nIndex/k , nIndex, L+1, C = Cgauge, radtype=radtype)
        #Fields computed over the radial grid provided as argument
        #Ar = A.torgrid(rgrid)
        #A0r = A0.torgrid(rgrid)
        """Er = E.torgrid(rgrids)
        Hr = H.torgrid(rgrids)"""
        #divAr = divergence(A).torgrid(rgrid)
        
        #Spin conserving components
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1,rgrids[0]+rshift[0]],E, 0, parallel = parallel)
        e12_0 =e*dV*temp if rank == 0 else None
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1,rgrids[1]+rshift[1]],E, 1, parallel = parallel)
        e12_1 =e*dV*temp if rank == 0 else None
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1,rgrids[2]+rshift[2]],E, 2, parallel = parallel)
        e12_2 =e*dV*temp if rank == 0 else None
        
        
        
        
        
        #Spin flip components:
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1], H, 2, parallel = parallel) 
        b00 = muB*dV*mu0* temp if rank == 0 else None
        
        if parallel:
            comm.Barrier()
        temp = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1], H, 0, parallel = parallel)
        temp1 = dot_n_VF_par(rgrids,[np.conj(wfr2),wfr1], H, 1, parallel = parallel)
        b01 = muB*dV*mu0* (temp + 1j*temp1) if rank == 0 else None
        b10 = muB*dV*mu0* (temp - 1j*temp1) if rank == 0 else None
        
        
        if rank == 0:
            V12[0,0] = e12_0+e12_1+e12_2+b00
            V12[1,1] = e12_0+e12_1+e12_2-b00
            V12[0,1] = b01
            V12[1,0] = b10

            return V12
        else:
            return None
    return

def V12_ED(pEDs, rvects, k, L, M, P, nIndex = 1., radtype = "J", Cgauge = 0, etaeff = 1., includespin = True):
    
    """
    Computes matrix element under the electric dipole approximation.

    Args:
        pEDs (numpy array of shape (3,)): The x, y, z components of the electric dipole moment vector.
        rvects (numpy array of shape (3, n1, n2, ... nM)): The x, y, z positions of the dipole.
        k (float): The wavevector.
        L, M, P (int): Multipole parameters.
        nIndex (float): Refractive index.
        radtype (str): Type of radial function.
        Cgauge (int): Gauge parameter.
        etaeff (float): Effective mass correction.
        includespin (bool): Whether to include spin effects.

    Returns:
        V12 (numpy array of shape (2, 2, n1, n2, ... nM)): The interaction matrix element under the electric dipole approximation.
    """  
    
    consts = Consts()
    mu0 = consts.mu0
    eps0 = consts.eps0
    lam = 2*np.pi*nIndex/k
    omega = 2*np.pi*consts.c/lam
    e = consts.qe
    hbar= consts.hplank/(2*np.pi)
    hplank = consts.hplank
    me = consts.me*etaeff
    muB = e*consts.hbar/(2*me)
    
    V12 = np.zeros((2,2)+ np.shape(rvects[0]), dtype = complex) 
    
    #Initialize the multipole:
    A, A0, E, H = InitializeMultipole([L], [M], [P], [1.], lam , nIndex, L+1, C = Cgauge, radtype=radtype)
    #Fields computed over the radial grid provided as argument
    #Ar = A.torgrid(rgrid)
    #A0r = A0.torgrid(rgrid)
    Er = E.torgrid(rvects)
    Hr = H.torgrid(rvects)
    #divAr = divergence(A).torgrid(rgrid)
    #Dipole-dipole coupling:
    v12 = Er[0]*pEDs[0]+Er[1]*pEDs[1]+Er[2]*pEDs[2]
    
    if not includespin: return v12*np.identity(2, dtype = complex)
    #Spin flip components:
    b00 = muB*mu0 * Hr[2] 
    b01 = muB*mu0 * (Hr[0]+1j*Hr[1]) 
    b10 = muB * mu0 * (Hr[0]-1j*Hr[1])

    V12[0,0] = v12+b00
    V12[1,1] = v12-b00
    V12[0,1] = b01
    V12[1,0] = b10
    
    
    return V12

def V12_MD(pMDs, rvects, k, L, M, P, nIndex = 1., radtype = "J", Cgauge = 0, etaeff = 1., includespin = True):
    """
    Computes matrix element under the magnetic dipole approximation.

    Args:
        pMDs (numpy array of shape (3,)): The x, y, z components of the magnetic dipole moment vector.
        rvects (numpy array of shape (3, n1, n2, ... nM)): The x, y, z positions of the dipole.
        k (float): The wavevector.
        L, M, P (int): Multipole parameters.
        nIndex (float): Refractive index.
        radtype (str): Type of radial function.
        Cgauge (int): Gauge parameter.
        etaeff (float): Effective mass correction.
        includespin (bool): Whether to include spin effects.

    Returns:
        V12 (numpy array of shape (2, 2, n1, n2, ... nM)): The interaction matrix element under the magnetic dipole approximation.
    """  
    
    consts = Consts()
    mu0 = consts.mu0
    eps0 = consts.eps0
    lam =  2*np.pi*nIndex/k
    omega = 2*np.pi*consts.c/lam
    e = consts.qe
    hbar= consts.hplank/(2*np.pi)
    hplank = consts.hplank
    me = consts.me*etaeff
    muB = e*consts.hbar/(2*me)
    
    V12 = np.zeros((2,2)+ np.shape(rvects[0]), dtype = complex) 
    #Initialize the multipole:
    A, A0, E, H = InitializeMultipole([L], [M], [P], [1.], lam , nIndex, L+1, C = Cgauge, radtype=radtype)
    #Fields computed over the radial grid provided as argument
    #Ar = A.torgrid(rgrid)
    #A0r = A0.torgrid(rgrid)
    #Er = E.torgrid(rvect)
    Hr = H.torgrid(rvects)
    #divAr = divergence(A).torgrid(rgrid)
    #Dipole-dipole coupling:
    v12 =mu0*(Hr[0]*pMDs[0]+Hr[1]*pMDs[1]+Hr[2]*pMDs[2])
    
    if not includespin: return v12*np.identity(2, dtype = complex)

    #Spin flip components:
    b00 = muB*mu0 * Hr[2] 
    b01 = muB*mu0 * (Hr[0]+1j*Hr[1]) 
    b10 = muB * mu0 * (Hr[0]-1j*Hr[1])

    V12[0,0] = v12+b00
    V12[1,1] = v12-b00
    V12[0,1] = b01
    V12[1,0] = b10
    
    
    return V12
    
def Veded(p1vec, p2vec, rvects, nIndex, omega):
    """
    Compute dipole-dipole interaction.

    Args:
        p1vec (numpy array of shape (3,)): The x, y, z components of the first dipole moment vector.
        p2vec (numpy array of shape (3,)): The x, y, z components of the second dipole moment vector.
        rvects (numpy array of shape (3, n1, n2, ... nM)): The x, y, z positions of the dipoles.
        nIndex (float): Refractive index.
        omega (float): Angular frequency.

    mode = "nearfield" or "general"
    
    Returns:
        Veded (numpy array of shape (n1, n2, ... nM)): Dipole-dipole interaction energy in SI units.
    """
    consts = Consts()
    r = np.sqrt(rvects[0]**2 +rvects[1]**2 +rvects[2]**2)
    
    k = nIndex*omega/ consts.c
    
    
    pdotp = np.sum(p1vec*p2vec)
    
    p1dotrhat = (p1vec[0]*rvects[0]+ p1vec[1]*rvects[1]+ p1vec[2]*rvects[2])/r
    p2dotrhat = (p2vec[0]*rvects[0]+ p2vec[1]*rvects[1]+ p2vec[2]*rvects[2])/r
    
    kr = k*r
    f = (kr**2 + 1j*kr - 1)* pdotp + (-kr**2 -3*1j*kr + 3) * p1dotrhat * p2dotrhat
    
    Veded = np.exp(1j*k*r)/(4*np.pi*consts.eps0*nIndex**2 * r**3) * f
    
    return Veded