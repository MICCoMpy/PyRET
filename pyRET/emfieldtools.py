import numpy as np
from .constants import Consts
from .fields import *



############################################
## Related to the photon
############################################

#Initialize a multipole mode
def InitializeMultipole(Ls, Ms, Ps, Ws, lam, nIndex, jmax, C = None, radtype = "J"):
    """
    Initiates a quantized photon multipole mode.

    Args:
        Ls(list of int): List of the total angular momentum quantum numbers of the multipole components
        Ms(list of int): List of the magnetic quantum numbers (J_z projections) of the multipole components
        Ps(list of int): List of the parity of the multipole components (-1)**L for electric type, (-1)**(L+1) for magnetic type
        Ws(list of float): List of the weights of the multipole components.

        lam(float): Wavelength of the mode in vacuum
        nIndex(float): Refractive index of the medium
        jmax(int): Maximum order of the spherical Bessel functions to be included in the expansion
        C(float): Gauge parameter, if None, the Lorenz gauge is used, otherwise the gauge is defined by the value of C.    
    
    Returns:
        Avect (VField): Vector potential of the mode
        A0 (SField): Scalar potential of the mode
        Evect (VField): Electric field of the mode
        Hvect (VField): Magnetic field of the mode
    """

    consts = Consts()
    k = 2*np.pi*nIndex/lam
    c = consts.c
    
    hbar = consts.hbar
    omega = 2*np.pi*c/lam
    mu0 = consts.mu0
    eps0 = consts.eps0
    
    Aconst = 1/(4*np.pi) * np.sqrt(mu0*hbar*omega/np.pi)
    Econst = 1/(4*np.pi) * np.sqrt(hbar*omega/np.pi /eps0)* k /nIndex
    Hconst = 1/(4*np.pi) * np.sqrt(hbar*omega/np.pi /mu0)* k 
    
    
    
    
    
    Avect = VField()
    Avect.add_component(k, jmax, {"type":radtype})
    
    A0 = SField()
    A0.add_component(k, jmax, {"type":radtype})
    
    Evect = VField()
    Evect.add_component(k, jmax, {"type":radtype})
    Hvect = VField()
    Hvect.add_component(k, jmax, {"type":radtype})

    for L, M, P, W in zip(Ls, Ms, Ps, Ws):
        if (-1)**L == P:#Electric type
            if C is None:
                
                Avect.mpvs[0].C[L][L-1][M] = Avect.mpvs[0].C[L][L-1][M]+Aconst*W*np.sqrt((2*L+1)/(L+1))
                A0.mps[0].C[L][M] = A0.mps[0].C[L][M]+ Aconst*W*np.sqrt(L/(L+1))*c/nIndex

            else:
                Avect.mpvs[0].C[L][L-1][M] = Avect.mpvs[0].C[L][L-1][M] + Aconst*W*(np.sqrt((L+1)/(2*L+1))+
                                                                            C*np.sqrt(L/(2*L+1)))
                Avect.mpvs[0].C[L][L+1][M] = Avect.mpvs[0].C[L][L+1][M] + Aconst*W*(np.sqrt(L/(2*L+1))-
                                                                            C*np.sqrt((L+1)/(2*L+1)))
                A0.mps[0].C[L][M] = A0.mps[0].C[L][M]+ Aconst*W*C*c/nIndex
            
            Evect.mpvs[0].C[L][L-1][M] = Evect.mpvs[0].C[L][L-1][M] + 1j *Econst*W*np.sqrt((L+1)/(2*L+1))
            Evect.mpvs[0].C[L][L+1][M] = Evect.mpvs[0].C[L][L+1][M] + 1j *Econst*W*np.sqrt((L)/(2*L+1))
            Hvect.mpvs[0].C[L][L][M] = Hvect.mpvs[0].C[L][L][M]-Hconst*W                                                                 
            
        else:#Magnetic type
            Avect.mpvs[0].C[L][L][M] = Avect.mpvs[0].C[L][L][M]+Aconst*W
            
            Evect.mpvs[0].C[L][L][M] = Evect.mpvs[0].C[L][L][M]+1j*Econst*W  
        
            Hvect.mpvs[0].C[L][L-1][M] = Hvect.mpvs[0].C[L][L-1][M]- Hconst*W*np.sqrt((L+1)/(2*L+1))
            Hvect.mpvs[0].C[L][L+1][M] = Hvect.mpvs[0].C[L][L+1][M]- Hconst*W*np.sqrt((L)/(2*L+1))
            
    return Avect, A0, Evect, Hvect

# Calculate E and H fields from the vector and scalar potentials given as VField and SField objects, respectively
def getEHfromA(Avect, A0, nIndex):
    """
    Given the vector potential and scalar potential of a mode, calculates the electric and magnetic fields.

    Args:
        Avect (VField): Vector potential of the mode
        A0 (SField): Scalar potential of the mode
        nIndex (float): Refractive index of the medium

    Returns:
        Evect (VField): Electric field of the mode
        Hvect (VField): Magnetic field of the mode
    """
    consts = Consts()
    k = Avect.ks[0]
    c = consts.c
    lda = 2*np.pi/(k/nIndex)
    omega = 2*np.pi*c/lda
    mu0 = consts.mu0
    
    Hvect = curl(Avect).scale(1/mu0)
    Evect = Avect.copy().scale(omega*1j) + gradient(A0).scale(-1)

    return Evect, Hvect
                                                                      