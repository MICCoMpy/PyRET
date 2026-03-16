import numpy as np
from scipy import special
from .constants import Consts
import logging
import matplotlib.pyplot as plt

#########################################
# Methods relating to angular functions #
#########################################

def Y(l, m , theta, phi):
    """
    Calculates the scalar spherical harmonics of radial order l and angular order m
    
    Args:
        l (int): number representing radial order
        m (int): number representing angular order
        theta (numpy.ndarray): numpy array containing polar angle values
        phi (numpy.ndarray): numpy array containing azimuthal angle values
        
    Returns: 
        numpy.ndarray: numpy array containing spherical harmonic values
    """
    
    
    if m > l or m < -l:
        return(np.zeros(np.shape(theta)))
    Y = special.sph_harm_y(m,l, phi, theta)#the convention of theta vs phi of scipy is opposite !!
    
    return Y

def Y_r(l, m, rgrid):
    """Calculates the scalar spherical harmonics of radial order l and angular order m
    
    Args:
        l (int): number representing radial order
        m (int): number representing angular order
        rgrid (numpy.ndarray): numpy array containing coordinates. Here x = rgrid[0], y = rgrid[1], z = rgrid[2]
    Returns: 
        Y (numpy.ndarray): numpy array containing spherical harmonic values
    """

    _, theta, phi = cart2pol(rgrid, np.zeros(3))
    return Y(l, m, theta, phi)

def cart2pol(rgrid, origin):
    """Converts cartesian coodrinates to polar coordinates
    
    Args:
        rgrid (numpy.ndarray): numpy array containing coordinates. Here x = rgrid[0], y = rgrid[1], z = rgrid[2]
        origin (numpy.ndarray): numpy array of shape (3,) containing cartesian coordinates of origin
        
    Returns: 
        r (numpy.ndarray): numpy array containing r coordinate values
        theta (numpy.ndarray): numpy array containing theta coordinate values
        phi (numpy.ndarray): numpy array containing phi coodrinate values
    """
    xgrid = rgrid[0]-origin[0]
    ygrid = rgrid[1]-origin[1]
    zgrid = rgrid[2]-origin[2]
    
    r = np.sqrt(xgrid**2 + ygrid**2 + zgrid**2)
    
    r1 = r.copy()
    if hasattr(r1, "__len__"):
        np.place(r1, r1==0., [1e-15])
    elif r1 == 0.:
        r1 = 1e-15
    else:
        pass
    
    theta = np.arccos(zgrid/r1)
    
    rho = np.sqrt(xgrid**2 + ygrid**2)
    if hasattr(rho, "__len__"):
        np.place(rho, rho==0., [np.inf])
    elif rho==0:
        rho = np.inf
    else:
        pass
    
    phi = np.arccos(xgrid/rho) 
    phi = phi -2*phi*(ygrid<0)
    #phi = np.nan_to_num(phi)
    
    
    return r, theta, phi

def pol2cart(r, theta, phi, origin):
    
    """
    Converts polar coodrinates to cartesian coordinates
    
    Args:
        r (numpy.ndarray): numpy array of arbitrary dimensions containing r coordinates
        theta (numpy.ndarray): numpy array of arbitrary dimensions containing theta coordinates
        phi (numpy.ndarray): numpy array of arbitrary dimensions containing phi coordinates
        origin (numpy.ndarray): numpy array of shape (3,) containing cartesian coordinates of origin
        
    Returns: 
        numpy.ndarray: numpy array for positions np.array([x, y, z])
    """
    
    xgrid = r*np.sin(theta)*np.cos(phi)+origin[0]
    ygrid = r*np.sin(theta)*np.sin(phi)+origin[1]
    zgrid = r*np.cos(theta)+origin[2]
    
    return np.array([xgrid, ygrid, zgrid])

def projectrad(V,theta, phi):
    
    """Projects a vector field from cartesian coodinates to polar coordinates
    
    Args:
        V (numpy.ndarray): numpy array np.array([xcomp, ycomp, zcomp])
        theta (numpy.ndarray): numpy array containing theta coordinate values
        phi (numpy.ndarray): numpy array containing phi coodrinate values
    
    Returns:
        tuple: tuple of numpy arrays containing the r, theta, and phi projections
    """
    
    xcomp = V[0]
    ycomp = V[1]
    zcomp = V[2]
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)
    
    rcomp = xcomp*st*cp + ycomp*st*sp+zcomp*ct
    thetacomp = xcomp*ct*cp + ycomp*ct*sp -zcomp*st
    phicomp = -xcomp*sp + ycomp*cp
    
    return rcomp, thetacomp, phicomp

def projectcart(V,rgrid):
    
    """
    Projects a vector field from polar coordinate to cartesian coordinate
    
    Args:
        V (numpy.ndarray): numpy array np.array([rcomp, thetacomp, phicomp])
        rgrid (numpy.ndarray): numpy array containing array([x, y, z])
    
    Returns:
        tuple of numpy arrays containing the x, y, and z projections
    """
    
    rcomp = V[0]
    tcomp = V[1]
    pcomp = V[2]
    
    r, theta, phi = cart2pol(rgrid, np.zeros(3))
    
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)
    
    xcomp = rcomp*st*cp+tcomp*ct*cp - pcomp*sp
    ycomp = rcomp*st*sp+tcomp*ct*sp + pcomp*cp
    zcomp = rcomp*ct-tcomp*st
    
    
    return xcomp, ycomp, zcomp

def Vcross(V1, V2):
    
    """
    Takes cross product of two vector fields  in the xyz basis over same coordinate grid.
    
    Args:
        V1, V2 (numpy.ndarray): Vector fields in the xyz basis-- as numpy arrays of shape (3,..,..,..,), first dimension being xy, y, z components
    
    Returns:
        numpy.ndarray: the vector field for the cross product
    """
    
    return np.cross(V1, V2, axisa = 0, axisb = 0,axisc = 0)

    
    
    
############################################
# Methods related to the radial functions ##
############################################

def J_sph(l, r):
    return(special.spherical_jn(l,r))

def Y_sph(l, r):
    return(special.spherical_yn(l,r))

def Hankel1_sph(l, r):
    return(J_sph(l,r)+ 1j*Y_sph(l,r))

def Hankel2_sph(l, r):
    return(J_sph(l,r)- 1j*Y_sph(l,r))

def g(l, r, rfun):
    """ Calculates the radial function of a specific type over specified position
        
    Args:    
        l (int): number representing radial order
        r (numpy.ndarray): values of the radial cordinates
        rfun (dict): dictionary contining various properties of the radial functions such as "type","n", "Z"..
    
    Returns: 
        numpy.ndarray: containing values of g
    """
    if rfun.get("type") == "J":
        return(4*np.pi* (1j)**l * J_sph(l,r))
    elif rfun.get("type") == "H1":
        return(4*np.pi* (1j)**l * Hankel1_sph(l,r))
    
    elif rfun.get("type") == "H2":
        return(4*np.pi* (1j)**l * Hankel2_sph(l,r))
    
    elif rfun.get("type") == "Atomic":
        """
        For this specific case, the atomic wavefunctions
        corresponding to the Hydrogen like atom are generated
        Here n is the principal quantum number
        l is the angular momentum quantum number
        Z is the atomic number

        """
        n = rfun.get("n",1)
        Z = rfun.get("Z",1)
        
        if n<=l:
            return np.zeros(np.shape(r))
        #print(f"Atomic radial part with {n=} {l=} {Z=}")
        rho = 2*Z*r/(n*Consts().abohr)
        assl = special.assoc_laguerre(rho, n-l-1, 2*l+1)
        
        Norm = np.sqrt((2*Z/(n*Consts().abohr))**3 * special.factorial(n-l-1) / (2*n*(special.factorial(n+l))**3))
        
        return Norm *assl* (rho**l) * np.exp(-rho/2)
    
def gfar(l, k, rfun, dR = 0):
    """Approximates the radial function at extreme far field:
        For kr >> 1, Only valid for radial function J and H1
    """
    if rfun.get("type") == "J":
        return(4*np.pi* (1j)**l *(1/k)*np.sin(k*dR -l/2 * np.pi) )
    elif rfun.get("type") == "H1":
        return(4*np.pi* (1j)**l * (1/k)* (1j)**(-l-1) * np.exp(1j*k*dR))
    
def gnear(l, r, rfun):
    """Approximates the radial function at extreme near field:
        For kr  << 1, Only valid for radial function J
    """
    
    if rfun.get("type") == "J":
        return(4*np.pi*(1j*r)**l/special.factorial2(2*l+1))
    else:
        logging.error("Diverging value for the radial function")
        return g(l, r, rfun)

def j2g(l):
    """Converting bessel j to the g(kr) functions
    """
    return 4*np.pi* (1j)**l


#############################################
# Methods related to the spin of the photon #
#############################################

#Spin matrices:
def s_mat():
    """Returns Pauli Spin Matrices 
    """
    
    sx = np.array([[0, 0, 0],
                  [0, 0, -1j],
                  [0, 1j, 0]])
    sy = np.array([[0, 0, 1j],
                  [0, 0, 0],
                  [-1j, 0, 0]])
    sz = np.array([[0, -1j, 0],
                  [1j,  0, 0],
                  [0,   0, 0]])
    
    return np.array([sx, sy, sz])

#Spin vectors as a function of the spin quantum number mu:
def xi(mu):
    """Returns photon spin eigenfunctions
    """
    
    if mu == 0:
        return np.array([0, 0, 1])
    elif mu == 1:
        return -1/np.sqrt(2) * np.array([1, 1j, 0])
    elif mu == -1:
        return 1/np.sqrt(2) * np.array([1, -1j, 0])
    


##################################################################################
# Methods related to the vector spherical matrics (spin included)
##################################################################################

def C_coeffs(j , l, M, mu):
    """ 
    Coeffients for combining the bare spin and angular momentums. These are derived from the Wigner-3j symbols.
    
    Args:
        j (int): total angular momentum quantum number
        l (int): orbital angular momentum quantum number
        M (int): magnetic quantum number
        mu (int): spin projection quantum number
    Returns:
        float: the coefficient value
    """
    if j == l+1:
        if mu == -1:
            #print (f"{j=} {l=} {M=}")
            return np.sqrt((l-M)*(l-M+1)/(2*l+1) /(2*l+2))
        elif mu == 0:
            return np.sqrt((l+M+1)*(l-M+1)/(2*l+1) /(l+1))
        elif mu == 1:
            return np.sqrt((l+M)*(l+M+1)/(2*l+1) /(2*l+2))
        else:
            return 0
    
    elif j == l:
        if mu == -1:
            return np.sqrt((l-M)*(l+M+1)/(2*l) /(l+1))
        elif mu == 0:
            return M/np.sqrt(l*(l+1))
        elif mu == 1:
            return -np.sqrt((l+M)*(l-M+1)/(2*l) /(l+1))
        else:
            return 0
        
    elif j == l-1:
        if mu == -1:
            return np.sqrt((l+M)*(l+M+1)/(2*l) /(2*l+1))
        elif mu == 0:
            return -np.sqrt((l+M)*(l-M)/(l) /(2*l+1))
        elif mu == 1:
            return np.sqrt((l-M)*(l-M+1)/(2*l) /(2*l+1))
        else:
            return 0
    
    else:
        return 0

def Yvect(j, l, M, theta, phi):
    """
    spherical vector harmonics.

    Args:
        j (int): total angular momentum quantum number
        l (int): orbital angular momentum quantum number
        M (int): magnetic quantum number
        theta (numpy.ndarray): numpy array containing polar angle values
        phi (numpy.ndarray): numpy array containing azimuthal angle values
    Returns:
        numpy.ndarray: the vector spherical harmonics as a numpy array of shape (3, ...), where the first dimension corresponds to the x, y, z components of the vector field, and the rest of the dimensions correspond to the angular grid defined by theta and phi.
    """
    out = np.array([np.zeros(np.shape(theta)), np.zeros(np.shape(theta)), np.zeros(np.shape(theta))])
    if j == 0:
        if not l == 1:
            return out
    elif j>0:
        if not j-1 <= l <= j+1:
            return out
        
    
    Ym1 = C_coeffs(j, l, M, -1)* Y(l, M+1, theta, phi)
    Y0 =  C_coeffs(j, l, M,  0)* Y(l, M, theta, phi)
    Y1 =  C_coeffs(j, l, M,  1)* Y(l, M-1, theta, phi)
    #print(Ym1)
    #print(Y0)
    #print(Y1)
    #print(f"{np.shape(Ym1)=}{np.shape(Y0)=}{np.shape(Y1)=}")
    out = np.array([(Ym1* xi(-1)[i]+ Y0*xi(0)[i]+Y1* xi(1)[i]) for i in range(3)])
    
    return out

def Yvect_Hansen(j, M, lam, theta, phi):
    """
    spherical vector harmonics with specific j and M states. Parameter lam =-1 means longitudinal,and lam = 0 or 1 means transverse.

    Args:
        j (int): total angular momentum quantum number
        M (int): magnetic quantum number
        lam (int): polarization state, -1 for longitudinal, 0 and 1 for transverse
        theta (numpy.ndarray): numpy array containing polar angle values
        phi (numpy.ndarray): numpy array containing azimuthal angle values
    Returns:
        numpy.ndarray: the vector spherical harmonics as a numpy array of shape (3, ...), where the first dimension corresponds to the x, y, z components of the vector field, and the rest of the dimensions correspond to the angular grid defined by theta and phi.
    """ 
    if lam == -1:
        #Longitudinal component:
        return (np.sqrt(j/(2*j+1))*Yvect(j, j-1, M, theta, phi) - np.sqrt((j+1)/(2*j+1))*Yvect(j, j+1, M, theta, phi))
    elif lam == 0:
        #One of the transverse components:
        return Yvect(j, j, M, theta, phi)
    elif lam == 1:
        #The other transverse component:
        return (np.sqrt(j/(2*j+1))*Yvect(j, j+1, M, theta, phi) + np.sqrt((j+1)/(2*j+1))*Yvect(j, j-1, M, theta, phi))
    else:
        return np.zeros(np.shape(theta))
    
def vdot(v1, v2):
    """
    Returns a scalar field over the same radial grid which is the vector dot product
    """
    #First axis is the vector. rest of the axes are the radial grid
    #Conplex conjugation automatically done
    s = np.zeros(np.shape(v1[0]))
    for i in range(np.size(v1, axis = 0)):
        s = s + np.conj(v1[i])*v2[i]
        
    return s 
    

                                                                                                                                                 
############################################
##          Utility functions:::
##############################################                 

def getED(T1, omega, nIndex):
    """
    Get transition electric dipole moment from radiative lifetime.

    Args:
        T1 (float): radiative lifetime
        omega (float): angular frequency of the transition
        nIndex (float): refractive index of the medium
    
    Returns:
        float: electric dipole moment pED
    """
    consts = Consts()
    
    pED = np.sqrt(consts.hplank/T1 *   6*consts.eps0*  consts.c**3 /omega**3 /nIndex)
    
    return pED

def getMD(T1, omega, nIndex):
    """
    Get transition magnetic dipole moment from radiative lifetime for purely MD transitions.

    Args:
        T1 (float): radiative lifetime
        omega (float): angular frequency of the transition
        nIndex (float): refractive index of the medium

    Returns:
        float: magnetic dipole moment mMD
    """
    consts = Consts()
    
    mMD = np.sqrt(consts.hplank* 6*consts.c**3 /(T1* nIndex**3 * omega**3 *consts.mu0))
    
    return mMD
                                                      
def getT1fromED(pED, omega, nIndex):
    """
    Computes the radiative lifetime from electric transition dipole moment.

    Args:
        pED (float or numpy.ndarray): electric dipole moment, either a scalar or a vector of size 3
        omega (float): angular frequency of the transition
        nIndex (float): refractive index of the medium

    Returns:
        float: radiative lifetime T1
    """
    
    consts = Consts()
    if np.shape(pED)[0] == 3:
        psq = pED[0]**2 +pED[1]**2 +pED[2]**2  
    else:
        psq = pED**2
    
    T1 = consts.hplank/psq *   6*consts.eps0*  consts.c**3 /omega**3 /nIndex
    return T1

###############################################
# Visualization utility
###############################################

def visualize_Matrix(
        M,
        label = "",
        mode = "full", #Takes value  = "full", or "zero or nonzero"
        ):
    
    if mode == "full":
        f = plt.figure()
        ax = plt.subplot(111)
        p = ax.pcolormesh(np.abs(M))
        plt.colorbar(p)
        ax.set(title = f"{label}")
        return
    
    #else if mode = "zero or nonzero"    
    
    Mdummy = 2*np.ones_like(M, dtype = int)
    Mdummy[np.isnan(M)] = 0
    Mdummy[M == 0] = 1
    
    f = plt.figure()
    ax = plt.subplot(111)
    p = ax.pcolormesh(Mdummy)
    plt.colorbar(p)
    ax.set(title = f"{label}\n 0=nan, 1=0, 2=nz")
    return


    
    
    
    
    
    
    
    
    







