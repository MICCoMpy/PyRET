from .constants import Consts
import numpy as np
from scipy import special
from copy import deepcopy
import logging


#Function to add vector spherical harmonics:
def addMP_VA(mps):
    """
    Helper function to add vector multipoles in a list.
    
    Args:
        mps(list of MP_VA): list of MP_VA objects to be added together
    
    Returns:
        MP_VA: combined MP_VA object
    """
    
    jmax = mps[0].jmax
    mpout = MP_VA(jmax)
    
    for mp in mps:
        for j in range(0, jmax+1):
            if j==0:
                lrange = [1]
            else:
                lrange = [j-1, j, j+1]
                
            for l in lrange:
                for M in range(-j,j+1):
                    mpout.C[j][l][M] = mpout.C[j][l][M]+mp.C[j][l][M]
    
    return mpout


#Function to add vector spherical harmonics:
def addMP_VA_Hansen(mps):
    """
    Helper function to add Hansen vector multipoles in a list.

    Args:
        mps(list of MP_VA_Hansen): list of MP_VA_Hansen objects
    
    Returns:
        MP_VA_Hansen: combined MP_VA_Hansen object

    """
    jmax = mps[0].jmax
    mpout = MP_VA_Hansen(jmax)
    
    for mp in mps:
        
        
        for j, value in mpout.C.items():
            for  M, value1 in value.items():
                for lam, value2 in value1.items():     
                    mpout.C[j][M][lam] = mpout.C[j][M][lam]+mp.C[j][M][lam]
                    
    
    return mpout        
        

# Multipolar vector angular class
class MP_VA():
    """
    Class representing Vector Multipole in angular coordinates with definite l values (j = Total angular momentum, l = orbital angular momentum, M = projected total angular momentum)
    
    Attributes:
        C (dict of dict of dict of complex): Dictionary containing multupole coefficients C[j][l][M]
        jmax (int): integer value of max angular momentum
        
        
    """
    def __init__(self, jmax):
        
        C = {}
        
        for j in range(0, jmax+1):
            C[j] = {}
            if j==0:
                lrange = [1]
            else:
                lrange = [j-1, j, j+1]
            
            for l in lrange:
                
                C[j][l] = {}
                
                for M in range(-j, j+1):
                    C[j][l][M] = 0.+0.0*1j
        
        self.C = C
        self.jmax = jmax
        return
    
    def Encode(self):
        Encoded = {}
        Encoded.update({"Class": self.__class__.__name__})
        
        jmax = self.jmax
        Encoded.update({"jmax": jmax})
        
        C =  {}
        for j in range(0, jmax+1):
            C[f"{j=}"] = {}
            if j==0:
                lrange = [1]
            else:
                lrange = [j-1, j, j+1]
            for l in lrange:
                C[f"{j=}"][f"{l=}"] = {}
                for M in range(-j, j+1):
                    C[f"{j=}"][f"{l=}"][f"{M=}"] = [np.real(self.C[j][l][M]), np.imag(self.C[j][l][M])]
        Encoded.update({"C": C})
        
        return Encoded
    
    def Decode(self, data):
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.jmax = data["jmax"]
        
        for j in range(0, self.jmax+1):
            if j==0:
                lrange = [1]
            else:
                lrange = [j-1, j, j+1]
            for l in lrange:
                for M in range(-j, j+1):
                    self.C[j][l][M] = data["C"][f"{j=}"][f"{l=}"][f"{M=}"][0] + 1j*data["C"][f"{j=}"][f"{l=}"][f"{M=}"][1]
        
        return self
         
    def decompose(self, V, theta, phi, dtheta, dphi):
        """
        Multipole decomposition of a given vector field
        Args:
            V (array of shape (3, Ntheta, Nphi)): Vector field over angular coordinates theta and phi. The first dimension is the vector component (r, theta, phi)
            theta (array of shape (Ntheta, Nphi)): Theta coordinates
            phi (array of shape (Ntheta, Nphi)): Phi coordinates
            dtheta (float): step size in theta
            dphi (float): step size in phi

        """
        #Decompose V into spherical harmonics:
        #Exploiting the orthogonality of the basis:
        
        
        for keyj, value in self.C.items():
            for  keyl, value1 in value.items():
                for keyM, value2 in value1.items():
                    
                    #Computing the integral of dot products:
                    yv = Yvect(keyj, keyl, keyM, theta, phi)
                    
                    c = np.sum((np.conj(yv[0])*V[0]
                               +np.conj(yv[1])*V[1]
                               +np.conj(yv[2])*V[2])*np.sin(theta)*dtheta*dphi)
                    value1.update({keyM: c})
        return
    
    def vector(self):
        """
        Transform the multipole coefficients in vector form.
        """
        V = []
        for keyj, value in self.C.items():
            for  keyl, value1 in value.items():
                for keyM, value2 in value1.items():        
                    V.append(value2)
        return np.array(V)
    
    def fromvector(self, V):
        """
        Get the object from a given complex vector.
        """
        N = np.size(V)
        jmax = np.sqrt((N+2)/3)-1
        if np.abs(jmax - round(jmax))>0.01:
            logging.warning("Vector length not what is expected.. filling in absent elements with zeros")
        jmax = round(np.ceil(jmax))
        
        self.jmax = jmax
        count = 0
        C = {}
        for j in range(0, jmax+1):
            C[j] = {}
            if j==0:
                lrange = [1]
            else:
                lrange = [j-1, j, j+1]
            
            for l in lrange:
                
                C[j][l] = {}
                
                for M in range(-j, j+1):
                    if count<np.size(V):
                        C[j][l][M] = V[count]
                    else:
                        C[j][L][M] = 0+0*1j
                    count = count+1
        self.C = C
        
        return self
    
    def absMP(self, func):
        """
        Absolute magnitude of sum of all (or specific angular components)
        
        Args:
            func (callable): A function that takes (j, l, M) and returns True if the component should be included.
        """
        val = 0
        for j in range(0, self.jmax+1):
            if j==0:
                lrange = [1]
            else:
                lrange = [j-1, j, j+1]
            for l in lrange:
                
                for M in range(-j,j+1):
                    if func(j,l,M):
                        val = val+np.abs(self.C[j][l][M])**2
        
        return val
    
    def scale(self,factor):
        """
        Scale all the coefficients with a specific constant (complex) factor.
        
        Args:
            factor (complex): The scaling factor.

        """
        #print(self.C)
        for j, val1 in self.C.items():
            for l, val2 in val1.items():
                for M, val3 in val2.items():
                    self.C[j][l][M]=factor*self.C[j][l][M]
        return self

    def toMP_VA_Hansen(self):
        """
        Transfer the MP_VA class object to the MP_VA_Hansen object.
        """
        mpvajM = MP_VA_Hansen(self.jmax)
        jmax = self.jmax
        C = {}
        for j in range(0, jmax+1):
            C[j] = {}
            for M in range(-j, j+1):
                C[j][M] = {}
                if j == 0:
                    #lam=0
                    #C[j][M][0] = 0
                    #lam=1
                    #C[j][M][1] = np.sqrt((j)/(2*j+1)) * self.C[j][1][M]
                    #lam=-1
                    C[j][M][-1] = - np.sqrt((j+1)/(2*j+1)) * self.C[j][1][M]
                    
                else:
                    #lam=0
                    C[j][M][0] = self.C[j][j][M]
                    #lam=1
                    C[j][M][1] = np.sqrt((j+1)/(2*j+1)) * self.C[j][j-1][M] + np.sqrt((j)/(2*j+1)) * self.C[j][j+1][M]
                    #lam=-1
                    C[j][M][-1] = np.sqrt((j)/(2*j+1)) * self.C[j][j-1][M] - np.sqrt((j+1)/(2*j+1)) * self.C[j][j+1][M]
                    
                    
                        
        mpvajM.C = C            
                    
        return mpvajM
    
    def plotamp(self, axis, jmax = None):
        """
        Plot the multipole coefficients of vector multipoles
        """
        xvals = []
        yvals = []
        labels = []
        colors = []
        colorvals = ["black", "red","blue","green","cyan","orange"]
        if jmax == None:
             jmax = self.jmax
        #Plot the C coefficients on to the axis that is passed as an argument
        for j, val1 in self.C.items():
            if j>jmax:
                break
                
            for l, val2 in val1.items():
                for M, val3 in val2.items():
                    xvals.append(3*j**2 +6*j +(l-j)*(2*j+1) + M*0.9)
                    yvals.append(np.abs(val3))
                    colors.append(colorvals[abs(M)])
                    labels.append(f"{j=},{l=},{M=}")
        
        axis.bar(xvals, yvals, color= colors)
        
        axis.set_xticks(xvals)
        axis.set_xticklabels(labels, rotation=45, ha='right')
        
        return
    
    def copy(self):
        
        MPcopy = MP_VA(self.jmax)
        MPcopy.C = deepcopy(self.C)
        
        return MPcopy
    

# Multipolar vector angular (Hansen type) class (modes with fixed j and M):
class MP_VA_Hansen():
    """
    Class representing Hansen Vector Multipole (Angular only) in j , M basis
    (j = Total angular momentum, M = projected total angular momentum, lam = a parameter to determine which mode-- longitudinal(-1), and two transverse (0, 1)  
    
    Attributes:
        C (dict of dict of dict of complex): Dictionary containing multupole coefficients C[j][M][lam]
        jmax (int): integer value of max angular momentum
        
    """
   
    def __init__(self, jmax):
        
        C = {}
        
        for j in range(0, jmax+1):
            C[j] = {}
            if not j == 0:
                for M in range(-j, j+1):
                    C[j][M] = {}
                    for lam in [-1, 0, 1]:
                        C[j][M][lam] = 0.+0.0*1j
            else:
                for M in range(-j, j+1):
                    C[j][M] = {}
                    for lam in [-1]:
                        C[j][M][lam] = 0.+0.0*1j
            
        self.C = C
        self.jmax = jmax
        return
    
    def Encode(self):
        Encoded = {}
        Encoded.update({"Class": self.__class__.__name__})
        
        jmax = self.jmax
        Encoded.update({"jmax": jmax})
        
        C =  {}
        for j in range(0, jmax+1):
            C[f"{j=}"] = {}
            if not j == 0:
                for M in range(-j, j+1):
                    C[f"{j=}"][f"{M=}"] = {}
                    for lam in [-1, 0, 1]:
                        C[f"{j=}"][f"{M=}"][f"{lam=}"] = [np.real(self.C[j][M][lam]), np.imag(self.C[j][M][lam])]
            else:
                for M in range(-j, j+1):
                    C[f"{j=}"][f"{M=}"] = {}
                    for lam in [-1]:
                        C[f"{j=}"][f"{M=}"][f"{lam=}"] = [np.real(self.C[j][M][lam]), np.imag(self.C[j][M][lam])]

        Encoded.update({"C": C})        
        return Encoded
    
    def Decode(self, data):
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.jmax = data["jmax"]
        
        
        for j in range(0, self.jmax+1):
            if not j == 0:
                for M in range(-j, j+1):
                    for lam in [-1, 0, 1]:
                        self.C[j][M][lam] = data["C"][f"{j=}"][f"{M=}"][f"{lam=}"][0] + 1j*data["C"][f"{j=}"][f"{M=}"][f"{lam=}"][1]
            else:
                for M in range(-j, j+1):
                    for lam in [-1]:
                        self.C[j][M][lam] = data["C"][f"{j=}"][f"{M=}"][f"{lam=}"][0] + 1j*data["C"][f"{j=}"][f"{M=}"][f"{lam=}"][1]
        
        return self
    
    def decompose(self, V, theta, phi, dtheta, dphi):
        
        for keyj, value in self.C.items():
            for  keyM, value1 in value.items():
                for keylam, value2 in value1.items():
                    
                    #Computing the integral of dot products:
                    yv = Yvect_Hansen(keyj, keyM, keylam, theta, phi)
                    
                    c = np.sum((np.conj(yv[0])*V[0]
                               +np.conj(yv[1])*V[1]
                               +np.conj(yv[2])*V[2])*np.sin(theta)*dtheta*dphi)
                    value1.update({keylam: c})
        return
    
    def vector(self):
        
        V = []
        for keyj, value in self.C.items():
            for  keyM, value1 in value.items():
                for keylam, value2 in value1.items():        
                    V.append(value2)
        return np.array(V)    
    
    def fromvector(self, V):
        
        N = np.size(V)
        jmax = np.sqrt((N+2)/3)-1
        if np.abs(jmax - round(jmax))>0.01:
            logging.warning("Vector length not what is expected.. filling in absent elements with zeros")
        jmax = round(np.ceil(jmax))
        
        self.jmax = jmax
        count = 0
        C = {}
        for j in range(0, jmax+1):
            C[j] = {}
            for M in range(-j, j+1):
                C[j][M] = {}
                if not j == 0:
                    for lam in [-1, 0, 1]:
                        if count<np.size(V):
                            C[j][M][lam] = V[count]
                        else:
                            C[j][M][lam] = 0+0*1j
                        count = count+1
                else:
                    for lam in [-1]:
                        if count<np.size(V):
                            C[j][M][lam] = V[count]
                        else:
                            C[j][M][lam] = 0+0*1j
                        count = count+1
        self.C = C
        
        return self
    
    def scale(self,factor):
        
        for val1 in self.C.values():
            for val2 in val1.values():
                for val3 in val2.values():
                    val3=val3*factor
        return self
    
    def toMP_VA(self):
        
        mpva = MP_VA(self.jmax)
        jmax = self.jmax
        C = {}
        for j in range(0, jmax+1):
            C[j] = {}
            if j==0:
                l=1
                    
                C[j][l] = {}
                    
                for M in range(-j, j+1):
                    C[j][l][M] = - np.sqrt((j+1)/(2*j+1))*self.C[j][M][-1] 
                    
            else:
                lrange = [j-1, j, j+1]      
                for l in lrange:
                    
                    C[j][l] = {}
                    
                    for M in range(-j, j+1):
                        if j == l:
                            C[j][l][M] = self.C[j][M][0]
                        if j+1 == l:
                            C[j][l][M] = - np.sqrt((j+1)/(2*j+1))*self.C[j][M][-1] + np.sqrt(j/(2*j+1))*self.C[j][M][1]
                        if j-1 == l:
                            C[j][l][M] =   np.sqrt((j+1)/(2*j+1))*self.C[j][M][1] + np.sqrt(j/(2*j+1))*self.C[j][M][-1]
        mpva.C = C            
                    
        return mpva
    
    def copy(self):
        
        MPcopy = MP_VA_Hansen(self.jmax)
        MPcopy.C = deepcopy(self.C)
        
        return MPcopy
    

    
        