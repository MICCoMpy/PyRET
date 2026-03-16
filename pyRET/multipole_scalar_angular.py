from .constants import Consts

import numpy as np
from scipy import special
from copy import deepcopy
from .utils import Y
import logging


#Function to add spherical harmonics
def addMP(mps):
    lmax = mps[0].lmax
    mpout = MP_SA(lmax)
    
    for mp in mps:
        for l in range(0, lmax+1):
            for m in range(-l,l+1):
                mpout.C[l][m] = mpout.C[l][m]+mp.C[l][m]
    
    return mpout

#Multipole scalar angular class
class MP_SA():
    """
    Class representing Scalar Multipole in angular coordinates

    Attributes:
        C: Dictionary containing multipole coefficients. C[l][m] is the coefficient of the multipole with angular momentum l and m.
        lmax: integer value of max angular momentum. Here l = 0,1,2,...,lmax. m = -l, -l+1,...,l-1,l
        
    """
    def __init__(self, lmax):
        
        C = {}
        
        for l in range(0, lmax+1):
            C[l] = {}
            for m in range(-l,l+1):
                C[l][m] = 0 + 0*1j
        
        self.C = C
        self.lmax = lmax
        return
    
    def Encode(self):
        Encoded = {}
        Encoded.update({"Class": self.__class__.__name__})
        
        lmax = self.lmax
        Encoded.update({"lmax": lmax})
        
        C =  {}
        for l in range(0, lmax+1):
            C[f"{l=}"] = {}
            for m in range(-l,l+1):
                C[f"{l=}"][f"{m=}"] = [np.real(self.C[l][m]), np.imag(self.C[l][m])]
        Encoded.update({"C": C})
        
        return Encoded
    
    def Decode(self, data):
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.lmax = data["lmax"]
        for l in range(0, self.lmax+1):
            for m in range(-l,l+1):
                self.C[l][m] = data["C"][f"{l=}"][f"{m=}"][0] + 1j*data["C"][f"{l=}"][f"{m=}"][1]
        
        return self
    
    def decompose(self, V, theta, phi, dtheta, dphi):
         
        for keyl, value in self.C.items():
            for  keym, value1 in value.items():
                c = np.sum(np.conj(Y(keyl, keym,theta, phi))*V*np.sin(theta)*dtheta*dphi)
                value.update({keym: c})
        return
    
        
    def vector(self):

        V = []
        for keyl, value in self.C.items():
            for  keym, value1 in value.items():                   
                V.append(value1)
        return np.array(V)
    
    def fromvector(self, V):
    #Get the object from a given complex vector
        N = np.size(V)
        lmax = np.sqrt(N)-1
        if np.abs(lmax - round(lmax))>0.01:
            logging.warning("Vector length not what is expected.. filling in absent elements with zeros")
        lmax = int(round(lmax))

        self.lmax = lmax
        count = 0
        C = {}
        for l in range(0, lmax+1):
            C[l] = {}
            for m in range(-l,l+1):
                if count<np.size(V):
                    C[l][m] = V[count]
                else:
                    C[l][m] =0+0*1j
                count = count+1
        self.C = C
        
        return self
    
    def absMP(self, func):
    
        val = 0
        for l in range(0, self.lmax+1):
            for m in range(-l,l+1):
                if func(l,m):
                    val = val+np.abs(self.C[l][m])**2
        
        return val
           
    def scale(self,factor):
        
        for l, val1 in self.C.items():
            for m, val2 in val1.items():
                self.C[l][m] = factor*self.C[l][m]
        return self
    
    def copy(self):
        
        MPcopy = MP_SA(self.lmax)
        MPcopy.C = deepcopy(self.C)
        
        return MPcopy
    
    def plotamp(self, axis, lmax = None):
        
        xvals = []
        yvals = []
        labels = []
        colors = []
        colorvals = ["black", "red","blue","green","cyan","orange", "purple", "grey"]
        if lmax == None:
             lmax = self.lmax
                
        #Plot the C coefficients on to the axis that is passed as an argument
        for l, val1 in self.C.items():
            if l>lmax:
                break
                
            for m, val2 in val1.items():
                
                xvals.append(3*l**2 +6*l + m*0.9)
                yvals.append(np.abs(val2))
                colors.append(colorvals[abs(m)])
                labels.append(f"{l=}{m=}")

        axis.bar(xvals, yvals, color= colors)
        
        axis.set_xticks(xvals)
        axis.set_xticklabels(labels, rotation=45, ha='right')
        
        return    
        
        

