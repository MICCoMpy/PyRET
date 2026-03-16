import numpy as np
from scipy import special

import h5py
from copy import deepcopy
from .fields import *
import logging


class WFunction():
    
    """
    Class representing  a Wavefunction
    
    Attributes:
        isGamma(bool): True if the wavefuncton is at Gamma point
        Gs(numpy array): numpy complex array shape (3, NG) denoting the G vectors for plane wave expansion 
        evcCs(numpy array): numpy complex array of shape (1, NG) - plane wave expansion coefficients for a specific KS state
        NG(int): int storing number of plane waves
        isplanewave(bool): True when the plane wave expansion exists.
        MPField(SField): SField object storing the multipole expanded scalar field. can be empty, in that case isplanewave = True
        
        
    Methods:
        __init__(): Initialization
        Encode(): encodes to jsonizable data
        Decode(data): decodes from json data
        readfromhdf5(self, filename, iKS, cpA): reads a specific KS state iKS into the plane wave basis 
            from a hdf5 file generrated by quantum espresso. cpA = cell parameter block.
        readfromhdf5_parallel(self, filename, iKS, cpA, comm):reads a specific KS state iKS into the plane wave basis 
            from a hdf5 file generrated by quantum espresso. cpA = cell parameter block. Uses parallel hdf5.
        torgrid(self, rgrid, istart = 0, istep = 1, kparity=1): Calculates values at coordinate space.
            istart and istep can be used to skip over plane waves. kparity is the parity in k-space- specifically relevant
            for calculating at Gamma point.
        getharmonics(self, lmax): Using the plane wave basis, analytically creates a Scalar Field object into the attr. MPField
        scale(fact): Scales the Wavefunction by a complex scalar fact and returns a copy.
        copy(): deep copy
        
    """

    def __init__(self):
        
        #Defined based on a planne wave basis
        self.isGamma = None
        self.Gs = None
        self.evcCs = None
        self.NG = None
        self.isplanewave = False
        #Definition in the multipole basis
        self.MPField = SField()
        
            
        return
       
    def Encode(self):
        Encoded = {}
        Encoded.update({"Class": self.__class__.__name__})
        Encoded.update({"isGamma": self.isGamma})
        
        Encoded.update({"Gs(Real)": np.real(self.Gs).tolist()})
        Encoded.update({"Gs(Imag)": np.imag(self.Gs).tolist()})
        
        Encoded.update({"evcCs(Real)": np.real(self.evcCs).tolist()})
        Encoded.update({"evcCs(Imag)": np.imag(self.evcCs).tolist()})
        
        
        Encoded.update({"NG": self.NG})
        Encoded.update({"isplanewave": self.isplanewave})
        
        Encoded.update({"MPField": self.MPField.Encode()})
        
        return Encoded
    
    def Decode(self, data):

        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.isGamma = data.get("isGamma")
        if data.get("Gs(Imag)") is None or data.get("Gs(Real)") is None:
            self.Gs = None
        else:
            self.Gs = np.array(data["Gs(Real)"])+ 1j*np.array(data["Gs(Imag)"])
            
        if data.get("evcCs(Real)") is None or data.get("evcCs(Imag)") is None:
            self.evcCs = None
        else:
            self.evcCs = np.array(data["evcCs(Real)"])+ 1j*np.array(data["evcCs(Imag)"])
        
        if data.get("NG") is None:
            self.NG = None
        else:
            self.NG = data["NG"]
        self.isplanewave = data.get("isplanewave")
        self.MPField.Decode(data.get("MPField"))
        
        return self
    
    def readfromhdf5(self, filename, iKS, cpA):
        """
        Reads the plane waves from hdf5 file

        Args:
            filename (str): path to the hdf5 file containing the plane wave expansion of the wavefunction
            iKS (int): index of the KS state to be read from the hdf5
            cpA (array-like): cell parameter block

        """
        
    
        aA = cpA[0][0]
        bcrystal = 2*np.pi /aA * 1e10
        #For now, lets assume that it is a cubic unit cell to estimate the reciproal lattice vectors:
    
    
    
        #Read the miller indices and the coefficients from the hdf5 file
        with h5py.File(filename,"r") as f:
            Gs = bcrystal* np.swapaxes(np.array(f["MillerIndices"]), 0,1)
            evc = np.array(f["evc"])
        NG = np.size(Gs, 1)
    
    
        #Isolate the real and imaginary parts of evc:
        evcR = evc[iKS, range(0, 2*np.size(Gs,1), 2)]
        evcI = evc[iKS, range(1, 2*np.size(Gs,1), 2)]
        #Define complex miller index:
        evcC = evcR+ 1j*evcI
        self.Gs = Gs
        self.evcCs = evcC
        self.NG = NG
        
        #Check if the Gs contains the parity also:
        
        #Find a G that is not zero:
        if np.sum(Gs) == 0:
            self.isGamma = False
        else:
            self.isGamma = True
        
                
        self.isplanewave = True    
        return self
    
    def readfromhdf5_parallel(self, filename, iKS, cpA, comm):
        """
        Reads the plane waves from hdf5 file in parallel

        Args:
            filename (str): path to the hdf5 file containing the plane wave expansion of the wavefunction
            iKS (int): index of the KS state to be read from the hdf5
            cpA (array-like): cell parameter block
            comm (MPI.Comm): MPI communicator for parallel reading

        """
        
    
        aA = cpA[0][0]
        bcrystal = 2*np.pi /aA * 1e10
        #For now, lets assume that it is a cubic unit cell to estimate the reciproal lattice vectors:
    
    
    
        #Read the miller indices and the coefficients from the hdf5 file
        with h5py.File(filename,"r", driver = "mpio", comm = comm) as f:
            Gs = bcrystal* np.swapaxes(np.array(f["MillerIndices"]), 0,1)
            evc = np.array(f["evc"])
        NG = np.size(Gs, 1)
    
    
        #Isolate the real and imaginary parts of evc:
        evcR = evc[iKS, range(0, 2*np.size(Gs,1), 2)]
        evcI = evc[iKS, range(1, 2*np.size(Gs,1), 2)]
        #Define complex miller index:
        evcC = evcR+ 1j*evcI
        self.Gs = Gs
        self.evcCs = evcC
        self.NG = NG
        
        #Check if the Gs contains the parity also:
        
        #Find a G that is not zero:
        if np.sum(Gs) == 0:
            self.isGamma = False
        else:
            self.isGamma = True
        
                
        self.isplanewave = True    
        return self
    
    def readfromdat(self, filename, iKS, cpA):
        """
        Reads the plane waves from a .dat file

        Args:
            filename (str): path to the .dat file containing the plane wave expansion of the wavefunction
            iKS (int): index of the KS state to be read from the .dat file
            cpA (array-like): cell parameter block

        """
        aA = cpA[0][0]
        bcrystal = 2*np.pi /aA * 1e10
        
        #Import the G and EVC array from the wavefunction file
        with open(filename, 'rb') as f:
            # Moves the cursor 4 bytes to the right
            f.seek(4)

            ik = np.fromfile(f, dtype='int32', count=1)[0]
            xk = np.fromfile(f, dtype='float64', count=3)
            ispin = np.fromfile(f, dtype='int32', count=1)[0]
            gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
            scalef = np.fromfile(f, dtype='float64', count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            ngw = np.fromfile(f, dtype='int32', count=1)[0]
            igwx = np.fromfile(f, dtype='int32', count=1)[0]
            npol = np.fromfile(f, dtype='int32', count=1)[0]
            nbnd = np.fromfile(f, dtype='int32', count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            b1 = np.fromfile(f, dtype='float64', count=3)
            b2 = np.fromfile(f, dtype='float64', count=3)
            b3 = np.fromfile(f, dtype='float64', count=3)

            f.seek(8,1)
            
            mill = np.fromfile(f, dtype='int32', count=3*igwx)
            mill = mill.reshape( (igwx, 3) ) 

            evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")

            f.seek(8,1)
            for i in range(nbnd):
                evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
                f.seek(8, 1)

        bcrystal = 2*np.pi /aA * 1e10
        Gs = bcrystal * mill.T

       
        self.Gs = Gs
        self.evcCs = evc[iKS-1,:]
        self.isGamma = True
        self.NG = Gs.shape[1]
        self.isplanewave = True

        return self
    

    def torgrid(self, rgrid, istart = 0, istep = 1, kparity=1):
        """
        Project the wavefunction from the plane wave basis onto a user specified position grid.
        
        Args:
            rgrid (array-like): 3 x N array of coordinates where the wavefunction needs to be evaluated.
            istart (int, optional): Starting index for plane wave summation. Defaults to 0.
            istep (int, optional): Step size for plane wave summation. Defaults to 1.
            kparity (int, optional): Parity in k-space, relevant for Gamma point. Defaults to 1.
        """
        xgrid = rgrid[0]
        psi = np.zeros(np.shape(xgrid),dtype = 'complex')
        
        #print(self.Gs)
        if self.isplanewave:
            """
            If the wavefunction already exists in the plane wave basis, 
            then just use the plane wave basis
            """
            #print("plane wave basis")
            if self.isGamma:
                psi = psi+ self.evcCs[0]*np.exp(1j*np.tensordot(self.Gs[:,0], rgrid, axes = (0,0)))

                for iG in range(1+istart,self.NG,istep):
                    psi = (psi+ self.evcCs[iG]*np.exp(1j*np.tensordot(self.Gs[:,iG], rgrid, axes = (0,0)))
                          +kparity*np.conj(self.evcCs[iG])*np.exp(-1j*np.tensordot(self.Gs[:,iG], rgrid, axes = (0,0))))
            else:
                for iG in range(istart, self.NG, istep):

                    psi = (psi+ self.evcCs[iG]*np.exp(1j*np.tensordot(self.Gs[:,iG], rgrid, axes = (0,0))))
        else:
            """
            Use directly the multipoles to create the radial part
            """
            #print(f"WFunction- torgrid-- directly evaluating from multipoles and radial functions")
            psi = self.MPField.torgrid(rgrid)
            
        return psi
    
    def kpsitorgrid(self, rgrid, istart = 0, istep = 1, dr = None):
        """
        Calculates k  operating on psi over a specific position grid. This is useful for calculating the momentum matrix elements.

        Args:
            rgrid (array-like): 3 x N array of coordinates where the wavefunction needs to be evaluated.
            istart (int, optional): Starting index for plane wave summation. Defaults to 0.
            istep (int, optional): Step size for plane wave summation. Defaults to 1.
            dr (float, optional): Finite difference step size for numerical differentiation when plane wave basis is not available. Defaults to None, in which case a small value of 1e-12 is used.
        
        """
        
        if self.isplanewave:
            """
            If the plane wave expansion already exists
            """
            #print("plane wave basis")
            kpsi = np.zeros(np.shape(rgrid),dtype = 'complex')
            if self.isGamma:
                for iG in range(1+istart,self.NG, istep):
                    for icomp in range(3):
                        kpsi[icomp] = kpsi[icomp] + self.Gs[icomp, iG]* self.evcCs[iG]*np.exp(1j*np.tensordot(self.Gs[:,iG], rgrid, axes = (0,0)))\
                        -self.Gs[icomp, iG]* np.conj(self.evcCs[iG])*np.exp(-1j*np.tensordot(self.Gs[:,iG], rgrid, axes = (0,0)))

            return kpsi

        else:
            """
            if the plane wave basis has not been defined and only the spherical harmonics with or without the Atomic wavefunctions are there
            ** Fully numeric evaluation of k psi
            
            """
            #print(f"WFunction- kpsitorgrid-- Full numerical evaluation")
            xgrid = rgrid[0]
            ygrid = rgrid[1]
            zgrid = rgrid[2]
            
            if dr == None:
                dx = dy = dz = 1e-12
            else:
                dx = dy = dz = dr
            
            rgridp = np.array([rgrid[0]+dx, rgrid[1], rgrid[2]])
            rgridm = np.array([rgrid[0]-dx, rgrid[1], rgrid[2]])
            
            delxpsir = (self.torgrid(rgridp)-self.torgrid(rgridm))/(2*dx)
            
            rgridp = np.array([rgrid[0], rgrid[1]+dy, rgrid[2]])
            rgridm = np.array([rgrid[0], rgrid[1]-dy, rgrid[2]])
            
            delypsir = (self.torgrid(rgridp)-self.torgrid(rgridm))/(2*dy)
            
            rgridp = np.array([rgrid[0], rgrid[1], rgrid[2]+dz])
            rgridm = np.array([rgrid[0], rgrid[1], rgrid[2]-dz])
            
            delzpsir = (self.torgrid(rgridp)-self.torgrid(rgridm))/(2*dz)
            
            
            kpsi = 1/1j * np.array([delxpsir, delypsir, delzpsir])
            return kpsi
                       
    def getharmonics(self, lmax):
        """
        Computes multipole expansion of the wavefunction using the plane wave expansion coefficients. This is done by using the identity that a plane wave can be expanded in terms of spherical harmonics and spherical Bessel functions. The resulting multipole expansion is stored in the MPField attribute of the wavefunction object.
        
        Args:
            lmax (int): maximum angular momentum quantum number for the multipole expansion
        """
        if self.Gs == None:
            logging.warning(f"Error in getharmonics- plane wave expansion not defined yet")
            return
        
        Gabsall = np.sqrt(self.Gs[0]**2 +self.Gs[1]**2 +self.Gs[2]**2)
        Gabs = np.unique(Gabsall)
        Nunq = np.size(Gabs)
        
        
        indices  = []
        for val in Gabs:
            indices.append(np.where(Gabsall == val)[0])
        
        
        MPs = []
        #Initialize the MPField of the wavefunction so that we do not accidentally double add things
        MPField = SField()
        for i in range(Nunq):
            MPField.add_component(Gabs[i], lmax,  {"type": "J"})
            mps = []
            for index in indices[i]:
                mp = MP_SA(lmax)
                
                if Gabs[i]==0:
                    theta = 0
                    phi = 0
                    mp.C[0][0]=np.conj(Y(0,0,theta, phi))*self.evcCs[0]
                    
                else:
                    G = self.Gs[:, index]
                    Gnorm = np.linalg.norm(G)
                    
                    #Evaluating polar coordingates
                    theta = np.arccos(G[2]/Gnorm)
                    temp = np.sqrt(G[0]**2 + G[1]**2)
                    
                    if not temp == 0:
                        phi = np.arccos(G[0]/temp) 
                    else:
                        phi = 0
                        
                    phi = phi -2*phi*(G[1]<0)
                    
                    
                    if not self.isGamma:
                        for l in range(0, lmax+1):
                            for m in range(-l,l+1):
                                mp.C[l][m] =  self.evcCs[index] * np.conj(Y(l,m,theta, phi))
                    else:
                        for l in range(0, lmax+1):
                            for m in range(-l,l+1):
                                mp.C[l][m] =self.evcCs[index] * np.conj(Y(l,m,theta, phi)) \
                                            + np.conj(self.evcCs[index]) * np.conj(Y(l,m,np.pi-theta, phi+np.pi))
                                

                
                mps.append(mp)
                
            MPField.mps[i] = addMP(mps)
            
            #MPs.append(addMP(mps))
            self.MPField = MPField
        return MPField
    
    def copy(self):
        """
        Deep copy Wavefunction
        """
        wfcopy = WFunction()
        
        wfcopy.isGamma = self.isGamma
        wfcopy.Gs = deepcopy(self.Gs)
        wfcopy.evcCs = deepcopy(self.evcCs)
        wfcopy.NG = self.NG
        wfcopy.MPField = self.MPField.copy()
        wfcopy.isplanewave = self.isplanewave
        
        return wfcopy
    
    def __add__(self, wf1):
        
        wfout = self.copy()
        
        if self.isplanewave:
            #This means plane wave basis:
            
            wfout.evcCs = self.evcCs + wf1.evcCs
        else:
            #This means only scalar field present
            wfout.MPField = self.MPField + wf1.MPField
            
        return wfout
    
    def scale(self, fact):
        
        wfout = self.copy()
        if self.isplanewave:
            #This means plane wave basis:
            
            wfout.evcCs = self.evcCs*fact
        else:
            #This means only scalar field present
            wfout.MPField = self.MPField.scale(fact)
            
        return wfout
        
    