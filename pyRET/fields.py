import numpy as np
from scipy import special
from copy import deepcopy
from .constants import Consts
from .multipole_scalar_angular import *
from .multipole_vector_angular import *
from .utils import *
import logging


###################################
# Vector Calculus on Fields
###################################

# Gradient: SField -> VField
def gradient(S):

    """ 
    Computes gradient of a scalar field. Only applicable for rfun{"order"} = None or l; and rfun["type"] = "J" or "H1".
    
    Args:
        S (SField): Scalar field
    
    Returns:
        V (VField): Gradient of the scalar field
        
    """
    
    V = VField(S.origin)
    
    
    
    for k, mp, rfun in zip(S.ks, S.mps, S.rfuns):
        #For each entry, check which class the multipole belongs to
        #and add the fields accordingly
        if rfun.get("type") == "Atomic":
            logging.warning(f"Atomic wavefunction encountered-- gradient not evaluated. ")
            continue
        
        V.add_component(k, mp.lmax, rfun)
        mpv = V.mpvs[-1]
        for l, value in mp.C.items():
            for  m, value1 in value.items():
                if l ==0:
                    mpv.C[0][1][m] = -1j*k*np.sqrt((l+1)/(2*l+1))*mp.C[l][m]
                else:
                    mpv.C[l][l+1][m] = -1j*k*np.sqrt((l+1)/(2*l+1))*mp.C[l][m]
                    mpv.C[l][l-1][m] =  1j*k*np.sqrt((l)/(2*l+1))*mp.C[l][m]
                    

    return V

# Curl: VField -> VField
def curl(V):
    """ 
    Computes curl of a vector field. Only applicable for rfun{"order"} = None or l; and rfun["type"] = "J" or "H1".

    
    Args:
        V (VField): Vector field
    
    Returns:
        Vc (VField): Curl of the vector field
    
    """

    Vc = VField(V.origin)    
    
    
    for k, mp, rfun in zip(V.ks, V.mpvs, V.rfuns):
        #For each entry, check which class the multipole belongs to
        #and add the fields accordingly
        
        Vc.add_component(k, mp.jmax, rfun)
        mpc = Vc.mpvs[-1]
        
        if isinstance(mp, MP_VA_Hansen):
            logging.warning("Calculating curl of the Hansen fields may result in errors.")
            mp = mp.toMP_VA()
            
        for j, value in mp.C.items():
            for  l, value1 in value.items():
                for M, value2 in value1.items():
                    if not j==0:
                        mpc.C[j][j][M] = -k*np.sqrt((j+1)/(2*j+1)) * mp.C[j][j-1][M] \
                                        - k*np.sqrt((j)/(2*j+1)) * mp.C[j][j+1][M]
                        
                        mpc.C[j][j+1][M] = - k*np.sqrt((j)/(2*j+1)) * mp.C[j][j][M]
                        mpc.C[j][j-1][M] = - k*np.sqrt((j+1)/(2*j+1)) * mp.C[j][j][M]
                        
    return Vc

# Divergence: VField -> SField
def divergence(V):
    """ 
    Computes divergence of a vector field. Only applicable for rfun{"order"} = None or l; and rfun["type"] = "J" or "H1".
    
    Args:
        V (VField): Vector field
    
    Returns:
        S (SField): Divergence of the vector field
    
    """

    S = SField(V.origin)
    for k, mp, rfun in zip(V.ks, V.mpvs, V.rfuns):
        #For each entry, check which class the multipole belongs to
        #and add the fields accordingly
        S.add_component(k, mp.jmax, rfun)
        mps = S.mps[-1]
        
        if isinstance(mp, MP_VA_Hansen):
            logging.warning("Computing divergence for the Hansen fields may result errors.")
            mp = mp.toMP_VA()
        
        for j, value in mp.C.items():
            for  l, value1 in value.items():
                for M, value2 in value1.items():
                    if l == j+1:
                        mps.C[j][M] = mps.C[j][M] -1j*k*np.sqrt((j+1)/(2*j+1))*mp.C[j][l][M]
                    elif l == j-1:
                        mps.C[j][M] = mps.C[j][M] +1j*k*np.sqrt((j)/(2*j+1))*mp.C[j][l][M]
        
        
    return S

# Scalar field class
class SField():
    """ 
    Class representing a Scalar Multipolar Field. This class contains a multipolar scalar field that is stored as a sum of
    various multipolar components, each with specific value of wavevectors and radial functions. The multipolar (angular) components are stored as MP_SA objects

    Attributes:
        origin(np.ndarray of shape (3,)): coordinates of the center.
        ks(list of float): list of radial wavevectors corresponding to different components
        mps(list of MP_SA): list of multipoles (MP_SA)
        rfuns(list of dict): list of dictionary containing information on the radial function. 
            must contain the key "type" = "J" or "H1" or "Atomic"
            optional keys: "order", "n","Z" etc etc.
          
    """

    
    def __init__(self, origin = np.array([0., 0., 0.])):

        self.origin = origin
        self.ks = [] #k amplitudes in 2pi/Angstrom unit
        self.mps = []#multipole vectors belonging to MP_SA class
        self.rfuns = [] # type J or H 
        
        return
  
    def Encode(self):
        """
        Encodes the SField object to a jsonizable dictionary. The multipoles are encoded using their own Encode() method.
        """
        
        Encoded = {}
        Encoded.update({"Class": self.__class__.__name__})
        Encoded.update({"origin": self.origin.tolist()})
        Encoded.update({"ks": self.ks})
        Encoded.update({"mps": [mp.Encode() for mp in self.mps]})
        Encoded.update({"rfuns": self.rfuns})
        return Encoded

    def Decode(self, data):
        """
        Decodes the SField object from a json dictionary.
        """
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.origin =np.array(data["origin"])
        self.ks = data["ks"]
        self.rfuns = data["rfuns"]
        for mp in data["mps"]:
            self.mps.append(MP_SA(mp["lmax"]).Decode(mp))
                
        
        return self
    
    def __add__(self, S1):
        
        """
        Adds two scalar fields together. Only valid when the origin is the same for both these objects.
        """
        #This method should return a new scalar field that is addition of the two fields:
        Sout = SField()
        Sout.origin = self.origin
        Sout.ks = []
        Sout.mps = []
        Sout.rfuns = []


        #We need to identify all the k values present in the assembly and then adding up the multipole 
        #coefficients if the rfuns also match up. If not, create two entries:

        kunq = np.sort(list(set(self.ks+S1.ks)))


        for k in kunq:

            indicesout = []

            #Search first field and add the elements directly 
            indexS0,  = np.where(np.array(self.ks)==k)
            if not np.size(indexS0) == 0:
                for i in indexS0:
                    Sout.ks.append(k)
                    Sout.rfuns.append(self.rfuns[i])
                    Sout.mps.append(self.mps[i])
                    indicesout.append(len(Sout.ks)-1)


                #Now, go through the second field and add the new elements where needed:
                indexS1,  = np.where(np.array(S1.ks)==k)
                if not np.size(indexS1) == 0:
                    for i in indexS1:
                        for iout in indicesout:
                            if Sout.rfuns[iout] == S1.rfuns[i]:
                                Sout.mps[iout] = addMP([Sout.mps[iout], S1.mps[i]])
                                break
                        else:
                            Sout.ks.append(k)
                            Sout.rfuns.append(S1.rfuns[i])
                            Sout.mps.append(S1.mps[i])

            else:
                indexS1,  = np.where(np.array(S1.ks)==k)
                for i in indexS1:
                    Sout.ks.append(k)
                    Sout.rfuns.append(S1.rfuns[i])
                    Sout.mps.append(S1.mps[i])


        return Sout
    
    def add_component(self, kval, lmax, rfun):
        
        """
        Adds a component- one multipole mode with specific rfunc and specific value of the wavevector with specificed combination of multipoles
        
        Args:
            kval(float): value of the wavevector in 2pi/Angstrom unit
            lmax(int): maximum order of the multipole
            rfun(dict): dictionary containing information on the radial function
        """
        self.ks.append(kval)
        self.mps.append(MP_SA(lmax))
        self.rfuns.append(rfun)
        
        return
    
    def torgrid(self, rgrid):
        """
        Project the scalar field in a position grid.

        Args:
            rgrid (np.ndarray of shape (3, N1, N2, ..)): position grid where the field needs to be evaluated. 
        
        Returns:
            S (np.ndarray of shape (N1, N2, ..)): scalar field evaluated at the positions specified by rgrid.
        """
        
        S = np.zeros(np.shape(rgrid[0]),dtype = 'complex')
        r, theta, phi = cart2pol(rgrid, self.origin)
        maxr = np.amax(r)
        for k, mp, rfun in zip(self.ks, self.mps, self.rfuns):
            #For each entry, check which class the multipole belongs to
            #and add the fields accordingly
            
            
            for l, value in mp.C.items():
                for  m, value1 in value.items():
                    if mp.C[l][m]==0:
                        continue
                        
                    #Evaluating the spherical part
                    ys = Y(l,m, theta, phi)
                    
                    
                    #Evaluating the radial part
                    if rfun.get("order") is not None:
                        lr = rfun.get("order")
                    else:
                        lr = l
                    
                    if maxr*k<0.01 and rfun.get("type") == "J":
                        gr = gnear(lr, k*r, rfun)
                    else:
                        gr = g(lr, k*r, rfun)
                    """if rfun.get("type") == "J":
                        gr = J_sph(lr, k*r)
                    elif rfun.get("type") == "H1":
                        gr = Hankel1_sph(lr, k*r)
                    else:
                        continue"""
                        #gr = np.zeros(np.shape(r))
                    S = S+ mp.C[l][m]*gr*ys
                    
        return S
    
    
    def scale(self,factor):
        """
        Scalar multiplication.
        """
        for mp in self.mps:
            mp.scale(factor)
        return self
    
    def copy(self):
        Scopy = SField(self.origin)
        Scopy.ks = deepcopy(self.ks)     
        Scopy.mps = [mp.copy() for mp in self.mps]
        Scopy.rfuns = deepcopy(self.rfuns)
        
        return Scopy
        
# Vector field class
class VField():
    """ 
    Class representing a Vector Field. This class contains a multipolar vector field that is stored as a sum of
    various multipolar components, each with specific value of wavevectors and radial functions.
    The multipolar (angular) components are stored as MP_VA or MP_VA_Hansen objects

    Attributes:
        origin(np.ndarray of shape (3,)): origin of the field
        ks(list of float): list of radial wavevectors corresponding to different components
        mpvs(list of MP_VA or MP_VA_Hansen): list of multipoles (MP_VA or MP_VA_Hansen)
        rfuns(list of dict): list of dictionary containing information on the radial function. 
            must contain the key "type" = "J" or "H1" or "Atomic"
            optional keys: "order", "n","Z" etc etc.
               
    """
    
    def __init__(self, origin = np.array([0., 0., 0.])):
        
        self.origin = origin
        self.ks = [] #k amplitudes in 2pi/Angstrom unit
        self.mpvs = []#multipole vectors belonging to either MP_VA or MP_VA_Hansen class
        self.rfuns = [] # type J or H 
        
        return
    
    def Encode(self):
        """
        Encodes the VField object to a jsonizable dictionary. The multipoles are encoded using their own Encode() method.
        """
        
        Encoded = {}
        Encoded.update({"Class": self.__class__.__name__})
        Encoded.update({"origin": self.origin.tolist()})
        Encoded.update({"ks": self.ks})
        Encoded.update({"mpvs": [mpv.Encode() for mpv in self.mpvs]})
        Encoded.update({"rfuns": self.rfuns})
        return Encoded
    
    def Decode(self, data):
        """
        Decodes the VField object from a jsonizable dictionary. The multipoles are decoded using their own Decode() method.
        """
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.origin =np.array(data["origin"])
        self.ks = data["ks"]
        self.rfuns = data["rfuns"]
        for mpv in data["mpvs"]:
            clsname = mpv["Class"]
            if clsname == "MP_VA":
                self.mpvs.append(MP_VA(mpv["jmax"]).Decode(mpv))
            elif clsname == "MP_VA_Hansen":
                self.mpvs.append(MP_VA_Hansen(mpv["jmax"]).Decode(mpv))
            else:
                logging.warning(f"Inappropriate decoder for object of type {clsname}")
        
        return self
    
    def __add__(self, V1):
        """
        Adds two vector fields together
        Only valid when the origin is the same for both these objects.
        """
        #This method should return a new scalar field that is addition of the two fields:
        Vout = VField()
        Vout.origin = self.origin
        Vout.ks = []
        Vout.mpvs = []
        Vout.rfuns = []


        #We need to identify all the k values present in the assembly and then adding up the multipole 
        #coefficients if the rfuns also match up. If not, create two entries:

        kunq = np.sort(list(set(self.ks+V1.ks)))




        for k in kunq:

            indicesout = []

            #Search first field and add the elements directly 
            indexV0,  = np.where(np.array(self.ks)==k)
            if not np.size(indexV0) == 0:
                for i in indexV0:
                    Vout.ks.append(k)
                    Vout.rfuns.append(self.rfuns[i])
                    Vout.mpvs.append(self.mpvs[i])
                    indicesout.append(len(Vout.ks)-1)


                #Now, go through the second field and add the new elements where needed:
                indexV1,  = np.where(np.array(V1.ks)==k)
                if not np.size(indexV1) == 0:
                    for i in indexV1:
                        for iout in indicesout:
                            if Vout.rfuns[iout] == V1.rfuns[i]:
                                Vout.mpvs[iout] = addMP_VA([Vout.mpvs[iout], V1.mpvs[i]])
                                break
                        else:
                            Vout.ks.append(k)
                            Vout.rfuns.append(V1.rfuns[i])
                            Vout.mpvs.append(V1.mpvs[i])

            else:
                indexV1,  = np.where(np.array(V1.ks)==k)
                for i in indexV1:
                    Vout.ks.append(k)
                    Vout.rfuns.append(V1.rfuns[i])
                    Vout.mpvs.append(V1.mpvs[i])


        return Vout
 
    def add_component(self, kval, jmax, rfun, isHansen = False):
        """
        Add a component. One multipole mode with specific rfunc and specific value of the wavevector with specificed combination of multipoles. The multipoles are initialized to zero and need to be filled in separately.

        Args:
            kval(float): value of the wavevector in 2pi/Angstrom unit
            jmax(int): maximum order of the multipole
            rfun(dict): dictionary containing information on the radial function
                must contain the key "type" = "J" or "H1" or "Atomic"
                optional keys: "order", "n","Z" etc etc.
                isHansen(bool): whether the multipole is Hansen type or not. If True, the multipole will be initialized as MP_VA_Hansen, else it will be initialized as MP_VA.
        """
        self.ks.append(kval)
        if isHansen:
            self.mpvs.append(MP_VA_Hansen(jmax))
            
        else:
            self.mpvs.append(MP_VA(jmax))
            
        self.rfuns.append(rfun)
        
        return
    
    def torgrid(self, rgrid):
        """
        Project the vector field in a position grid.

        Args:
            rgrid (np.ndarray of shape (3, N1, N2, ..)): position grid where the field needs to be evaluated. 
        
        Returns:
            V (np.ndarray of shape (3, N1, N2, ..)): vector field evaluated at the positions specified by rgrid.
        """
        
        V = np.zeros(np.shape(rgrid),dtype = 'complex')
        r, theta, phi = cart2pol(rgrid, self.origin)
        maxr = np.amax(r)
        for k, mp, rfun in zip(self.ks, self.mpvs, self.rfuns):
            #For each entry, check which class the multipole belongs to
            #and add the fields accordingly
            
            if isinstance(mp, MP_VA):
                for j, value in mp.C.items():
                    for  l, value1 in value.items():
                        for M, value2 in value1.items():
                            
                            if mp.C[j][l][M] == 0:
                                continue
                            
                            
                            #Evaluating the spherical part
                            yv = Yvect(j, l, M, theta, phi)
                            
                            #Evaluating the radial part
                            if rfun.get("order") is not None:
                                lr = rfun.get("order")
                            else:
                                lr = l
                            
                            if maxr*k<0.01 and rfun.get("type") == "J":
                                gr = gnear(lr, k*r, rfun)
                            else:
                                gr = g(lr, k*r, rfun)
                            """if rfun.get("type") == "J":
                                gr = J_sph(lr, k*r)
                            elif rfun.get("type") == "H1":
                                gr = Hankel1_sph(lr, k*r)
                            else:
                                continue"""
                                #gr = np.zeros(np.shape(r))
                            
                            #Overall Vector field:
                            V = V+ mp.C[j][l][M]* np.array([gr*yv[0], gr*yv[1], gr*yv[2]])
                            
                            
                            
            elif isinstance(mp, MP_VA_Hansen):
                for j, value in mp.C.items():
                    for  M, value1 in value.items():
                        for lam, value2 in value1.items():
                            
                            if mp.C[j][M][lam] == 0:
                                continue
                            
                            #Evaluating the spherical part
                            yv = Yvect_Hansen(j, M, lam, theta, phi)
                            
                            #Evaluating the radial part
                            if rfun.get("order") is not None:
                                lr = rfun.get("order")
                            else:
                                lr = j
                            
                            if maxr*k<0.01 and rfun.get("type") == "J":
                                gr = gnear(lr, k*r, rfun)
                            else:
                                gr = g(lr, k*r, rfun)
                                
                            """if rfun.get("type") == "J":
                                gr = J_sph(lr, k*r)
                            elif rfun.get("type") == "H1":
                                gr = Hankel1_sph(lr, k*r)
                            else:
                                continue"""
                                #gr = np.zeros(np.shape(r))
                            
                            #Overall Vector field:
                            V = V+ mp.C[j][M][lam]*np.array([gr*yv[0], gr*yv[1], gr*yv[2]])

        
            
        
        return V
         
    def scale(self,factor):
        """
        Scalar multiplication.
        """
        
        for mpv in self.mpvs:
            mpv.scale(factor)
        return self
    
    def Farfield(self, theta, phi):
        """
        Calculates the far field distribution of the vector field on a theta and phi grid.

        Args:
            theta (np.ndarray of shape (N1, N2, ...)): polar angle grid
            phi (np.ndarray of shape (N1, N2, ...)): azimuthal angle grid

        Returns:
            V (np.ndarray of shape (3, N1, N2, ..)): far field vector field evaluated at the angles specified by theta and phi.

        """
        V = np.array([np.zeros(np.shape(theta),dtype = 'complex_'),np.zeros(np.shape(theta),dtype = 'complex_'),np.zeros(np.shape(theta),dtype = 'complex_')])
        
        for k, mp, rfun in zip(self.ks, self.mpvs, self.rfuns):
            #For each entry, check which class the multipole belongs to
            #and add the fields accordingly
            
            if isinstance(mp, MP_VA):
                for j, value in mp.C.items():
                    for  l, value1 in value.items():
                        for M, value2 in value1.items():
                            
                            #Evaluating the spherical part
                            yv = Yvect(j, l, M, theta, phi)
                            
                            #Evaluating the radial part at the far field
                            if rfun.get("order") is not None:
                                lr = rfun.get("order")
                            else:
                                lr = l
                            
                            gr = gfar(lr, k, rfun)
                            
                            """if rfun.get("type") == "J":
                                gr = J_sph(lr, k*r)
                            elif rfun.get("type") == "H1":
                                gr = Hankel1_sph(lr, k*r)
                            else:
                                continue"""
                                #gr = np.zeros(np.shape(r))
                            
                            #Overall Vector field:
                            V = V+ mp.C[j][l][M]* np.array([gr*yv[0], gr*yv[1], gr*yv[2]])

        return V

    def copy(self):

        Vcopy = VField(self.origin)
        Vcopy.ks = deepcopy(self.ks)     
        Vcopy.mpvs = [mpv.copy() for mpv in self.mpvs] 
        Vcopy.rfuns = deepcopy(self.rfuns)
        
        return Vcopy
    
    
