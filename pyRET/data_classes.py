import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from .wavefunctions import *
import logging
#from .photoniccavity import CylindricalCavity

@dataclass
class WFunctions_data():

    """ 
    Dataclass for handling wavefunctions.
    
    Attributes:
        emitterTypes (list of string): list of emitter types.
        states (list of list of string): list of list of states.
        iKSs (list of list of int): list of list of KS indices for the states.
        rgridwf (list of arrays): list of rgridwfs- separate for each emitter.
        Ns (list of dict): list of dict of normalization constants for each state.
        pwout (list of string): list of pw.out texts correspinding to each emitter for book keeping.
        wfs (list of dict): list of dict of wfs (WFunction class object) for each state as the key. 
        kwfs (list of dict): list of dict of k operated on wfs (Vector Fields) for each state as the key. Can be empty
        wfrs (list of dict): list of dict of values of the wavefunctions over corresponding position grids.
        kwfrs (list of dict): list of dict of values of the k operated wavefunctions over corresponding position grids.
        
    """    

    
    def __init__(self, emitterTypes = None, states = None,
                 iKSs = None,
                 rgridwf = None,
                 Ns = None,
                 pwout = None,
                 wfs = None,
                 kwfs = None,
                 wfrs = None,
                 kwfrs = None):
        
        #Defined based on a planne wave basis
        if emitterTypes is None: 
            self.emitterTypes = []
        else:
            self.emitterTypes = emitterTypes
        
        if states is None: 
            self.states = []
        else:
            self.states = states
        
        if iKSs is None: 
            self.iKSs = []
        else:
            self.iKSs = iKSs
            
            
        if rgridwf is None: 
            self.rgridwf = []
        else:
            self.rgridwf = rgridwf
            
        if Ns is None: 
            self.Ns = []
        else:
            self.Ns = Ns

        if pwout is None: 
            self.pwout = []
        else:
            self.pwout = pwout
        
        if wfs is None: 
            self.wfs = []
        else:
            self.wfs = wfs
            
        if kwfs is None: 
            self.kwfs = []
        else:
            self.kwfs = kwfs
            
        if wfrs is None: 
            self.wfrs = []
        else:
            self.wfrs = wfrs
            
        if kwfrs is None: 
            self.kwfrs = []
        else:
            self.kwfrs = kwfrs
            
        return
    
    def Encode(self):
        """
        Jsonize the data
        """
        Encoded = {}
        
        Encoded.update({"Class": self.__class__.__name__})
        Encoded.update({"emitterTypes": self.emitterTypes})
        #print("emitters")
        
        Encoded.update({"states": self.states})
        #print("states")
        
        Encoded.update({"iKSs": self.iKSs})
        #print("iKSs")
        
        Encoded.update({"rgridwf": [rgrid.tolist() for rgrid in self.rgridwf]})
        #print("rgridwf")
        
        
        Ns = []
        for item in self.Ns:
            N = {}
            for key in item.keys():
                N.update({key: (np.real(item[key]), np.imag(item[key]))})
            Ns.append(N)
        Encoded.update({"Ns": Ns})
        #print("Ns")
        
        
        Encoded.update({"pwout" : [item for item in self.pwout]})
        #print("pwout")
        
        
        wfs = []
        for item in self.wfs:
            wf = {}
            for key in item.keys():
                wf.update({key: item[key].Encode()})
            wfs.append(wf)
        Encoded.update({"wfs": wfs})
        #print("wfs")
        
        kwfs = []
        for item in self.kwfs:
            kwf = {}
            for key in item.keys():
                kwf.update({key: item[key].Encode()})
            kwfs.append(kwf)
        Encoded.update({"kwfs": kwfs})
        #print("kwfs")
        
        wfrs = []
        for item in self.wfrs:
            wfr = {}
            for key in item.keys():
                wfr.update({key: (np.real(item[key]).tolist(), np.imag(item[key]).tolist())})
            wfrs.append(wfr)
        Encoded.update({"wfrs": wfrs})
        #print("wfrs")
        
        
        kwfrs = []
        for item in self.kwfrs:
            kwfr = {}
            for key in item.keys():
                kwfr.update({key: (np.real(item[key]).tolist(), np.imag(item[key]).tolist())})
            kwfrs.append(kwfr)
        Encoded.update({"kwfrs": kwfrs})
        #print("kwfrs")
        
        return Encoded
    
    def Decode(self, data):

        """
        Load data from a jsonized form.
        """

        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.emitterTypes = data["emitterTypes"]
        self.states = data["states"]
        self.iKSs = data["iKSs"]
        
        self.rgridwf = [np.array(item) for item in data["rgridwf"]]
        
        
        for item in data["Ns"]:
            N = {}
            for key in item.keys():
                
                N.update({key: item[key][0]+ 1j*item[key][1]})
            self.Ns.append(N)
        
        self.pwout = data["pwout"]
        
        
        
        for item in data["wfs"]:
            wf = {}
            for key in item.keys():
                wf.update({key: WFunction().Decode(item[key])})
            self.wfs.append(wf)
        
        for item in data["kwfs"]:
            kwf = {}
            for key in item.keys():
                kwf.update({key: WFunction().Decode(item[key])})
            self.kwfs.append(wf)
        
        for item in data["wfrs"]:
            wfr = {}
            for key in item.keys():
                wfr.update({key: np.array(item[key][0])+ 1j*np.array(item[key][1])})
            self.wfrs.append(wfr)
        
        for item in data["kwfrs"]:
            kwfr = {}
            for key in item.keys():
                kwfr.update({key: np.array(item[key][0])+ 1j*np.array(item[key][1])})
            self.kwfrs.append(kwfr)
        
        
        
        return self

    
@dataclass
class Positions_data():
    """ 
    Dataclass representing relative positions of the emitters. The attributes are dictionaries with the convention of first key = absorber id and second key = emitter id. 
    The values are arrays of the corresponding data.
    
    Attributes:
        Rs (dict of dict of numpy arrays): dictionary of dictionaries of radial positions of the emitters with respect to the absorber.
        Ts (dict of dict of numpy arrays): dictionary of dictionaries of polar angles of the emitters with respect to the absorber.
        Ps (dict of dict of numpy arrays): dictionary of dictionaries of azimuthal angles of the emitters with respect to the absorber.
        Rg (dict of dict of numpy arrays): R in grid format.
        Tg (dict of dict of numpy arrays): T in grid format.
        Pg (dict of dict of numpy arrays): P in grid format.
        Rvects (dict of dict of numpy arrays): each array has a shape of (3, Nr, Nt, Np) and represents the position vectors of the emitters with respect to the absorber in Cartesian coordinates over the position grid. The first index of the array is for the x, y, z components of the position vector.
        
    """    
    
    def __init__(self):
        
        
        self.Rs = {}  
        self.Ts = {}
        self.Ps = {}
        self.Rg = {}
        self.Tg = {}
        self.Pg = {}
        self.Rvects = {}
        
        return
    
    def Encode(self):

        """
        Jsonize the data
        """        

        Encoded = {}
        
        Encoded.update({"Class": self.__class__.__name__})
        
        Encoded.update({"Rs": {}})
        Encoded.update({"Ts": {}})
        Encoded.update({"Ps": {}})
        Encoded.update({"Rg": {}})
        Encoded.update({"Tg": {}})
        Encoded.update({"Pg": {}})
        Encoded.update({"Rvects": {}})
        for key1, item1 in self.Rs.items():
            Encoded["Rs"][key1] = {}
            Encoded["Ts"][key1] = {}
            Encoded["Ps"][key1] = {}
            Encoded["Rg"][key1] = {}
            Encoded["Tg"][key1] = {}
            Encoded["Pg"][key1] = {}
            Encoded["Rvects"][key1] = {}
                                        
            for key2, item2 in item1.items():
                Encoded["Rs"][key1][key2] = self.Rs[key1][key2].tolist()
                Encoded["Ts"][key1][key2] = self.Ts[key1][key2].tolist()
                Encoded["Ps"][key1][key2] = self.Ps[key1][key2].tolist()
                Encoded["Rg"][key1][key2] = self.Rg[key1][key2].tolist()
                Encoded["Tg"][key1][key2] = self.Tg[key1][key2].tolist()
                Encoded["Pg"][key1][key2] = self.Pg[key1][key2].tolist()
                Encoded["Rvects"][key1][key2] = self.Rvects[key1][key2].tolist()
                
    
        return Encoded

    
    def Decode(self, data):
        
        """
        Load data from a jsonized form.
        """
        
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
    
        for key1, item1 in data["Rs"].items():
            self.Rs[key1] = {}
            self.Ts[key1] = {}
            self.Ps[key1] = {}
            self.Rg[key1] = {}
            self.Tg[key1] = {}
            self.Pg[key1] = {}
            self.Rvects[key1] = {}
                                        
            for key2, item2 in item1.items():
                self.Rs[key1][key2] = np.array(data["Rs"][key1][key2])
                self.Ts[key1][key2] = np.array(data["Ts"][key1][key2])
                self.Ps[key1][key2] = np.array(data["Ps"][key1][key2])
                self.Rg[key1][key2] = np.array(data["Rg"][key1][key2])
                self.Tg[key1][key2] = np.array(data["Tg"][key1][key2])
                self.Pg[key1][key2] = np.array(data["Pg"][key1][key2])
                self.Rvects[key1][key2] = np.array(data["Rvects"][key1][key2])
                
                
        return self
    

@dataclass
class Positions_data_1config():
    """
    Dataclass representing relative positions of the emitters. This is similar to the Positions_data class but is for a single configuration of emitter-absorber positions.

    Attributes:
        Rs (numpy array): radial positions of the emitters with respect to the absorber.
        Ts (numpy array): polar angles of the emitters with respect to the absorber.
        Ps (numpy array): azimuthal angles of the emitters with respect to the absorber.
        Rvects (numpy array): position vectors of the emitters with respect to the absorber in Cartesian coordinates.
        
    """    
    
    def __init__(self):
        
        
        self.Rs = np.array([])    
        self.Ts = np.array([])
        self.Ps = np.array([])
        self.Rvects = np.array([])
        
        return
    
    def Encode(self):
        
        """
        Jsonize the data
        """

        Encoded = {}
        
        Encoded.update({"Class": self.__class__.__name__})
        
        Encoded.update({"Rs": self.Rs.tolist()})
        Encoded.update({"Ts": self.Ts.tolist()})
        Encoded.update({"Ps": self.Ps.tolist()})
        Encoded.update({"Rvects": self.Rvects.tolist()})
        
        return Encoded

    
    def Decode(self, data):
        """
        Load data from a jsonized form.
        """
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"Wrong decoder function. The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        
        self.Rs =  np.array(data["Rs"])
        self.Ts =  np.array(data["Ts"])
        self.Ps =  np.array(data["Ps"])
        self.Rvects =  np.array(data["Rvects"])
        
        
                
        return self    
    
    
@dataclass
class Emitters_data():
    """ 
    Dataclass representing types and corresponding information on the emitters.

    Attributes:
        EmType(dict): dictionary with Emitter IDs as keys and emitter types as strings.
        T1(dict): dictionary with Emitter IDs as keys and T1 values.
        w(dict): dictionary with Emitter IDs as keys and transition frequencies (rad/s).
        k(dict): dictionary with Emitter IDs as keys and wave numbers (1/m).
        nIndex(dict): dictionary with Emitter IDs as keys and refractive indices of the medium the emitter sits in.
        pMPs(dict): dictionary with Emitter IDs as keys-- predefined multipole moments (if any) of the emitters.        
    """    
    
    def __init__(self):
        self.EmType = {}
        self.T1 = {}    
        self.w = {}
        self.k = {}
        self.nIndex = {}
        self.pMPs = {}
        
        return
    
    def Encode(self):
        
        """
        Jsonize the data
        """

        Encoded = {}
        
        Encoded.update({"Class": self.__class__.__name__})
        Encoded.update({"nIndex": self.nIndex})
        Encoded.update({"EmType": self.EmType})
        
        T1 = {}
        for em, item1 in self.T1.items():
            T1[em] = {}
            for transition, item2 in item1.items():
                T1[em][transition] = [np.real(item2),np.imag(item2)]
        
        Encoded.update({"T1": T1})
        
        w = {}
        for em, item1 in self.w.items():
            w[em] = {}
            for transition, item2 in item1.items():
                w[em][transition] = [np.real(item2),np.imag(item2)]
            
        Encoded.update({"w": w})
        
        
        k = {}
        for em, item1 in self.k.items():
            k[em] = {}
            for transition, item2 in item1.items():
                k[em][transition] = [np.real(item2),np.imag(item2)]
            
        Encoded.update({"k": k})
        
        
        pMPs = {}
        for em, item1 in self.pMPs.items():
            #print(f"{em=}")
            pMPs[em] = {}
            for transition, item2 in item1.items():
                #print(f"{transition=}")
                pMPs[em][transition] = {}
                for L, item3 in item2.items():
                    #print(f"{L=}")
                    pMPs[em][transition][f"{L=}"] = {}
                    for P, item4 in item3.items():
                        #print(f"{P=}")
                        pMPs[em][transition][f"{L=}"][f"{P=}"] = \
                        [np.real(self.pMPs[em][transition][L][P]).tolist(),
                         np.imag(self.pMPs[em][transition][L][P]).tolist()]
                
        Encoded.update({"pMPs": pMPs })
    
        return Encoded
    
    def Decode(self, data):

        """
        Load data from a jsonized form.
        """

        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
        
        self.nIndex = data["nIndex"] 
        self.EmType = data["EmType"]
        
        T1 = {}
        for em, item1 in data["T1"].items():
            #print(f"{em=}")
            T1[em] = {}
            for transition, item2 in item1.items():
                T1[em][transition] = item2[0] + 1j*item2[1]
        self.T1 = T1
        
        w = {}
        for em, item1 in data["w"].items():
            #print(f"{em=}")
            w[em] = {}
            for transition, item2 in item1.items():
                w[em][transition] = item2[0] + 1j*item2[1]
        self.w = w
        
        k = {}
        for em, item1 in data["k"].items():
            #print(f"{em=}")
            k[em] = {}
            for transition, item2 in item1.items():
                k[em][transition] = item2[0] + 1j*item2[1]
        self.k = k
        
        
        
        pMPs = {}
        for em, item1 in data["pMPs"].items():
            #print(f"{em=}")
            pMPs[em] = {}
            for transition, item2 in item1.items():
                #print(f"{transition=}")
                pMPs[em][transition] = {}
                for strL, item3 in item2.items():
                    #print(f"{L=}")
                    pMPs[em][transition][int(strL.split("=")[1])] = {}
                    for strP, item4 in item3.items():
                        #print(f"{P=}")
                        pMPs[em][transition][int(strL.split("=")[1])][int(strP.split("=")[1])] = \
                        np.array(item4[0])+ 1j*np.array(item4[1])
                        
        self.pMPs = pMPs
        
        return self
 
    
@dataclass
class V_data():
    """
    Dataclass containing matrix elements computed in spherical multipolar basis.
    
    Attributes:
        ws(dict): dictionary with absorber IDs as keys and values as dictionaries with emitter IDs as keys and complex numbers as values.
        V(dict): nested dictionary of the format V[absorber_id][emitter_id][absorber_transition][L][M][P] = complex ND Array
            absorber_transition is of the format "'initial_state'-to-'final_state'"
            L, M, P are the multipole order, multipole component and parity of the transition respectively.
    """    
    
    def __init__(self):
        
        self.ws = {}
        self.V = {}    
        
        return
    
    def Encode(self):
        """
        Jsonize the data
        """

        Encoded = {}
        
        Encoded.update({"Class": self.__class__.__name__})
        
        ws = {}
        for absorber, item1 in self.ws.items():
            ws[absorber] = {}
            for emitter, item2 in item1.items():
                ws[absorber][emitter] = [np.real(item2).tolist(), np.imag(item2).tolist()]
        Encoded.update({"ws": ws})
        
        
        
        V = {}
        for absorber, item1 in self.V.items():
            V[absorber] = {}
            for emitter, item2 in item1.items():
                V[absorber][emitter] = {}
                for transition, item3 in item2.items():
                    V[absorber][emitter][transition] = {}
                    for L, item4 in item3.items():
                        V[absorber][emitter][transition][f"{L=}"] = {}
                        for M, item5 in item4.items():
                            V[absorber][emitter][transition][f"{L=}"][f"{M=}"] = {}
                            for P, item6 in item5.items():
                                V[absorber][emitter][transition][f"{L=}"][f"{M=}"][f"{P=}"] = \
                                [np.real(item6).tolist(), np.imag(item6).tolist()]
        
        Encoded.update({"V": V})
        
        return Encoded

    def Decode(self, data):
        """
        Load data from a jsonized form.
        """
        if not data["Class"] == self.__class__.__name__:
            clsname = data["Class"]
            logging.warning(clsname)
            logging.warning( self.__class__.__name__)
            logging.warning(f"The json objeect was from {clsname} class, but is being decoded by"+
            f"the decoder of the class {self.__class__.__name__}")
            
        ws = {}
        for absorber, item1 in data["ws"].items():
            ws[absorber] = {}
            for emitter, item2 in item1.items():
                ws[absorber][emitter] = np.array(item2[0])+ 1j*np.array(item2[1])
        self.ws = ws
        
        V = {}
        for absorber, item1 in data["V"].items():
            V[absorber] = {}
            for emitter, item2 in item1.items():
                V[absorber][emitter] = {}
                for transition, item3 in item2.items():
                    V[absorber][emitter][transition] = {}
                    for strL, item4 in item3.items():
                        L = int(strL.split("=")[1])
                        V[absorber][emitter][transition][L] = {}
                        for strM, item5 in item4.items():
                            M = int(strM.split("=")[1])
                            V[absorber][emitter][transition][L][M] = {}
                            for strP, item6 in item5.items():
                                P = int(strP.split("=")[1])
                                V[absorber][emitter][transition][L][M][P] = \
                                np.array(item6[0])+1j*np.array(item6[1])
        
        self.V = V
        return self
        
       