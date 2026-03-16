#Parser program to parse wavefunctions and V calculation input files.

import numpy as np
import os
import json

# Wavefunction calculation for atomic states
class WFin_atomic():
    """ 
    Class to handle wavefunction calculation input file for atomic wavefunction

    Attributes:
        EmType(str): unique label identifying the emitter.
        Z(float): nuclear charge of the atom
        abya0(float): ratio of the characteristic bohr radius of the wavefunction to 1 bohr.
        Nxyz(int): number of points in the x,y,z directions for the wavefunction grid.
        states(list of str): list of state names for the wavefunction calculation.
        ns(list of int): list of principal quantum numbers for the states in the wavefunction calculation.
        Ls(list of list of int): list of lists of orbital quantum numbers for the states in the wavefunction calculation.
        Ms(list of list of int): list of lists of magnetic quantum numbers for the states in the wavefunction calculation.
        Ws(list of list of complex): List of complex weight values.
        savepath(str): path to save the wavefunction data file.
        savefile(str): name of the wavefunction data file to save.
    """
    
    
    def __init__(self, EmType = "A", Z = None, abya0 = None, Nxyz = None, 
                 savepath = "./", savefile = "WFdata.json"):
        self.EmType = EmType
        self.Z = Z
        self.abya0 = abya0
        self.Nxyz = Nxyz
        self.states = []
        self.ns = []
        self.Ls = []
        self.Ms = []
        self.Ws = []
        self.savepath = savepath
        self.savefile = savefile
        return 
    
    
    def add_state(self, statename, n, Ls, Ms, Ws):
        self.states.append(statename)
        self.ns.append(n)
        self.Ls.append(Ls)
        self.Ms.append(Ms)
        self.Ws.append(Ws)
        
        return self
    
    
    def write_file(self, filename):
        
        strout = (f"Atomic Wavefunction Calculation: \n\n"+
                 f"EmType   =   {self.EmType} \n"+
                 f"a/z   =   {self.abya0} \n"+
                 f"Nxyz = {self.Nxyz} \n"+
                 f"savepath = {self.savepath}\n"+
                 f"savefile = {self.savefile}\n\n")
        
        for state, n, L, M, W in zip(self.states, self.ns, self.Ls, self.Ms, self.Ws):
            strout+=f"State Name = {state} \n"
            strout+=f"n = {n} \n"
            strout+=f"Z = {self.Z} \n"
            strout+=f"Ls = "
            for Lval in L:
                strout+=f"{Lval}, "
            strout+=f"\n"
            
            strout+=f"Ms = "
            for Mval in M:
                strout+=f"{Mval}, "
            strout+=f"\n"
            
            strout+=f"Ws = "
            for Wval in W:
                strout+=f"{Wval}, "
            strout+=f"\n"
        
        
        
            
        with open(filename, "w") as f:
            f.write(strout)
            
            
        return self

# Wavefunction calclation for DFT KS states    
class WFin_QE():
    """
    Class to handle wavefunction calculation input file- from quantum espresso output
    
    Attributes:
        EmType(str): unique label identifying the emitter.
        aA(float): cell size in Angstrom
        Nxyz(int): number of points in the x,y,z directions for the wavefunction grid.
        wfloadfile(str): path to the file to load the wavefunction from.
        qeoutfile(str): path to the quantum espresso output file.
        states(list of str): list of state names for the wavefunction calculation (used as labels).
        iKSs(list of list of int): list of lists of Kohn-Sham state indices for the states in the wavefunction calculation.
        Ws(list of list of complex): List of complex weight values.
        savepath(str): path to save the wavefunction data file.
        savefile(str): name of the wavefunction data file to save.
        centering(bool): whether to apply centering to the wavefunction grid.
    """
    
    
    def __init__(self, EmType = "A",aA = None, Nxyz = None, wfloadfile = None, qeoutfile = None, 
                 savepath = "./", savefile = "WFdata.json", centering = False):
        self.EmType = EmType
        self.aA = aA
        self.Nxyz = Nxyz
        self.wfloadfile = wfloadfile
        self.qeoutfile = qeoutfile
        self.centering = centering
        self.states = []
        self.iKSs = []
        self.Ws = []
        
        self.savepath = savepath
        self.savefile = savefile
        
        return 
    
    
    def add_state(self, statename, iKSs, Ws):
        self.states.append(statename)
        self.iKSs.append(iKSs)
        self.Ws.append(Ws)
        
        return self
    
    
    def write_file(self, filename):
        
        strout = (f"Quantum Espresso Wavefunction Calculation: \n\n"+
                 f"EmType   =   {self.EmType} \n"+
                 f"aA   =   {self.aA} \n"+
                 f"Nxyz = {self.Nxyz} \n"+
                 f"wfloadfile = {self.wfloadfile} \n"+
                 f"qeoutfile = {self.qeoutfile} \n"+
                 f"savepath = {self.savepath}\n"+
                 f"savefile = {self.savefile}\n\n")
        
        for state, iKS, W in zip(self.states, self.iKSs, self.Ws):
            strout+=f"State Name = {state} \n"
            strout+=f"\t iKSs = "
            for iKSval in iKS:
                strout+=f"{iKSval}, "
            strout+=f"\n"
            
            strout+=f"\t Ws = "
            for Wval in W:
                strout+=f"{Wval}, "
            strout+=f"\n"
            
            
            
        with open(filename, "w") as f:
            f.write(strout)
            
            
        return self
    
#Function to parse input file
def wfreadinput(filename):
    with open(filename,"r") as f:
        lines = [line.rstrip() for line in f]
    
    if "Atomic Wavefunction" in lines[0]:
        #Atomic wavefunction
        wfin = WFin_atomic()
        
        states = []
        ns = []
        Ls = []
        Ms = []
        Ws = []

        iterobj = iter(lines)

        while True:
            try:
                line = next(iterobj)
                #print("line= ", line )
            except StopIteration:
                break

            if "EmType" in line:
                wfin.EmType = line.split("=")[1].strip()
            if "a/z" in line:
                wfin.abya0 = float(line.split("=")[1])



            if "Nxyz" in line:
                wfin.Nxyz = int(line.split("=")[1])
                
            if "savepath" in line:
                wfin.savepath = line.split("=")[1].strip()
            
            if "savefile" in line:
                wfin.savefile = line.split("=")[1].strip()
            
            if "centering" in line:
                wfin.centering = bool(line.split("=")[1].strip())

            if "State Name" in line:
                states.append(line.split("=")[1].strip())

                ns.append(int( next(iterobj).split("=")[1]))
                wfin.Z = float(next(iterobj).split("=")[1])

                L = next(iterobj).split("=")[1].split(",")[:-1]
                #print("L= ", L)
                L = [int(item) for item in L]
                Ls.append(L)
                
                M = next(iterobj).split("=")[1].split(",")[:-1]
                #print("M= ", M)
                M = [int(item) for item in M]
                Ms.append(M)
                
                W = next(iterobj).split("=")[1].split(",")[:-1]
                #print("W= ",W)
                W = [complex(item) for item in W]
                Ws.append(W)
        
        wfin.states = states
        wfin.ns = ns
        wfin.Ls = Ls
        wfin.Ms = Ms
        wfin.Ws = Ws
    
    
    if "Quantum Espresso Wavefunction" in lines[0]:
        #Wavefunction from quantum espresso
        wfin = WFin_QE()
        states = []
        
        iKSs = []
        Ws = []
        
        iterobj = iter(lines)

        while True:
            try:
                line = next(iterobj)
                #print("line= ", line )
            except StopIteration:
                break

            if "EmType" in line:
                wfin.EmType = line.split("=")[1].strip()
            if "aA" in line:
                wfin.aA = float(line.split("=")[1])

            if "Nxyz" in line:
                wfin.Nxyz = int(line.split("=")[1])
            
            if "wfloadfile" in line:
                wfin.wfloadfile = line.split("=")[1].strip()
            
            if "qeoutfile" in line:
                wfin.qeoutfile = line.split("=")[1].strip()
            
            if "savepath" in line:
                wfin.savepath = line.split("=")[1].strip()
            
            if "savefile" in line:
                wfin.savefile = line.split("=")[1].strip()
            
                

            if "State Name" in line:
                states.append(line.split("=")[1].strip())

                iKS = next(iterobj).split("=")[1].split(",")[:-1]
                #print("L= ", L)
                iKS = [int(item) for item in iKS]
                iKSs.append(iKS)
                
                W = next(iterobj).split("=")[1].split(",")[:-1]
                #print("W= ",W)
                W = [complex(item) for item in W]
                Ws.append(W)
        

            if "centering" in line:
                wfin.centering = bool(line.split("=")[1].strip())
            
        wfin.states = states
        wfin.iKSs = iKSs
        wfin.Ws = Ws
        
        
    return wfin        

# Class to represent input for matrix element calculation    
class Vin():
    
    """
    Class to handle the input configurations for the calculations of the matrix elements

    Attributes:
        emdatafile(str): path to the json containing emitter data.
        posdatafile(str): path to the json containing position data.
        wfdatafiles(list of str): list of paths to the json files containing wavefunction data.
        absorber_id(str): unique label identifying the absorber.
        emitter_id(str): unique label identifying the source.
        absorber_transitions(list of str): list of unique labels identifying the transitions in the absorber for which to calculate V.
                These transitions are in the form "initial_state_to_final_state".
        nIndex(float): refractive index of the medium.
        Lmax(int): maximum multipole order to consider in the V calculation.
        ws(list of complex): list of frequency values to calculate V at.
        savepath(str): path to save the V data file.
        savefile(str): name of the V data file to save.
        radtype_abs(str): radial type of the multipolar modes. "J" for bessel J functions and "H1" for type 1 Hankel functions.
        Hint(str): type of interaction Hamilonian to consider in the V calculation. "Adotp" for A dot p interaction and "E.r" for E dot r interaction.
    """
        
    def __init__(
            self,
            emdatafile = None,
            posdatafile = None,
            wfdatafiles = None,
            absorber_id = None,
            emitter_id = None,
            absorber_transitions = None,
            nIndex = 1.,
            Lmax = 3,
            ws = None,
            savepath = "./",
            savefile = "V.json",
            radtype_abs = "H1",
            Hint = "Adotp",
    ):
        self.emdatafile = emdatafile
        self.posdatafile = posdatafile
        self.wfdatafiles = wfdatafiles
        self.absorber_id = absorber_id
        self.absorber_transitions = absorber_transitions
        self.emitter_id = emitter_id
        
        self.nIndex = nIndex
        self.Lmax = Lmax
        self.ws = ws
        
        self.savepath = savepath
        self.savefile = savefile
        self.radtype_abs = radtype_abs
        self.Hint = Hint

        return 
    
    
    def add_trans(self, new_transitions):
        
        self.absorber_transitions = self.absorber_transitions + new_transitions 
        
        return self
            
            
    def write_file(self, filename):
        """
        Write a input file for V computation
        """
        strabstrans = " , ".join(self.absorber_transitions) + ","
        strout = (f"Computation of V_ijkLMP: \n\n"+
                 f"Emitters Data File   =   {self.emdatafile} \n"+
                 f"Positions Data File   =   {self.posdatafile} \n"+
                 f"Absorber = {self.absorber_id} \n"+
                 f"Emitter = {self.emitter_id} \n"+
                 f"Absorber Transitions = {strabstrans} \n" +
                 f"Radial type of multipole = {self.radtype_abs}\n")
        
        if not self.Hint == "Adotp":
            strout+= f"Interaction = {self.Hint} \n"
        
        strout+= f"Wavefunction Data Files = \n"
        for item in self.wfdatafiles:
            strout+= f"\t {item} \n"
        strout+= "\n\n"
        
        strout+=f"Frequency Values = "
        for wval in self.ws:
            strout+=f"{wval}, "
        strout+=f"\n"
        
        strout+= (f"nIndex = {self.nIndex} \n"+
                 f"Lmax = {self.Lmax: } \n"+
                 f"savepath = {self.savepath}\n"+
                 f"savefile = {self.savefile}\n\n")
        
        
        with open(filename, "w") as f:
            f.write(strout)
            
            
        return self
           
#Function to parse input file for V calculation           
def vreadinput(filename):
    #print(f"Trying to open {filename} from directory {os.getcwd()}")
    with open(filename,"r") as f:
        lines = [line.rstrip() for line in f]
    vin = Vin()
  
    iterobj = iter(lines)

    while True:
        try:
            line = next(iterobj)
            #print("line= ", line )
        except StopIteration:
            break
        
        if "Radial type of multipole" in line:
            vin.radtype_abs = line.split("=")[1].strip()
        if "Emitters Data File" in line:
            vin.emdatafile = line.split("=")[1].strip()
        if "Positions Data File" in line:
            vin.posdatafile = line.split("=")[1].strip()
        if "Absorber" in line and "Transitions" not in line:
            vin.absorber_id = line.split("=")[1].strip()
        if "Emitter" in line:
            vin.emitter_id = line.split("=")[1].strip()
        
        if "Absorber Transitions" in line:
            vin.absorber_transitions =\
            [item.strip() for item in line.split("=")[1].split(",")[:-1]]

        if "Interaction" in line:
            vin.Hint = line.split("=")[1].strip()
        
        if "Wavefunction Data Files" in line:
            vin.wfdatafiles = []
            while True:
                temp = next(iterobj).strip()
                if temp == "":
                    break
                else:
                    vin.wfdatafiles.append(temp)
                    
        if "Frequency Values" in line:            
            temp = line.split("=")[1].split(",")[:-1]
            #print("W= ",W)
            vin.ws = [complex(item) for item in temp]

        if "nIndex" in line:
            vin.nIndex = float(line.split("=")[1])
        if "Lmax" in line:
            vin.Lmax = int(line.split("=")[1])
        
        if "savepath" in line:
            vin.savepath = line.split("=")[1].strip()
        if "savefile" in line:
            vin.savefile = line.split("=")[1].strip()
        

    return vin
    
                
            
        
        
        
        
        
        
        
        