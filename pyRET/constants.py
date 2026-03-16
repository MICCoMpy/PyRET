import numpy as np


class Consts():
    
    def __init__(self):
        self.c = 299792458.0
        self.mu0 = 4*np.pi*1e-7
        self.eps0 = 1/(self.mu0* self.c**2)
        
        self.hplank = 6.62607015e-34
        self.hbar = self.hplank/(2*np.pi)
        self.qe = 1.60217663e-19
        self.me = 9.1093837e-31
        self.abohr = self.eps0 * self.hplank**2 /(np.pi* self.qe**2  * self.me)
        self.kB = 1.380649e-23

        self.H2eV = 27.211386245988
        self.Ry2eV = 0.5*self.H2eV
        self.au2A = 5.29177210903e-1
        return
    