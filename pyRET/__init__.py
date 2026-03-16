from .constants import Consts
from .utils import *
from .multipole_scalar_angular import *
from .multipole_vector_angular import *
from .fields import *
from .wavefunctions import *
from .emfieldtools import *
from .data_classes import *
from .Vtools import *
from .WFtools import *
from .inputparser import *
from .pyjobtools import *

__version__ = "v0.0"


def print_logo():
    print(r"   //                                      \\   ")
    print(r"  // ______                       ______    \\  ")
    print(r" || |      |`                    |      |`   || ")
    print(r" || |   S  | |  -_--_--_--_- >   |   A  | |  || ")
    print(r" || |______| |                   |______| |  || ")
    print(r"  \\ `-------`                    `-------` //  ")
    print(r"   \\   pyRET-- First Principles RET       //   ")

    return


def header():
    #from mpi4py import MPI
    #if not MPI.COMM_WORLD.Get_rank() == 0:
    #    return


    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0
        
    """Prints welcome header."""
    import datetime

    if rank == 0:

        print_logo()

        print("version          : ", __version__)
        print("Today            : ", datetime.datetime.today())
    
    return

header()



