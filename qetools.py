# Code for handling Quantum Espresso input and output files, and for creating structures for QE calculations.
import numpy as np
from dataclasses import dataclass
from .constants import Consts
#from mp_api.client import MPRester
from pymatgen.core.structure import Structure

#Quantum Espresso Structure
class QE_Structure():
    """Class that contains a pymatgen structure object
    and methods to generate the structure-related text blocks for the QE input file.
    """
    def __init__(
            self,
            structure: Structure = None,
            label: str = None,
            pseudopotentials: dict = None,
            Zs: dict = None,
            ):
        
        self.structure = structure
        self.label = label
        self.pseudopotentials = pseudopotentials if pseudopotentials is not None else {}
        
        self.Zs = Zs if Zs is not None else {}
        
        return
    
    # def retrieve_from_materials_project(
    #         self,
    #         material_id: str,
    #         isconventional: bool = True,
    #         api_key = None,
    # ):
    #     """
    #     Retrieve a structure from Materials Project using its material_id
    #     """
    #     if api_key is None:
    #         print("No API key provided. aborting.")
    #         return self

    #     with MPRester(api_key) as mpr:
    #         struct = mpr.get_structure_by_material_id(material_id)
    #         if isconventional:
    #             struct = struct.to_conventional()
    #         self.structure = struct
    #         self.label = material_id
    #     self.Zs = {element.name: float(element.atomic_mass) for element in set(self.structure.species)}
    #     return self
    
    def toXYZ(
            self,    
    ):
        """
        Returns a list of strings 
        corresponding to the xyz format in Angstrom unit
        """

        xyztext = []
        for isite, element in enumerate(self.structure.species):

            xyztext.append(
                f"{element.name} {self.structure.cart_coords[isite, 0]:.8f} {self.structure.cart_coords[isite, 1]:.8f} {self.structure.cart_coords[isite, 2]:.8f} \n"
            )
        
        return xyztext

    def generate_atomic_species_text(
            self,
            ):
        """
        Generate the atomic species text for the QE input file.
        """
        atomic_species_text = []
        atomic_species_text.append("ATOMIC_SPECIES\n")
        for element in set(self.structure.species):
            
            name = element.name
            Z = self.Zs.get(name)
            
            pseudo = self.pseudopotentials.get(name, None)
            atomic_species_text.append(f"{name} {Z} {pseudo} \n")
                
        return atomic_species_text
    
    def generate_atomic_positions_text(
            self,       
    ):
        """
        Generate the atomic positions text for the QE input file.
        """

        atomic_positions_text = []
        atomic_positions_text.append("ATOMIC_POSITIONS angstrom \n")
        
        for isite, element in enumerate(self.structure.species):
            coords = self.structure.cart_coords[isite]
            atomic_positions_text.append(
                f"{element.name} {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f} \n"
            )
        
        return atomic_positions_text

    def generate_cell_parameters_text(self):

        cell_parameters_text = []
        cell_parameters_text.append("CELL_PARAMETERS angstrom \n")
        
        for i in range(3):
            cell_parameters_text.append(
                f"{self.structure.lattice.matrix[i, 0]:.8f} {self.structure.lattice.matrix[i, 1]:.8f} {self.structure.lattice.matrix[i, 2]:.8f} \n"
            )
        
        return cell_parameters_text


    def writeXYZ(
            self,
            filepath,
    ):
        
        with open(filepath, "w") as f:
            f.write(f"{len(self.structure)}\n")
            f.write(f"{self.label}\n")
            for line in self.toXYZ():
                f.write(line)
        
        return

    def loadfromXYZ(
            self,
            filepath,
            lattice_vectors = None,
    ):
        
        
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        num_atoms = int(lines[0])
        self.label = lines[1].strip()
        
        species = []
        coords = []
        for i in range(2, 2 + num_atoms):
            line = lines[i].strip().split()
            species.append(line[0])
            coords.append(np.array([float(x) for x in line[1:4]]))



        self.structure = Structure(
            lattice = lattice_vectors if lattice_vectors is not None else None,
            species = species,
            coords = coords,
            coords_are_cartesian = True,
        )


        return self

    def read_from_infile(self, filename):
    
        with open(filename, "r") as f:
            lines = f.readlines()
        
        #Find the ATOMIC_POSITIONS block
        atomic_positions_start = lines.index("ATOMIC_POSITIONS angstrom \n") + 1
        atomic_positions_end = lines.index("\n", atomic_positions_start)
        atomic_positions_lines = lines[atomic_positions_start:atomic_positions_end]
        
        species = []
        coords = []
        for line in atomic_positions_lines:
            parts = line.split()
            species.append(parts[0])
            coords.append(np.array([float(x) for x in parts[1:4]]))
        
        
        #Find the CELL_PARAMETERS block
        cell_parameters_start = lines.index("CELL_PARAMETERS angstrom \n") + 1
        cell_parameters_end = lines.index("\n", cell_parameters_start)
        cell_parameters_lines = lines[cell_parameters_start:cell_parameters_end]
        lattice_vectors = []
        for line in cell_parameters_lines:
            parts = line.split()
            lattice_vectors.append(np.array([float(x) for x in parts]))
        lattice_vectors = np.array(lattice_vectors)

        
        
        self.structure = Structure(
            lattice = lattice_vectors,
            species = species,
            coords = coords,
            coords_are_cartesian = True,
        )

        #Find the ATOMIC_SPECIES block
        atomic_species_start = lines.index("ATOMIC_SPECIES\n") + 1
        atomic_species_end = lines.index("\n", atomic_species_start)
        atomic_species_lines = lines[atomic_species_start:atomic_species_end]
        self.pseudopotentials = {}
        for line in atomic_species_lines:
            parts = line.split()
            if len(parts) == 3:
                name = parts[0]
                Z = float(parts[1])
                pseudo = parts[2]
                self.pseudopotentials[name] = pseudo
                self.Zs[name] = Z



        return self

class QE_kpoints():
    """
    Class to create the kpoints for quantum espresso input:
    """
    def __init__(self,mode,nk,*argv):
        self.mode = mode
        if mode == "automatic":
            self.nk = nk
            self.kshift = argv[0]
        elif mode == "gamma":
            self.nk =1
            self.kshift = 0
        elif mode == "crystal_b":
        #in this case nk is the number of symmetry points in the k-path
        #argv passes: #kpath_point_labels (e.g. ['\\Gamma', 'X', 'L'])
                      #kpoints: the k-vectors of the symmetry points (generated from pymagen)
                      #dk (step in the k-value along the symmetry path) in the units of 2pi/a
                      #vspuerA (supercell cell parameters)
                      #and aA    
            pass
        elif mode == "crystal":
            
            self.nk = nk
            self.kpoints = argv[0]
            self.weights = argv[1]
            
            pass
        else:
            pass
        
    def gentext(self):
        block=[]
        if self.mode == "automatic":
            block.append("K_POINTS automatic \n")
            
            block.append(f"{self.nk}  {self.nk} {self.nk} "+
                         f"{self.kshift} {self.kshift} {self.kshift}" + "\n" )
            
        elif self.mode == "gamma":
            block.append("K_POINTS gamma \n")
        elif self.mode == "crystal":
            block.append("K_POINTS crystal \n")
            block.append(f"{self.nk} \n")
            for ik in range(self.nk):
                kval = self.kpoints[ik]
                wt = self.weights[ik]
                block.append(f"{kval[0]} {kval[1]} {kval[2]} {wt} \n")
            
        else:
            pass
        return block
    
    def readfromfile(self, filename):
        with open(filename,"r") as f:
            lines = [line for line in f]
        
        for iline, line in enumerate(lines):
            if "K_POINTS" in line:
                if "gamma" in line:
                    self.mode = "gamma"
                    self.nk = 1
                    self.kshift = 0
                elif "automatic" in line:
                    vals = lines[iline+1].strip().split()
                    self.mode = "automatic"
                    self.nk = int(vals[0])
                    self.kshift = int(vals[3])
        return self

class PW_Input():
    """
    Handling the Quantum Espresso Input files:
    """
    def __init__(self, QEStructure):
        
        self.STRUCTURE = QEStructure
        self.CONTROL = {
            "prefix": "prefix",
            "calculation": "scf",
            "outdir": "./out/",
            "pseudo_dir":"./",
            "restart_mode": "from_scratch",
            }   
        
        ntyp = len(set(QEStructure.structure.species))
        nat = len(QEStructure.structure.species)

        self.SYSTEM = {
            "input_dft": "PBE", 
            "ibrav": 0,
            "ntyp": ntyp,
            "nat": nat,
            "ecutwfc": 60.0,
            "ecutrho": 720.0,
            "nspin": 2,
            "nbnd": 400,
            }
        
        self.ELECTRONS = {
            "mixing_beta": 0.7,
            "conv_thr":"1d-8",
            }
        
        self.IONS = {
            "ion_dynamics":"bfgs",
            }
        
        self.CELL = {
            "cell_dynamics": "bfgs",
            "press": 0.0,
            "press_conv_thr": 0.5,
        }

        self.KPOINTS = QE_kpoints("gamma",1)

        return

    def write_to_file(self, filename):
        """
        Write the Quantum Espresso input file.
        """
        lines_to_write = []
        #Write CONTROL namelist
        lines_to_write.append("&CONTROL\n")
        for key, value in self.CONTROL.items():
            lines_to_write.append(f"  {key} = '{value}',\n")
        lines_to_write.append("/\n")

        #Write SYSTEM namelist
        lines_to_write.append("&SYSTEM\n")
        for key, value in self.SYSTEM.items():
            if isinstance(value, str):
                lines_to_write.append(f"  {key} = '{value}'\n")
            else:
                lines_to_write.append(f"  {key} = {value}\n")
        lines_to_write.append("/\n")

        #Write ELECTRONS namelist
        lines_to_write.append("&ELECTRONS\n")
        for key, value in self.ELECTRONS.items():
            lines_to_write.append(f"  {key} = {value},\n")
        lines_to_write.append("/\n")

        #Write IONS namelist
        lines_to_write.append("&IONS\n")
        for key, value in self.IONS.items():
            lines_to_write.append(f"  {key} = '{value}',\n")
        lines_to_write.append("/\n")

        #Write CELL namelist
        lines_to_write.append("&CELL\n")
        for key, value in self.CELL.items():
            if isinstance(value, str):
                lines_to_write.append(f"  {key} = '{value}',\n")
            else:
                lines_to_write.append(f"  {key} = {value},\n")
        lines_to_write.append("/\n")
        
        #Write ATOMIC_SPECIES
        lines_to_write += \
            self.STRUCTURE.generate_atomic_species_text() + ["\n"]
        
        #Write ATOMIC_POSITIONS
        lines_to_write += \
            self.STRUCTURE.generate_atomic_positions_text() + ["\n"]
        
        #Write K_POINTS
        lines_to_write += \
            self.KPOINTS.gentext() + ["\n"]

        #Write CELL_PARAMETERS
        lines_to_write += \
            self.STRUCTURE.generate_cell_parameters_text() + ["\n"]

        #Write HUBBARD if exists
        if hasattr(self, "HUBBARD"):
            lines_to_write.append("HUBBARD (ortho-atomic)\n")
            for line in self.HUBBARD:
                lines_to_write.append(f"{line}\n")
            lines_to_write.append("\n")
        
        #Write the final text to the file
        with open(filename, "w") as f:
            f.writelines(lines_to_write)
        return
    

    def read_from_file(self, filename):
        """
        Read the Quantum Espresso input file.
        """
        
        #Read kpoints first
        self.KPOINTS = QE_kpoints("gamma",1)
        self.KPOINTS.readfromfile(filename)


        #Read the rest of the file
        with open(filename, "r") as f:
            lines = f.readlines()
        
        #Parse CONTROL namelist
        control_start = lines.index("&CONTROL\n") + 1
        control_end = lines.index("/\n", control_start)
        control_lines = lines[control_start:control_end]
        for line in control_lines:
            #Everything is string
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = value.strip().rstrip(",")
                self.CONTROL[key] = value.strip("'\"")
        
        #Parse SYSTEM namelist
        system_start = lines.index("&SYSTEM\n") + 1
        system_end = lines.index("/\n", system_start)
        system_lines = lines[system_start:system_end]
        for line in system_lines:
            #Everything except input_dft is a number
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = value.strip().rstrip(",")
                if key == "input_dft":
                    self.SYSTEM[key] = value.strip("'\"")
                else:
                    #Convert to float if possible
                    if "." in value:
                        self.SYSTEM[key] = float(value)
                    else:
                        self.SYSTEM[key] = int(value)
        #Parse ELECTRONS namelist
        electrons_start = lines.index("&ELECTRONS\n") + 1
        electrons_end = lines.index("/\n", electrons_start)
        electrons_lines = lines[electrons_start:electrons_end]
        for line in electrons_lines:
            if "mixing_beta" in line:
                self.ELECTRONS["mixing_beta"] = float(line.split("=")[1].strip().rstrip(","))
            elif "conv_thr" in line:
                self.ELECTRONS["conv_thr"] = line.split("=")[1].strip().rstrip(",")
        
        #Parse IONS namelist
        ions_start = lines.index("&IONS\n") + 1
        self.IONS["ion_dynamics"] = lines[ions_start].split("=")[1].strip().rstrip(",").strip("'\"")
        #Parse CELL namelist
        cell_start = lines.index("&CELL\n") + 1
        cell_end = lines.index("/\n", cell_start)
        cell_lines = lines[cell_start:cell_end]
        for line in cell_lines:
            if "cell_dynamics" in line:
                self.CELL["cell_dynamics"] = line.split("=")[1].strip().rstrip(",").strip("'\"")
            elif "press" in line:
                self.CELL["press"] = float(line.split("=")[1].strip().rstrip(","))
            elif "press_conv_thr" in line:
                self.CELL["press_conv_thr"] = float(line.split("=")[1].strip().rstrip(","))
        
        return self

class PW_Output():
    
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "r") as f:
            self.lines = f.readlines()
        
        return
    
    def attach_input_file(self, input_filepath):
        """
        Attach the input file to the output object for further analysis.
        """
        self.input_filepath = input_filepath
        QEStructure = QE_Structure().read_from_infile(input_filepath)
        self.input = PW_Input(QEStructure).read_from_file(input_filepath)
        
        return self


    def get_status(self):
        

        #If "JOB DONE." is found, the job was terminated successfully
        for line in self.lines[::-1]:
            if "JOB DONE." in line:
                self.finished_in_time = True
                break
        
        else:
            self.finished_in_time = False
        

        #If "convergence NOT achieved" is found, the SCF did not converge
        for line in self.lines:
            if "convergence NOT achieved" in line:
                self.scf_converged = False
                break
        else:
            self.scf_converged = True


        return self


    def parse_relax_calculation(self, verbose = False):
        """
        parse results of a relax calculation
        """

        #If input file is attached, check if calculation type is relax or vc-relax
        if hasattr(self, "input"):
            calc_type = self.input.CONTROL.get("calculation", "scf")
            if calc_type not in ["relax", "vc-relax"]:
                raise ValueError("Calculation type is not relax or vc-relax.")
            elif verbose:
                print(f"Parsing relax calculation of type: {calc_type}")
        
        #Parse final structure
        if verbose: print("Parsing final structure...")
        
        #Read alat
        if verbose: print("Reading lattice parameter (alat)...")
        for line in self.lines:
            if "lattice parameter (alat)" in line:
                self.alat_au = float(line.split("=")[1].strip().split()[0])
                break
        self.alat_A = self.alat_au * Consts().au2A  
        if verbose: print(f"alat (Bohr) = {self.alat_au}")
        if verbose: print(f"alat (Angstrom) = {self.alat_A}")

        


        #Read the cell parameters
        if verbose: print("Reading cell parameters...")
        cell_start = None
        for iline, line in enumerate(self.lines):
            if "crystal axes: (cart. coord. in units of alat)" in line:
                cell_start = iline + 1
                break
    
        if cell_start is not None:
            lattice_vectors = []
            for i in range(3):
                parts = self.lines[cell_start + i].strip().split()
                vec = [float(x) * self.alat_A for x in parts[3:6]]
                lattice_vectors.append(vec)
            self.cell_parameters_A = np.array(lattice_vectors)
        if verbose: print(f"Cell parameters (Angstrom):\n{self.cell_parameters_A}")

        #Read atomic positions
        species = []
        coords = []
        final_positions_start = None
        for iline, line in enumerate(self.lines):
            if "Begin final coordinates" in line:
                if verbose: print(f"Found beginning of final coordinates block.{iline=}")
                final_positions_start = iline + 1
                break
        
        if final_positions_start is None:
            print("Structure not found in output file.")
            return self
        
       
        for line in self.lines[final_positions_start:]:
            if verbose: print(f"Parsing line: {line.strip()}")
            if "End final coordinates" in line:
                break
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            species.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])

        #define the pymatgen structure
        self.final_structure = Structure(
            lattice = self.cell_parameters_A,
            species = species,
            coords = coords,
            coords_are_cartesian = True,
        )

        return self
    
