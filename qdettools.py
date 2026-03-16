#%%
import numpy as np
import os, sys, json

from westpy.qdet import QDETResult
from westpy.qdet import visualize_correlated_state

from pyscf.fci import cistring
from copy import deepcopy



def print_solution_info(solution,effective_hamiltonian):
    
    print("Printing solution information")
    print("-----------------------------")
    #print(solution.keys())
    print(f"Number of states = {solution.get('nstates')}")
    print(f"Number of electrons = {solution.get('nelec')}")
    print(f"Number of orbitals = {solution.get('norb')}")

    for i, state in enumerate(solution['evcs']):
        en = solution['evs'][i]
        ss, mult = effective_hamiltonian.heff.spin_square_spin_polarized(state, norb=solution['norb'], nelec=solution['nelec'])
        print(f"i={i}, energy [eV]={en:.3f}, <S^2>={ss:.3f}, mult={mult:.3f}")
    print("\n\n")
    
    return

def visualize_states(solution):
    print("Printing states")
    print("---------------")
    
    for i, state in enumerate(solution['evcs']):
        
        en = solution['evs'][i]
        print(f'i= {i}, en [eV]={en}')
        print(visualize_correlated_state(state, solution['norb'], solution['nelec'], cutoff=10 **(-2)))
        print("\n")
    return

def tabulate_transitions(solutions, iKSs, initial_state, final_state, cutoff_amplitude = 1e-2):
    
    
    solution_initial = solutions[initial_state[0]]
    index_state_initial = initial_state[1]
    norb_initial = solution_initial["norb"]
    nelec_initial = solution_initial["nelec"]
    determinants_alpha_initial = cistring.make_strings(range(norb_initial), nelec_initial[0])
    determinants_beta_initial = cistring.make_strings(range(norb_initial), nelec_initial[1])
    Energy_initial = solution_initial["evs"][index_state_initial]
    state_initial = solution_initial["evcs"][index_state_initial]
    state_string_initial = visualize_correlated_state(state_initial, norb_initial, nelec_initial, cutoff=10 **(-2))
    print("Initial State: ")
    print(state_string_initial)
    print("\n")
    
    solution_final = solutions[final_state[0]]
    index_state_final = final_state[1]
    norb_final = solution_final["norb"]
    nelec_final = solution_final["nelec"]
    determinants_alpha_final = cistring.make_strings(range(norb_final), nelec_final[0])
    determinants_beta_final = cistring.make_strings(range(norb_final), nelec_final[1])
    Energy_final = solution_final["evs"][index_state_final]
    state_final = solution_final["evcs"][index_state_final]
    state_string_final = visualize_correlated_state(state_final, norb_final, nelec_final, cutoff=10 **(-2))
    print("Final State: ")
    print(state_string_final)
    print("\n")
    
    
    #Now tabulating transitions
    transitions = []
    print("Listing Allowed Transitons:")
    for ialpha_i, phialpha_i in enumerate(determinants_alpha_initial):
        for ibeta_i, phibeta_i in enumerate(determinants_beta_initial):
            for ialpha_f, phialpha_f in enumerate(determinants_alpha_final):
                for ibeta_f, phibeta_f in enumerate(determinants_beta_final):
                    
                    amplitude = state_initial[ialpha_i, ibeta_i] * state_final[ialpha_f, ibeta_f]
                    
                    if np.abs(amplitude) < cutoff_amplitude: continue
    
                    _tot = bin(phialpha_i^phialpha_f).count("1") + bin(phibeta_i^phibeta_f).count("1")
                    if not _tot == 2: continue #More than a single eletron tasntition is needed, not included.
                    
                    print(f"{phialpha_i :06b},{phibeta_i :06b} -- > {phialpha_f :06b},{phibeta_f :06b}")
    
    
                    #Determine which KS state
                    if bin(phialpha_i^phialpha_f).count("1") ==2:
                        print(f"\tSpin conserving transition within alpha channel")
                        binary_representation = bin(phialpha_i^phialpha_f)[2:]  # Remove the '0b' prefix
                        positions = [i for i, bit in enumerate(reversed(binary_representation)) if bit == '1']
                        print(f"\tTransition from {iKSs[positions[0]]} to {iKSs[positions[1]]} with amplitude={amplitude:.3e}")
                        transitions.append({"amplitude": amplitude,
                                                  "iKS1": min(iKSs[positions]),
                                                  "iKS2": max(iKSs[positions]),
                                                  "isspinflip": False})
                            
                    if bin(phibeta_i^phibeta_f).count("1") ==2:
                        print(f"\tSpin conserving transition within beta channel")
                        binary_representation = bin(phibeta_i^phibeta_f)[2:]  # Remove the '0b' prefix
                        positions = [i for i, bit in enumerate(reversed(binary_representation)) if bit == '1']
                        print(f"\tTransition from {iKSs[positions[0]]} to {iKSs[positions[1]]} with amplitude={amplitude:.3e}")
                        transitions.append({"amplitude": amplitude,
                                                  "iKS1": min(iKSs[positions]),
                                                  "iKS2": max(iKSs[positions]),
                                                  "isspinflip": False})
                            
                    if bin(phibeta_i^phibeta_f).count("1") ==1:
                        print(f"\tSpin flip transition- between alpha and beta channel")
                        binary_representation = bin(phibeta_i^phibeta_f)[2:]  # Remove the '0b' prefix
                        positions_beta = [i for i, bit in enumerate(reversed(binary_representation)) if bit == '1']
                        binary_representation = bin(phialpha_i^phialpha_f)[2:]  # Remove the '0b' prefix
                        positions_alpha = [i for i, bit in enumerate(reversed(binary_representation)) if bit == '1']
                        positions = positions_alpha + positions_beta
                        print(f"\tTransition from {iKSs[positions[0]]} to {iKSs[positions[1]]} with amplitude={amplitude:.3e}")
                        
                        transitions.append({"amplitude": amplitude,
                                                  "iKS1": min(iKSs[positions]),
                                                  "iKS2": max(iKSs[positions]),
                                                  "isspinflip": True})
                            
    
    return transitions


