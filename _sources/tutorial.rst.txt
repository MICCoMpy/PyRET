.. _tutorial:

Tutorial
========

Introduction
------------
In this page we discuss the basic approach implemented in **pyRET** and discuss a few examples of how to use the various packages.

Let us say that we have to compute the photon mediated near field energy transfer between a source (labeled as S) and the absorber (labeled as A). In the dilute limit, the isolated emitters comprises discrete energy levels with corresponding energy eigenstates- ground state :math:`\mid \Phi_{GS}^{(S/A)} \rangle` and the excited state :math:`\mid \Phi_{ES}^{(S/A)} \rangle`. These states are assumed to be multireference in general, and thus can be expressed as a linear combination of single Slater determinants. For example,

.. math::
    :label: multireference_state
    
    \langle \mid \Phi_{GS}^{(S)} \rangle = \sum_{\substack{j \in \mathrm{occ} \\ i \in \mathrm{unocc}}} \alpha_{ij}^{(GS)} c_j^{(S/A)\dagger} c_i^{(S)} \langle \mid D^{(S)} \rangle

and similar expressions would exist for the other states. Here :math:`\langle \mid D^{(S)} \rangle` is a single Slater determinant built from filling out the first N orbitals in the single electron picture. For details, please see Appendix C and D of Ref. [1].

Further, we aim to compute the matrix element corresponding to transitions between the multireference states with the interaction Hamiltonian :math:`H_{int}` [1].

.. math::
    :label: Hint

    H_{int} = \sum_{E = S, A} \sum_{i=1}^{N_E}\left[-\frac{e\hat{\textbf{p}}_i\cdot\hat{\textbf{A}}}{2m_0}-\frac{e\hat{\textbf{A}}\cdot\hat{\textbf{p}}_i}{2m_0}+e\hat{A}_0 +g\frac{e\hbar}{2m_0}\hat{\bf{\sigma}}_i\cdot\bar\nabla\times\hat{\textbf{A}}\right].

We employ the Slater-Condon rule to reduce the matrix element between Slater determinant to matrix element between orbitals. For example,

.. math::
    :label: slater_condon_example

    \langle \Phi_{ES}^{(S)} \mid H_{\mathrm{int}} \mid \Phi_{GS}^{(S)}, 1_{k,\alpha s} \rangle
    =\sum_{\substack{j \in \mathrm{occ} \\ i \in \mathrm{unocc}}}
    \alpha^{*}_{ij}
    \langle \phi^{(S)}_{j} \mid H_{\mathrm{int}} \mid \phi^{(S)}_{i}, 1_{k,\alpha s} \rangle

The single particle orbitals here are :math:`\left|\phi^{(S)}_{i}\right\rangle`. This code provides an example of how we extract and store these single particle wavefunctions calculated using the Quantum Espresso Density Functional Theory code.

The matrix element corresponding to the photon emission process to a photon mode of energy :math:`\hbar c k` and :math:`\alpha` denoting all other degrees of freedom (e.g. in this case orbital angular momentum L, total z-projected angular momentum M, and parity P) can be expressed as,

.. math::
    :label: emission_matrix_element

    v_{k,\alpha}^{(S)} \sqrt{\Delta k} = \langle \Phi_{GS}^{(S)}, 1_{k,\alpha} \mid H_{\mathrm{int}} \mid \Phi_{ES}^{(S)} \rangle

and the matrix element corresponding to the photon absorption process can be expressed as,

.. math::
    :label: absorption_matrix_element

    v_{k,\alpha}^{(A)} \sqrt{\Delta k} = \langle \Phi_{ES}^{(A)} \mid H_{\mathrm{int}} \mid \Phi_{GS}^{(A)}, 1_{k,\alpha} \rangle.

Here :math:`\Delta k` is the mode width parameter used to normalize the photon modes. For details, see Ref. [1].

Once the matrix elements are computed, we then computed the overall energy transfer process matrix element as,

.. math::
    :label: M_NRET

    M = \frac{i\pi n_i}{\hbar c}\sum_{k,\alpha} v_{k,\alpha}^{(S)} v_{k,\alpha}^{(A)}.

Once M is computed, the energy transfer can be computed under second order perturbation using Fermi's golden rule as outlined in Ref. [1].

With this background, now to start the proess we need the orbital wavefuntions that constitute the multireference states. In the following section, we discuss how the wavefunctions are handled in **pyRET**.


Loading Wavefunctions
---------------------

The wavefunction object in **pyRET** is designed to contain three categories of information:

(1) Plane wave expansion
~~~~~~~~~~~~~~~~~~~~~~~~
A plane wave expansion of the wavefunction, as typically done in DFT codes such as Quantum ESPRESSO. In plane wave basis, the single electron orbitals are defined as

.. math::
    :label: plane_wave_expansion

    \phi(r) = \frac{1}{\sqrt{V}} \sum_{\mathbf{G}} c^{-}_{n\mathbf{k}}(\mathbf{G})e^{i(\mathbf{k}+\mathbf{G}) \cdot \mathbf{r}}

The plane wave vectors :math:`\mathbf{G}` and the expansion coefficients :math:`c` are stored as .dat or .hdf5 files outputed by Quantum Espresso. **pyRET** provides modules that can read these plane wave basis and store the coefficients in an object of the `WFunction` class. The plane wave expansion is used to directly compute the k-operated wavefunction values, which are used in the computation of the matrix elements of the interaction Hamiltonian. For example,

.. math::
    :label: plane_wave_expansion

    \begin{aligned}
    k_x \phi(r) &= -i \frac{\partial}{\partial x} \phi(r) \\
    &= \frac{1}{\sqrt{V}} \sum_{\mathbf{G}} c^{-}_{n\mathbf{k}}(\mathbf{G}) (k_x+G_x) e^{i(\mathbf{k}+\mathbf{G}) \cdot \mathbf{r}}
    \end{aligned}

(2) Spherical harmonic expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, **pyRET** can operate the `WFunction` class in the form of a multipolar expansion of the wavefunction using the `Scalar_field` class where the angular components are expressed using the `MP_SA` (multipolar-scalar-angular) class and the radial function can be chosen to be either spherical Bessel functions or atomic orbital like radial dependence. The last feature helps in creating atomic orbital basis within **pyRET**.

(3) Real space grid
~~~~~~~~~~~~~~~~~~~
A real space grid of :math:`\mid \phi(r)\rangle`, :math:`k_x \mid \phi(r)\rangle`, :math:`k_y \mid \phi(r)\rangle`, and :math:`k_z \mid \phi(r)\rangle`. These k-operated values are directly computed from the plane wave expansion, and thus free from any numerical differentiation errors in computing the spatial derivatives. The calculated position grid are stored as an object of the `WFunctions_data` class defined in the `data_classes.py` module. For more details please see the examples in the examples directory.


Define emitters and their positions
-----------------------------------
The next step is to define the emitter types and emitter positions corresponding to the system we want to simulate. **pyRET** stores the information on the emitters (source and absorber) and their relative positions as objects of classes `Emitters_data` and `Positions_data`. The emitters can be defined as dipolar emitters (electric dipole or magnetic dipole), or realistic emitters the orbitals of which are defined from first principles. The objects of the `Emitters_data` class stores as a dictionary all the defined single-orbital transitions, radiative lifetimes, energies, and other optional properties (like transition dipole moment for ideal dipole-like emitters) where the keys of the dictionary represents the emitter-id (represented as a string). Each emitter in the system of emitters possess an unique id. However, emitters with different emitter id can have same emitter types (also representd as a string). The `Positions_data` class objects store the relative positions between all possible pairs of emitters. Please refer to the detailed documentation and the example scripts for more details on usage.


Compute matrix elements
-----------------------
Once the single electron orbital wavefunctions have been imported into the `WFunction` and `WFunctions_data` class objects, and the emitter configurations are defined, we are now ready to compute the matrix elements corresponding to the absorption and emission, i.e. :math:`v_{k,\alpha}^{(S)}` and :math:`v_{k,\alpha}^{(A)}`. In practice this computation can be done in three different ways listed below with the order of increasing scalability:

(1) Computation for single multipoles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To quickly check calculation of a single case in python you can directly run the `V12_par` method for computation using wavefunctions, `V12_ED` for matrix element computation for electric dipolar transition, and `V12_MD` for magnetic dipolar transition. Please refer to the documentation for more details. As an example, the `V12_par` method can be invoked by using

.. code-block:: python

    v = V12_par(
        initial_wavefunction_grid: np.ndarray, # Wavefunction grid data for initial state
        k_initial_wavefunction_grid: np.ndarray, # k-operated wavefunction grid data for initial state
        final_wavefunction_grid: np.ndarray, # Wavefunction grid data for final state
        k_final_wavefunction_grid: np.ndarray, # k-operated wavefunction grid data for final state
        rgrids: np.ndarray, #Positiongrid including wavefunction grid and relative positions between multipole center and the emitter center
        k: float, # wavenumber of the photon in the medium
        L: int, # Orbital angular momentum of the photon
        M: int, # Projected total angular momentum of the photon
        P: int, # Parity of the photon
        nIndex: float = 1., # Refractive index of the medium
        Cgauge: int = 0, # Arbitrary gauge parameter. C=0 Coulomb.
        Hint: str = "Adotp", # Interaction hamiltonian type
        rshift: np.ndarray = np.zeros(3), # Additional shift of position as a control parameter
        radtype: str = "J", # Radial function of the photon mode.
        dV: np.ndarray = None, # Volume element of WF grid
        etaeff: float = 1., # Effective mass scaling
        parallel: bool = True, # execution platform- parallel or serial
        ):
    
In a similar way, `V12_ED` and `V12_MD` can be used when the emitter transition is known to be of ideal electric dipole or magnetic dipole types. For example, 

.. code-block:: python

    v = V12_ED(
        pEDs: np.ndarray, # Transition dipole moment vector for the emitter transition
        rvects: np.ndarray, # Relative position vector between the emitter and the multipole center
        k: float, # wavenumber of the photon in the medium
        L: int, # Orbital angular momentum of the photon
        M: int, # Projected total angular momentum of the photon
        P: int, # Parity of the photon mode
        nIndex: float = 1., # Refractive index of the medium
        radtype: str = "J", # Radial function of the photon mode.
        Cgauge: int = 0, # Arbitrary gauge parameter. C=0 Coulomb.
        etaeff: float = 1., # Effective mass scaling
        includespin: bool = True, # Include spin in the calculation
        )

and 

.. code-block:: python

    v = V12_MD(
        pMDs: np.ndarray, # Transition magnetic dipole moment vector for the emitter transition
        rvects: np.ndarray, # Relative position vector between the emitter and the multipole center
        k: float, # wavenumber of the photon in the medium
        L: int, # Orbital angular momentum of the photon
        M: int, # Projected total angular momentum of the photon
        P: int, # Parity of the photon mode
        nIndex: float = 1., # Refractive index of the medium
        radtype: str = "J", # Radial function of the photon mode.
        Cgauge: int = 0, # Arbitrary gauge parameter. C=0 Coulomb.
        etaeff: float = 1., # Effective mass scaling
        includespin: bool = True, # Include spin in the calculation
        )

:code:`v[i,j]` represents the matrix elements for different spin flip configurations:
- :code:`i,j = 0,0`: spin conserving transition at both initial and final states
- :code:`i,j = 0,1`: spin conserving transition at initial state, spin flip at final state
- :code:`i,j = 1,0`: spin flip transition at initial state, spin conserving transition at final state
- :code:`i,j = 1,1`: spin flip transition at both initial and final states

(2) Computation for all multipoles and all positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While the above way computes the matrix elements for a specific multipolar mode of photon, in computing the NRET rates one needs to sum over all multipolar modes, i.e. :math:`L = 1, 2, ... L_{max}`, :math:`M = -L, -L+1,..., L` and :math:`P = -1 or 1`. This can be done using the :code:`computeV` method. An example usage is follows:

.. code-block:: python
    
    v = computeV(
        absorber_id: str, # unique emitter-id corresponding to the absorber
        source_id: str, # unique emitter-id corresponding to the source
        absorber_transitions: list of str, # List of transitions at absorber
        ws: list of float, # List of energies of photons (in rad/s)
        Lmax: int = 2, # Max order of the multipoles
        Em_data: Emitters_data, # Emitters data object
        Pos_data: Positions_data, # Positions data object
        WF_data: list of WFunctions_data, # List of WFunctions_data objects that will be searched for the states
        parallel: bool = False,
        )

In this case the return (:code:`v`) is an object of the class :code:`Vdata` and specific matrix elements can be retrieved as

.. code-block:: python

    v_LMP = v.V[absorber_id][source_id][absorber_transition][L][M][P]

The shape of :code:`v_LMP` is :code:`(2,2,) + (Nw,) + Pos_data.Rg["absorber_id"]["source_id"]`. The first two axis encodes the information on whether there is a spin conserving or spin flip process at the initial and final states as also discussed under item (1) above.


(3) Compute for all multipoles and all positions in a parallel setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run calculations in parallel, first save the :code:`Emitters_data` and :code:`Positions_data` class objects containing the configurations of the emitter transitions and the spatial positions into json files by using the inbuild :code:`Encode` methods. For example

.. code-block:: python

    with open("Em_data.json", "w") as f:
        json.dump(Em_data.Encode(), f)

    with open(f"Pos_data.json", "w") as f:
        json.dump(Pos_data.Encode(), f)

Then, the :code:`Vin` class can be used to create an input file corresponding to the matrix element calculation setting, for example:

.. code:: python

    vin = Vin(
        emdatafile, # path to Em_data.json 
        posdatafile, # path to Pos_data.json 
        wfdatafiles, # list of paths to WFdata.json files
        absorber_id, # absorber_id, 
        emitter_id, # source_id, 
        absorber_transitions, # list of transitions
        nIndex, # refractive index of medium
        Lmax, # Max multipolar order
        ws, # Photon energies (rad/s),
        savepath = "./",
        savefile = "V.json",
        radtype_abs = "J",
        )

    vin.write_file(f"v.in")
    
Finally, the job can be run using the :code:`run.py` script. For example,

.. code-block:: bash

    $ srun python <path to pyRET>/run.py -in v.in >v.out

Alternatively,

.. code-block:: bash

    $ srun python pyRET_compute_matrix_elements -in v.in >v.out

The output will be a :code:`V.json` file that can be decoded to the matrix element objects by using

.. code-block:: python

    with open("V.json", "r") as f: 
        V = Vdata().Decode(json.load(f))
    
References
----------
[1] S. Chattaraj and G. Galli, Energy transfer between localized emitters in photonic cavities from first principles, Phys. Rev. Research 7, 033229 (2025).

[2] S. Chattaraj and G. Galli, Energy transfer between localized emitters in photonic cavities from first principles, Phys. Rev. Research 7, 033229 (2025).

