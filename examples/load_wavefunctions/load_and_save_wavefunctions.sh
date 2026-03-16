#!\bin\bash

#First, copy the data files conaining DFT-calculated wavefunctions in plane wave basis
#git clone https://github.com/chattarajs/pyRET-Data.git

#Extract wavefunctions
module load python
conda activate ts1

#hdf5
srun -n 20 -c 3  python -u from_hdf5.py 

#dat
srun -n 1 -c 60 python -u from_dat.py
