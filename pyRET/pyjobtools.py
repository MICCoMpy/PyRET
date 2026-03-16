"""
Helper script to write slurm job scripts for submitting python programs to HPC clusters.  The script includes functions to generate job scripts for different platforms (e.g., Perlmutter, Midway2) with customizable options such as quality of service, time limit, number of nodes, tasks per node, and environment settings. The generated job scripts can be used to submit python programs to the cluster for execution.

"""
#Some helper job script generators
def pyjobtext(
        platform = "Perlmutt",
        qos = "regular",
        constraint = "knl",
        time = "00:10:00",
        jname = "job",
        n = 1,
        tpn = 30,
        cpt = 1,
        environment = "",
        pyprogram = None,
        options = None,
        outfile = "./out",
        account = "",
        ):
    
    if platform == "Perlmutter":
        jtext = [f"#!/bin/bash -x \n",  
            f"#SBATCH --qos={qos} \n",
            f"#SBATCH --account={account} \n",  
            f"#SBATCH -C {constraint} \n",  
            f"#SBATCH -t {time} \n",  
            f"#SBATCH -J {jname}  \n", 
            f"#SBATCH --nodes={n} \n",
            f"#SBATCH --ntasks-per-node={tpn} \n",
            f"#SBATCH --cpus-per-task={cpt} \n",
            f"module load python \n",
            f"conda activate {environment} \n",
            f"srun -n {n*tpn} -c {cpt} python -u {pyprogram}" ]
        
        for option, value in options.items():
            jtext += f"  -{option} {value} "
        
        jtext+= f" > {outfile}"
        return jtext

    elif platform == "Midway2":
        jtext = [f"#!/bin/bash -x \n",  
            f"#SBATCH --qos={qos} \n",
            f"#SBATCH --partition={constraint} \n",  
            f"#SBATCH --time={time} \n",  
            f"#SBATCH -J {jname}  \n", 
            f"#SBATCH --nodes={n} \n",
            f"#SBATCH --ntasks-per-node={tpn} \n",
            f"#SBATCH --cpus-per-task={cpt} \n",
            f"module load intelmpi-2018.2.199+intel-18.0 \n",
            f"module load python \n",
            f"module load mpi4py \n",
            f"conda activate {environment} \n",
            f"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \n",
            f"NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) \n",
            f"mpirun -np $NTASKS python -u {pyprogram}" ]
        
        for option, value in options.items():
            jtext += f"  -{option} {value} "
        
        jtext+= f" > {outfile}"
        return jtext

def pyjobtext_batch(
        platform = "Perlmutter",
        qos = "regular",
        constraint="knl",
        time = "00:10:00",
        jname = "job",
        n = 1,
        tpn = 30,
        cpt = 1,
        environment = "",
        pyprogram = None,
        paths=None,
        options=None,
        outfile = "./out",
        account = "",
        ):
    if platform == "Perlmutter":
        jtext = [f"#!/bin/bash -x \n",  
            f"#SBATCH --qos={qos} \n",
            f"#SBATCH --account={account} \n",  
            f"#SBATCH -C {constraint} \n",  
            f"#SBATCH -t {time} \n",  
            f"#SBATCH -J {jname}  \n", 
            f"#SBATCH --nodes={n} \n",
            f"#SBATCH --ntasks-per-node={tpn} \n",
            f"#SBATCH --cpus-per-task={cpt} \n",
            f"module load python \n",
            f"conda activate {environment} \n",
            ]
        for path, option in zip(paths, options):
            jtext += f"cd {path} \n"
            jtext+= f"srun -n {n*tpn} -c {cpt} python -u {pyprogram}"
            
            for opt, value in option.items():
                jtext += f"  -{opt} {value} "
            
            jtext+= f" > {path}/{outfile} \n"
        
        return jtext
    elif platform == "Midway2":
        jtext = [f"#!/bin/bash -x \n",  
            f"#SBATCH --qos={qos} \n",
            f"#SBATCH --partition={constraint} \n",  
            f"#SBATCH --time={time} \n",  
            f"#SBATCH -J {jname}  \n", 
            f"#SBATCH --nodes={n} \n",
            f"#SBATCH --ntasks-per-node={tpn} \n",
            f"#SBATCH --cpus-per-task={cpt} \n",
            f"module load intelmpi-2018.2.199+intel-18.0 \n",
            f"module load python \n",
            f"module load mpi4py \n",
            f"conda activate {environment} \n",
            f"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \n",
            f"NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) \n"]
        
        for path, option in zip(paths, options):
            jtext += f"cd {path} \n"
            jtext+= f"mpirun -np $NTASKS python -u {pyprogram}"
            
            for opt, value in option.items():
                jtext += f"  -{opt} {value} "
            
            jtext+= f" > {path}/{outfile} \n"
        
        return jtext