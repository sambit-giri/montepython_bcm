#SBATCH --time=24:00:00

#SBATCH --ntasks=12

#SBATCH --cpus-per-task=4

# 12 tasks * 4 cpus = 48 cores = 1 node in our cluster
# 48 cores * 24 hours = 1152 core-hours for this task

#SBATCH --mem-per-cpu=2G

#SBATCH --job-name=euclid_photo

#SBATCH --output=output_%J.txt


# delete fiducial data file
# the name of this file is specified in montepython/likelihoods/<likelihood name>/<likelihood name>.data
rm data/euclid_photometric_fiducial.npz
# generate new fiducial with -f0 flag
python3 montepython/MonePython.py run -o chains/euclid_photometric -p input/euclid_like.param -f0
# this generates a log.param file in the output directory. This process should not be parallelised, because then you generate 12 or so
# files 'inside each other', which is very messy (the classical parallelisation problem, I guess)



# run parallelised MCMCs
# -o sets the output directory, -f the jumping factor (default 2.4), --update the number of steps between two covmat update proposals,
# --superupdate the number of steps between each jumping factor update, and -j global disables super-sampling of nuisance parameters
$MPIEXEC $FLAGS_MPI_BATCH python3 montepython/MonePython.py run -o chains/euclid_photometric -f 1 --update 50 --superupdate 20 -j global
