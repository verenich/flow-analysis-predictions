# Specifies an email address
#PBS -M ilyavere@gmail.com

# Says that an email will be sent to the specified address 
# at the beginning (-b) and end (-e) of a job and in case the job gets aborted (-a)
#PBS -m bea

# Sets the working directory of this jobscript
##PBS -d /home/n9334378/new-flow-analysis2/

# Here, finally you can put the actual commands of your job, that will be run
# on the cluster.
source /etc/profile.d/modules.sh
module load python/3.6.1-intel-2017a
python -V
export PYTHONPATH=/home/n9334378/new-flow-analysis2/:$PYTHONPATH
cd /home/n9334378/new-flow-analysis2/core/
#echo "started hyperparameter optimization FA at $(date)"
#python optimize_hyperparameters_FA.py $dataset zero $encoding xgb $min_cases $niter
#echo "finished hyperparameter optimization at $(date)"
#python extract_best_params.py $dataset
echo "started experiments with optimal parameters FA at $(date)"
python train_FA_calib.py $dataset zero combined xgb $min_cases $optim_type
