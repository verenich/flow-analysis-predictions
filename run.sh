# Specifies an email address
#PBS -M ilyavere@gmail.com

# Says that an email will be sent to the specified address 
# at the beginning (-b) and end (-e) of a job and in case the job gets aborted (-a)
#PBS -m bea

# Sets the working directory of this jobscript
##PBS -d /home/n9334378/new-flow-analysis/

# Here, finally you can put the actual commands of your job, that will be run
# on the cluster.
source /etc/profile.d/modules.sh
module load python/3.5.2-foss-2016b
python -V
export PYTHONPATH=/home/n9334378/new-flow-analysis/
cd /home/n9334378/new-flow-analysis/core/
python train_direct.py $dataset $method $encoding xgb 30
python train_FA.py $dataset $method $encoding xgb $min_cases
