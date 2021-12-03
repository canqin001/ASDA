#!/bin/bash
step='ASDA'
date='1002'
method='ASDA'
model='resnet34'
LR=0.01
num=3
device_id=1
main_file='main_asda.py'
dataset='multi'
mmd_type='rbf'

source=real
target=clipart
CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
--method $method --dataset $dataset --name $step \
 --source $source --target $target --net $model --lr $LR --num $num \
 > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

# source=real
# target=painting
# CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
#  --method $method --dataset $dataset --name $step \
#  --source $source --target $target --net $model --lr $LR --num $num \
#  > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

# source=painting
# target=clipart
# CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
#  --method $method --dataset $dataset --name $step \
#  --source $source --target $target --net $model --lr $LR --num $num \
#  > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

# source=clipart
# target=sketch
# CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
#  --method $method --dataset $dataset --name $step \
#  --source $source --target $target --net $model --lr $LR --num $num --script $main_file \
#  > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

# source=sketch
# target=painting
# CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
#  --method $method --dataset $dataset --name $step \
#  --source $source --target $target --net $model --lr $LR --num $num --script $main_file \
#  > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

# source=real
# target=sketch
# CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
# --method $method --dataset $dataset --name $step \
#  --source $source --target $target --net $model --lr $LR --num $num --script $main_file \
#  > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

# source=painting
# target=real
# CUDA_VISIBLE_DEVICES=$device_id python $main_file --mmd_type $mmd_type \
#  --method $method --dataset $dataset --name $step \
#  --source $source --target $target --net $model --lr $LR --num $num --script $main_file \
#  > '../semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$step'_'$LR'.file'

