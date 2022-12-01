#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 3 ]
then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE]"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
DEVICE_RANGE=$3
START_DEVICE=${DEVICE_RANGE:1:1}
END_DEVICE=${DEVICE_RANGE:3:1}

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

if [ ! -f $CONFIG_FILE ]
then
    echo "error: config_path=$CONFIG_FILE is not a file"
exit 1
fi

if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
    echo "error: start_device=$START_DEVICE is not a number"
exit 1
fi

if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
    echo "error: end_device=$END_DEVICE is not a number"
exit 1
fi

ulimit -u unlimited
export RANK_SIZE=$(($END_DEVICE - $START_DEVICE))
export RANK_TABLE_FILE=$PATH1

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((i-START_DEVICE))
    rm -rf ./pretrain_parallel$i
    mkdir ./pretrain_parallel$i
    cp ../*.py ./pretrain_parallel$i
    cp *.sh ./pretrain_parallel$i
    cp -r ../config ./pretrain_parallel$i
    cp -r ../register ./pretrain_parallel$i
    cp -r ../ringmo_framework ./pretrain_parallel$i
    cd ./pretrain_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python pretrain.py --config=$CONFIG_FILE &> pretrain_log &
    cd ..
done
#sleep 1s
#cd ./pretrain_parallel${START_DEVICE} || exit
#tail -f pretrain_log

# if you want kill current job, you can use as follow:
# kill -9 $(ps aux | grep "python pretrain.py" | grep -v grep | awk '{print $2}')
