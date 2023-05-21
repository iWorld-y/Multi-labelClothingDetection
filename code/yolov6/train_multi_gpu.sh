#!/bin/bash

# 获取第一个参数，作为 Git 提交的消息
message="$1"

# 检查是否提供了 Git 提交的消息
if [ -z "$message" ]; then
    echo "请提供 Git 提交的消息作为第一个参数"
    exit 1
fi
# P5 models

python -m torch.distributed.launch --nproc_per_node 2 \
    /root/code/YOLOv6/tools/train.py \
    --batch 220 \
    --epoch 300 \
    --conf configs/yolov6n.py \
    --data /root/code/Multi-labelClothingDetection/dataset/voc.yaml \
    --fuse_ab \
    --device 0,1 \
    --name /root/code/Multi-labelClothingDetection/runs/v6n_coco_multiGPU \
    --check-images \
    --check-labels \
    --write_trainbatch_tb &&

# 提交 Git 更改
mygit "$message"

# 关闭计算机（一分钟后）
echo -e "\033[41m\033[37m警告：计算机将在一分钟后关闭，如果想取消请按下 Ctrl + C\033[0m"
sleep 60
/usr/bin/shutdown