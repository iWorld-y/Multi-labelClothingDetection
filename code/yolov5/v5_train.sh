#!/bin/bash

# 获取第一个参数，作为 Git 提交的消息
message="$1"

# 检查是否提供了 Git 提交的消息
if [ -z "$message" ]; then
  echo "请提供 Git 提交的消息作为第一个参数"
  exit 1
fi

# 运行 YOLOv5 训练脚本
# python ~/code/yolov5/train.py \
python -m torch.distributed.run --nproc_per_node 2 \
    --master_port 1024 \
    ~/code/yolov5/train.py \
    --batch-size 384 \
    --data dataset/fashion2.yaml \
    --img 640 \
    --epochs 300 \
    --weight models/yolov5n.pt \
    --project runs/v5 \
    --device 0,1 &&

# 提交 Git 更改
mygit "$message"

# 关闭计算机（一分钟后）
echo -e "\033[41m\033[37m警告：计算机将在一分钟后关闭，如果想取消请按下 Ctrl + C\033[0m"
sleep 60
/usr/bin/shutdown 