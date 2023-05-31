python /root/code/Multi-labelClothingDetection/code/pytorch_object_detection/ssd/train_ssd300.py \
    --data-path "/root/autodl-tmp/VOC_DeepFashion2" \
    --output-dir "/root/autodl-tmp/ssd_wegihts" \
    --epochs 100 \
    --batch-size 12 && 
mygit "SSD 训练完成，epochs 100，batch-size 96" 

# 关闭计算机（一分钟后）
echo -e "\033[41m\033[37m警告：计算机将在一分钟后关闭，如果想取消请按下 Ctrl + C\033[0m"
sleep 60
/usr/bin/shutdown