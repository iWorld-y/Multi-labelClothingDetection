python /home/eugene/code/Multi-labelClothingDetection/code/pytorch_object_detection/mask_rcnn/train.py \
	--data-path "/home/eugene/autodl-tmp/" \
	--output-dir "/home/eugene/autodl-tmp/frcnn_2023-5-9_weights/2023-5-9_2" \
	--epochs 1 \
	--batch_size 2

#mygit "Faster RCNN 第 2 次训练结束，epochs=20; batch_size=16"
## 关闭计算机（一分钟后）
#echo -e "\033[41m\033[37m警告：计算机将在一分钟后关闭，如果想取消请按下 Ctrl + C\033[0m"
#sleep 1
#/usr/bin/shutdown