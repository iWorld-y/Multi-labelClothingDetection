python ~/code/yolov5/train.py --resume;

# 提交 Git 更改
git add ~/code/Multi-labelClothingDetection ;
git commit -m "延长 YOLOv5 的第一次训练完成" ;
git push origin main ;

# 关闭计算机
/usr/bin/shutdown

# 训练完成后须将 `/root/code/yolov5/utils/torch_utils.py` 的 `smart_resume` 函数的 `start_epoch = ckpt['epoch'] + 1` 取消注释并删除新增行