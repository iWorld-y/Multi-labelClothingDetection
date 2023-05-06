# 将训练图片和验证图片在逻辑上归为一个文件夹中
find /home/eugene/autodl-tmp/DeepFashion2/train/images/ -type f -exec ln -s {} /home/eugene/autodl-tmp/VOC_DeepFashion2/JPEGImages/ \;
find /home/eugene/autodl-tmp/DeepFashion2/validation/images/ -type f -exec ln -s {} /home/eugene/autodl-tmp/VOC_DeepFashion2/JPEGImages/val \;
