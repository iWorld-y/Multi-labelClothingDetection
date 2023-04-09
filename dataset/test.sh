# 生成训练集标签
python /home/eugene/code/Multi-labelClothingDetection/example.py --datasets COCO --img_path ./train/image/ --label train.json --convert_output_path YOLO/ --img_type ".jpg" --manipast_path train.txt --cls_list_file fashion_classes.txt

# 生成验证集标签
python /home/eugene/code/Multi-labelClothingDetection/example.py --datasets COCO --img_path ./validation/image/ --label valid.json --convert_output_path YOLO/ --img_type ".jpg" --manipast_path valid.txt --cls_list_file fashion_classes.txt
