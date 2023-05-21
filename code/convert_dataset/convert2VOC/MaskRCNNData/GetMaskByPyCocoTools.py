import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# 加载 COCO 数据集
dataDir = '/path/to/coco'
annFile = os.path.join(dataDir, 'annotations/instances_train2017.json')
coco = COCO(annFile)

# 获取所有图像 ID
imgIds = coco.getImgIds()

# 循环遍历所有图像 ID
for imgId in imgIds:
    # 获取该图像的标注信息
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)

    # 获取该图像的文件名
    img = coco.loadImgs(imgId)[0]
    imgName = img['file_name']

    # 创建空白图像
    maskImg = Image.new('1', (img['width'], img['height']))

    # 填充掩膜像素
    for ann in anns:
        rle = coco.annToRLE(ann)
        maskData = coco.maskUtils.decode(rle)
        maskImgData = np.asarray(maskImg)
        maskImgData |= maskData
        maskImg = Image.fromarray(maskImgData)

    # 保存 Mask PNG 文件
    maskImg.save(os.path.join(
        dataDir, 'annotations/masks/' + imgName[:-4] + '.png'))
