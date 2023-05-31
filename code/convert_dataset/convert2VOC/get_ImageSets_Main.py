"""
@Time       : 2023-05-06 23:56
@Author     : iWorld-y
@FileName   : get_ImageSets_Main.py
"""
import os
import getpass


def main(save: str):
    username = getpass.getuser()
    if (username == 'root'):
        root_path = r'/root/autodl-tmp/VOC_DeepFashion2/'
    else:
        root_path = rf'/home/{username}/autodl-tmp/VOC_DeepFashion2/'

    train_images_path = os.path.join(root_path, 'JPEGImages')
    val_images_path = os.path.join(root_path, 'JPEGImages', "val")

    # 获取 train 图像文件名列表，写入 train.txt 文件中
    with open(os.path.join(root_path, f'ImageSets/{save}/train.txt'), 'w') as f, open(
            os.path.join(root_path, f'ImageSets/{save}/trainval.txt'), 'w') as f2:
        for name in sorted(os.listdir(train_images_path)):
            if os.path.splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png']:
                image_num = os.path.splitext(name)[0]
                f.write(image_num + '\n')
                f2.write(image_num + '\n')

    # 获取 validation 图像文件名列表，写入 val.txt 文件中
    with open(os.path.join(root_path, f'ImageSets/{save}/val.txt'), 'w') as f, open(
            os.path.join(root_path, f'ImageSets/{save}/trainval.txt'), 'a') as f2:
        for name in sorted(os.listdir(val_images_path)):
            if os.path.splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png']:
                image_num = os.path.splitext(name)[0]
                f.write(image_num + '\n')
                f2.write(f"val/{image_num}" + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str,
                        help="ImageSets 下的路径，Main or Segmentation")
    args = parser.parse_args()
    assert args.save, "--save 不可空"
    main(args.save)
