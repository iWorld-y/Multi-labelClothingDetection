# coding:utf-8
import json
import os
import os.path

from PIL import Image
from tqdm import tqdm

root_path = r"/root/autodl-tmp/DeepFashion2"
train_data_path = fr"{root_path}/train"
vaild_data_path = fr"{root_path}/validation"
data_path = vaild_data_path


def listPathAllfiles(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


if __name__ == '__main__':
    for data_path in [train_data_path, vaild_data_path]:
        annos_path = rf"{data_path}/annos"  # 改成需要路径
        image_path = rf"{data_path}/images"  # 改成需要路径
        labels_path = rf"{data_path}/labels"  # 改成需要路径

        num_images = len(os.listdir(annos_path))

        for num in tqdm(range(1, num_images + 1)):
            json_name = os.path.join(annos_path, str(num).zfill(6) + '.json')
            image_name = os.path.join(image_path, str(num).zfill(6) + '.jpg')
            txtfile = os.path.join(labels_path, str(num).zfill(6) + '.txt')
            imag = Image.open(image_name)
            width, height = imag.size

            res = []
            with open(json_name, 'r') as f:
                temp = json.loads(f.read())
                for i in temp:
                    if i == 'source' or i == 'pair_id':
                        continue
                    else:
                        box = temp[i]['bounding_box']
                        x_1 = round((box[0] + box[2]) / 2 / width, 6)
                        y_1 = round((box[1] + box[3]) / 2 / height, 6)
                        w = round((box[2] - box[0]) / width, 6)
                        h = round((box[3] - box[1]) / height, 6)

                        category_id = int(temp[i]['category_id'] - 1)

                        res.append(
                            " ".join([str(category_id), str(x_1), str(y_1), str(w), str(h)]))

            open(txtfile, "w").write("\n".join(res))
