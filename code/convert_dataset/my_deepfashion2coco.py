"""
@Time    : 2023/4/21 12:53
@Author  : Eugene
@FileName: my_deepfashion2coco.py 
"""
import json
import os

from PIL import Image
import numpy as np
from tqdm import tqdm


def get_dataset() -> dict:
    """
    :return: 返回 dataset 字典
    """
    dataset: dict = {
        "info": {}, "licenses": [], "images": [], "annotations": [], "categories": []
    }
    cloth_categorie: list = ["short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
                             "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
                             "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"]

    categories: dict = {
        "id": '',
        'name': '',
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'supercategory': 'clothes',
        'skeleton': []
    }
    for i in range(13):
        temp = categories.copy()
        temp['id'] = i + 1
        temp['name'] = cloth_categorie[i]
        dataset['categories'].append(temp)
    return dataset


def change_points(points, left, rigth, points_x, points_y, points_v):
    for n in range(left, rigth):
        points[3 * n] = points_x[n - left]
        points[3 * n + 1] = points_y[n - left]
        points[3 * n + 2] = points_v[n - left]
    return points


def get_annotation(dataset_path: str, dataset: dict, dataType: str) -> dict:
    if (os.path.exists(os.path.join(dataset_path, dataType, "image"))):
        num_images = len(os.listdir(
            os.path.join(dataset_path, dataType, "image")))
        image_file_name = "image"
    else:
        num_images = len(os.listdir(os.path.join(
            dataset_path, dataType, "images")))
        image_file_name = "images"
    sub_index = 0
    # the index of ground truth instance
    if (dataset_path[-1] == '/'):
        dataset_path = dataset_path[:-1]
    print(f"{dataType}:\t{num_images}")
    for num in tqdm(range(1, num_images)):
        json_name = os.path.join(
            dataset_path, dataType, "annos", f"{num:06d}.json")
        image_name = os.path.join(
            dataset_path, dataType, image_file_name, f"{num:06d}.jpg")

        if (num >= 0):
            imag = Image.open(image_name)
            width, height = imag.size
            with open(json_name, 'r') as f:
                temp = json.loads(f.read())
                pair_id = temp['pair_id']

                dataset['images'].append({
                    'coco_url': '', 'date_captured': '', 'file_name': str(num).zfill(6) + '.jpg', 'flickr_url': '',
                    'id': num, 'license': 0, 'width': width, 'height': height})
                for i in temp:
                    if i == 'source' or i == 'pair_id':
                        continue
                    else:
                        points = np.zeros(294 * 3)
                        sub_index = sub_index + 1
                        box = temp[i]['bounding_box']
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        x_1 = box[0]
                        y_1 = box[1]
                        bbox = [x_1, y_1, w, h]
                        cat = temp[i]['category_id']
                        style = temp[i]['style']
                        seg = temp[i]['segmentation']
                        landmarks = temp[i]['landmarks']

                        points_x = landmarks[0::3]
                        points_y = landmarks[1::3]
                        points_v = landmarks[2::3]
                        points_x = np.array(points_x)
                        points_y = np.array(points_y)
                        points_v = np.array(points_v)

                        if cat == 1:
                            points = change_points(
                                points, 0, 25, points_x, points_y, points_v)
                        elif cat == 2:
                            points = change_points(
                                points, 25, 58, points_x, points_y, points_v)
                        elif cat == 3:
                            points = change_points(
                                points, 58, 89, points_x, points_y, points_v)
                        elif cat == 4:
                            points = change_points(
                                points, 89, 128, points_x, points_y, points_v)
                        elif cat == 5:
                            points = change_points(
                                points, 128, 143, points_x, points_y, points_v)
                        elif cat == 6:
                            points = change_points(
                                points, 143, 158, points_x, points_y, points_v)
                        elif cat == 7:
                            points = change_points(
                                points, 158, 168, points_x, points_y, points_v)
                        elif cat == 8:
                            points = change_points(
                                points, 168, 182, points_x, points_y, points_v)
                        elif cat == 9:
                            points = change_points(
                                points, 182, 190, points_x, points_y, points_v)
                        elif cat == 10:
                            points = change_points(
                                points, 190, 219, points_x, points_y, points_v)
                        elif cat == 11:
                            points = change_points(
                                points, 219, 256, points_x, points_y, points_v)
                        elif cat == 12:
                            points = change_points(
                                points, 256, 275, points_x, points_y, points_v)
                        elif cat == 13:
                            points = change_points(
                                points, 275, 294, points_x, points_y, points_v)
                        num_points = len(np.where(points_v > 0)[0])

                        dataset['annotations'].append({
                            'area': w * h, 'bbox': bbox, 'category_id': cat, 'id': sub_index, 'pair_id': pair_id,
                            'image_id': num, 'iscrowd': 0, 'style': style, 'num_keypoints': num_points,
                            'keypoints': points.tolist(), 'segmentation': seg})
    return dataset


def main(path: str):
    for dataType in ["train", "validation"]:
        dataset = get_annotation(path, get_dataset(), dataType)
        json_name = {"train": "./train.json",
                     "validation": "./valid.json"}[dataType]
        with open(os.path.join(path, json_name), 'w') as f:
            json.dump(dataset, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="DeepFashion2 根目录")
    args = parser.parse_args()

    assert args.path, "--path 为必填"
    main(args.path)
