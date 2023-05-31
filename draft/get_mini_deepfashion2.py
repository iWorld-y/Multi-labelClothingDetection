"""
@Time    : 2023/5/18 18:55
@Author  : Eugene
@FileName: get_mini_deepfashion2.py 
"""
# %%
import json
import os
from tqdm import tqdm

data_path = r"E:\BaiduNetdiskDownload\Deepfashion2\Deepfashion2"
train = os.path.join(data_path, "train")
val = os.path.join(data_path, "validation")


# %%
class ImageClass:
    def __init__(self, is_train):
        self.is_train = is_train
        self.images = {}


# %%
image_class = []
for train_val in [train, val]:
    temp = ImageClass(is_train=(train_val == train))
    num_of_all_images = len(os.listdir(os.path.join(train_val, "image")))
    annos = os.path.join(train_val, "annos")
    image = os.path.join(train_val, "image")
    for entry in tqdm(os.scandir(annos), total=num_of_all_images):
        if entry.is_file() and entry.name.endswith(".json"):
            with open(entry.path) as f:
                temp.images[entry.name] = json.loads(f.read())
    image_class.append(temp)
# %%
category = {"short sleeve top": 0, "long sleeve top": 0, "short sleeve outwear": 0, "long sleeve outwear": 0,
            "vest": 0, "sling": 0, "shorts": 0, "trousers": 0, "skirt": 0, "short sleeve dress": 0,
            "long sleeve dress": 0, "vest dress": 0, "sling dress": 0}
category_train = {'short sleeve top': 71645, 'long sleeve top': 36064, 'short sleeve outwear': 543,
                  'long sleeve outwear': 13457, 'vest': 16095, 'sling': 1985, 'shorts': 36616, 'trousers': 55387,
                  'skirt': 30835, 'short sleeve dress': 17211, 'long sleeve dress': 7907, 'vest dress': 17949,
                  'sling dress': 6492}
category_val = {'short sleeve top': 12556, 'long sleeve top': 5966, 'short sleeve outwear': 142,
                'long sleeve outwear': 2011, 'vest': 2113, 'sling': 322, 'shorts': 4167, 'trousers': 9586,
                'skirt': 6522, 'short sleeve dress': 3127, 'long sleeve dress': 1477, 'vest dress': 3352,
                'sling dress': 1149}
# %%
