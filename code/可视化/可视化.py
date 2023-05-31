"""
@Time    : 2023/5/31 11:12
@Author  : Eugene
@FileName: 可视化.py 
"""
# %%
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import json
from tqdm import tqdm

# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
# plt.rcParams["font.sans-serif"] = ["SimSun"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
cn_font = font_manager.FontProperties(fname=r'D:\Code\PythonLearn\1Ali\BigWork\能鸽善鹉\SimSun.ttf')
en_font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
# %%
json_path = r"E:\BaiduNetdiskDownload\Deepfashion2\Deepfashion2\train\annos"
print()


# %%
def get_classes_num(dicts):
    for json_num in tqdm(range(1, len(os.listdir(json_path)) + 1)):
        # for json_num in tqdm(range(1, 10)):
        with open(os.path.join(json_path, f"{json_num:06d}.json"), 'r') as f:
            data = json.load(f)
            for i in range(10):
                key = f"item{i + 1}"
                try:
                    cloth_class = data[key]["category_name"]
                    dicts[cloth_class] += 1
                except Exception as e:
                    pass
    return dicts


# %%
dicts = {'short sleeve top': 71645,
         'trousers': 55387,
         'long sleeve dress': 7907,
         'long sleeve top': 36064,
         'skirt': 30835,
         'shorts': 36616,
         'long sleeve outwear': 13457,
         'vest dress': 17949,
         'short sleeve dress': 17211,
         'vest': 16095,
         'sling dress': 6492,
         'short sleeve outwear': 543,
         'sling': 1985}
dicts = get_classes_num(dicts)
# %%
title = "数据集类别分布"
x = list(dicts.values())
y = list(dicts.keys())
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
for i in range(len(x)):
    ax.barh(y[i], x[i])
plt.title(title, fontproperties=cn_font, fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
# 保存图片时加上 bbox_inches='tight'
fig.savefig(os.path.join("D:\CodeProject\Multi-labelClothingDetection\code\可视化", f"{title}.png"),
            bbox_inches='tight', dpi=800)
