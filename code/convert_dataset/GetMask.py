import numpy as np
import json
import os
import cv2
from tqdm import tqdm


class GetMask:
    def __init__(self, data_root_path: str, annos: str, images: str, save_path: str, is_multithreading: bool,
                 is_debug: bool):
        """
        :param data_root_path: 数据集根目录
        :param annos: 原始标注文件
        :param images: 图片集根目录
        """
        self.data_root_path = data_root_path
        self.annos = os.path.join(self.data_root_path, annos)
        self.images = os.path.join(self.data_root_path, images)
        self.save_path = save_path
        self.is_multithreading = is_multithreading
        self.is_debug = is_debug

    def get_image(self, image_num: str) -> np.ndarray:
        """
        读取图片
        :param image_num: 图片号码
        :return: 返回包含图片数据的 np.ndarray
        """
        # return plt.imread(os.path.join(self.images, f"{image_num:06d}.jpg"))
        return cv2.imread(os.path.join(self.images, f"{image_num:06d}.jpg"))

    def get_image_json(self, image_num: str) -> dict:
        """
        获取图片信息
        :param image_num: 图片号码
        :return: 返回包含图片信息的 dict
        """
        with open(os.path.join(self.annos, f"{image_num:06d}.json"), 'r') as f:
            return json.load(f)

    def get_mask(self, image_num) -> np.ndarray:
        """
        返回图片掩膜图，只包含黑色背景和纯色掩膜
        :param image_num: 图片号码
        :return: 返回掩膜图
        """
        image_json = self.get_image_json(image_num)

        # 创建一个全黑的背景图
        black_bg = np.zeros_like(self.get_image(image_num))

        # 掩膜颜色
        colors = [[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 128, 0], [0, 0, 255], [75, 0, 130], [238, 130, 238]]
        color_cnt = 0
        colors_size = len(colors)
        # 绘制目标边框并生成目标掩膜
        for i in range(1, 10):
            item_key = f"item{i}"
            if item_key not in image_json:
                break
            # 按顺序上色
            mask_color = colors[color_cnt]
            color_cnt = (color_cnt + 1) % colors_size
            # 获取一系列多边形

            arrs = image_json[item_key]["segmentation"]
            for arr in arrs:
                # 对当前物体的分割点所连成的多边形进行处理
                pts = np.array([[arr[i], arr[i + 1]] for i in range(0, len(arr), 2)]).astype(int)
                cv2.fillPoly(black_bg, [pts], mask_color)
                cv2.polylines(black_bg, [pts], True, (255, 255, 255))
        return black_bg

    def count_images(self) -> int:
        """
        :return: 图片总数
        """
        return len(os.listdir(self.images))

    def save_mask(self, image_num: int) -> None:
        assert isinstance(image_num, int), f"image_num 必须为正数！\t实际类型：{type(image_num)}\t其值为: {image_num}"
        cv2.imwrite(os.path.join(self.save_path, f"{image_num:06d}.png"), self.get_mask(image_num))

    def multithreading_save_mask(self, start: int, end: int):
        for i in tqdm(range(start, end + 1)):
            self.save_mask(i)

    def multithreading(self):
        import threading
        threads = []
        # 图片总数
        if (self.is_debug):
            images_num = 1024
        else:
            images_num = self.count_images()
        # 线程数
        threads_num = 5
        offset = images_num // threads_num
        start = 1
        end = start + offset
        for i in range(threads_num):
            while (start <= images_num):
                t = threading.Thread(target=self.multithreading_save_mask, args=(start, end))
                threads.append(t)
                t.start()
                print(f"{start} ~ {end}")
                start = end + 1
                end = start + offset if (start + offset <= images_num) else images_num
        for t in threads:
            t.join()

    def do_convert(self):
        if (self.is_debug and (not self.is_multithreading)):
            for image_num in tqdm(range(1, 11)):
                self.save_mask(image_num)
        elif (self.is_multithreading):
            self.multithreading()
        else:
            for image_num in tqdm(range(self.count_images())):
                # cv2.imwrite(os.path.join(self.save_path, f"{image_num:06d}.png"), self.get_mask(image_num))
                self.save_mask(image_num)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成 DeepFashion2 掩膜图像数据")

    parser.add_argument("--data", type=str, help="训练集、验证集或者测试集的目录")
    parser.add_argument("--annos", type=str, default="annos", help="图片信息所在目录")
    parser.add_argument("--images", type=str, default="image", help="图片目录名称")
    parser.add_argument("--save", type=str, help="掩膜保存目录")
    parser.add_argument("--multithreading", action="store_true", help="是否开启多线程")
    parser.add_argument("--debug", action="store_true", help="是否开启 debug")

    args = parser.parse_args()
    GetMask(
        data_root_path=args.data,
        annos=args.annos,
        images=args.images,
        save_path=args.save,
        is_multithreading=args.multithreading,
        is_debug=args.debug
    ).do_convert()
