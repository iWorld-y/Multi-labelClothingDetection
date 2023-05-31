"""
@Time    : 2023/5/16 16:19
@Author  : Eugene
@FileName: AddSegmented2XML.py
@Describe: 为 VOC 数据集插入 Segmentation 标签，以便开始 Mask RCNN 训练
"""
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


class XmlEditor:
    def __init__(self, xmls_path, save_path) -> None:
        self.xmls_path = xmls_path
        self.save_path = save_path if save_path else self.xmls_path

    def editor(self, this_xml: int) -> None:
        assert isinstance(
            this_xml, int), f"this_xml 必须为正数！\t实际类型：{type(this_xml)}\t其值为: {this_xml}"
        tree = ET.parse(os.path.join(self.xmls_path, f"{this_xml:06d}.xml"))
        root = tree.getroot()
        seg = ET.SubElement(root, "segmented")
        seg.text = '1'
        tree.write(os.path.join(self.save_path, f"{this_xml:06d}.xml"))

    def multithreading_save_mask(self, start: int, end: int):
        for i in tqdm(range(start, end + 1)):
            self.editor(i)

    def multithreading(self):
        threads = []
        threads_num = 10
        xmls_num = len(os.listdir(self.xmls_path))
        offset = xmls_num // threads_num
        start = 1
        end = start + offset
        for i in range(threads_num):
            while (start <= xmls_num):
                t = threading.Thread(
                    target=self.multithreading_save_mask, args=(start, end))
                threads.append(t)
                t.start()
                start = end + 1
                end = start + offset \
                    if (start + offset <= xmls_num) else xmls_num
        for t in threads:
            t.join()


if __name__ in "__main__":
    import argparse
    import threading

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, help="Xml 所在目录")
    parser.add_argument("--save", type=str, help="Xml 保存目录")
    args = parser.parse_args()
    assert args.path, f"args.path 为必填参数"
    XmlEditor(args.path, args.save).multithreading()
