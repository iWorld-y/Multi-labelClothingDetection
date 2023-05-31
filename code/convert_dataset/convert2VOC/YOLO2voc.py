"""
@Time       : 2023/4/24 15:40
@Author     : @秋野
@FileName   : YOLO2voc.py
@URL        : https://blog.csdn.net/BGMcat/article/details/120889683
@Modifier   : iWorld-y
"""
import getpass
from xml.dom.minidom import Document
import os
import cv2
from tqdm import tqdm
import threading


def makexml(picPath: str, txtPath: str, xmlPath: str) -> None:
    """
    此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    :param picPath: txt所在文件夹路径
    :param txtPath: xml文件保存路径
    :param xmlPath: 图片所在文件夹路径
    """
    # 创建字典用来对类型进行转换
    # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
    dic = {'0': 'short sleeve top', '1': 'long sleeve top', '2': 'short sleeve outwear', '3': 'long sleeve outwear',
           '4': 'vest', '5': 'sling',
           '6': 'shorts', '7': 'trousers', '8': 'skirt', '9': 'short sleeve dress', '10': 'long sleeve dress',
           '11': 'vest dress', '12': 'sling dress'}

    files = os.listdir(txtPath)
    for i, name in tqdm(enumerate(files), total=len(files)):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        img = cv2.imread(picPath + name[0:-4] + ".jpg")
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("images")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(
                ((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(
                ((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(
                ((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(
                ((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml_no_declaration(f, indent='\t', newl='\n',
                                           addindent='\t', encoding='utf-8')
        f.close()


# 定义函数，用于多线程执行 makexml() 函数


def process_data(picPath, txtPath, xmlPath):
    makexml(picPath, txtPath, xmlPath)


def main():
    username = getpass.getuser()
    if (username == 'root'):
        dataset_path = rf'/root/autodl-tmp/DeepFashion2'
    else:
        dataset_path = rf'/home/{username}/autodl-tmp/DeepFashion2'
    # dataset_kinds = ["train", "validation"]
    dataset_kinds = ["train", ]
    picPath = "images/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = "labels/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = "annotations/"  # xml文件保存路径，后面的/一定要带上

    # 创建线程列表
    threads = []

    # 遍历数据集种类
    for kinds in dataset_kinds:
        # 构造参数
        pic_path = os.path.join(dataset_path, kinds, picPath)
        txt_path = os.path.join(dataset_path, kinds, txtPath)
        xml_path = os.path.join(dataset_path, kinds, xmlPath)
        # 创建线程并启动
        thread = threading.Thread(
            target=process_data, args=(pic_path, txt_path, xml_path))
        thread.start()
        threads.append(thread)

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
