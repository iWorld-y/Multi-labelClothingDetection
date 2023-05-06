_B='在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！'
_A='utf-8'
import os,random,xml.etree.ElementTree as ET,numpy as np
from utils.utils import get_classes
annotation_mode=2
classes_path='dataset/classes.txt'
trainval_percent=.9
train_percent=.9
DeepFashion2_path='~/autodl-tmp/DeepFashion2/train'
VOCdevkit_sets=[('2007','train'),('2007','val')]
classes,_=get_classes(classes_path)
photo_nums=np.zeros(len(VOCdevkit_sets))
nums=np.zeros(len(classes))
def convert_annotation(year,image_id,list_file):
	E='difficult';F=open(os.path.join(DeepFashion2_path,f"annotations/{image_id}.xml"),encoding=_A);G=ET.parse(F);H=G.getroot()
	for A in H.iter('object'):
		D=0
		if A.find(E)!=None:D=A.find(E).text
		B=A.find('name').text
		if B not in classes or int(D)==1:continue
		I=classes.index(B);C=A.find('bndbox');J=int(float(C.find('xmin').text)),int(float(C.find('ymin').text)),int(float(C.find('xmax').text)),int(float(C.find('ymax').text));list_file.write(' '+','.join([str(A)for A in J])+','+str(I));nums[classes.index(B)]=nums[classes.index(B)]+1
if __name__=='__main__':
	random.seed(0)
	if' 'in os.path.abspath(DeepFashion2_path):raise ValueError('数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。')
	if annotation_mode==0 or annotation_mode==1:
		print('Generate txt in ImageSets.');xmlfilepath=os.path.join(DeepFashion2_path,'annotations');saveBasePath=os.path.join(DeepFashion2_path,'ImageSets/Main');temp_xml=os.listdir(xmlfilepath);total_xml=[]
		for xml in temp_xml:
			if xml.endswith('.xml'):total_xml.append(xml)
		num=len(total_xml);list=range(num);tv=int(num*trainval_percent);tr=int(tv*train_percent);trainval=random.sample(list,tv);train=random.sample(trainval,tr);print('train and val size',tv);print('train size',tr);ftrainval=open(os.path.join(saveBasePath,'trainval.txt'),'w');ftest=open(os.path.join(saveBasePath,'test.txt'),'w');ftrain=open(os.path.join(saveBasePath,'train.txt'),'w');fval=open(os.path.join(saveBasePath,'val.txt'),'w')
		for i in list:
			name=total_xml[i][:-4]+'\n'
			if i in trainval:
				ftrainval.write(name)
				if i in train:ftrain.write(name)
				else:fval.write(name)
			else:ftest.write(name)
		ftrainval.close();ftrain.close();fval.close();ftest.close();print('Generate txt in ImageSets done.')
	if annotation_mode==0 or annotation_mode==2:
		print('Generate 2007_train.txt and 2007_val.txt for train.');type_index=0
		for(year,image_set)in VOCdevkit_sets:
			image_ids=open(os.path.join(DeepFashion2_path,f"ImageSets/Main/{image_set}.txt"),encoding=_A).read().strip().split();list_file=open(f"{image_set}.txt",'w',encoding=_A)
			for image_id in image_ids:list_file.write(f"{os.path.abspath(DeepFashion2_path)}/JPEGImages/{image_id}.jpg");convert_annotation(year,image_id,list_file);list_file.write('\n')
			photo_nums[type_index]=len(image_ids);type_index+=1;list_file.close()
		print('Generate 2007_train.txt and 2007_val.txt for train done.')
		def printTable(List1,List2):
			A=List1
			for C in range(len(A[0])):
				print('|',end=' ')
				for B in range(len(A)):print(A[B][C].rjust(int(List2[B])),end=' ');print('|',end=' ')
				print()
		str_nums=[str(int(A))for A in nums];tableData=[classes,str_nums];colWidths=[0]*len(tableData);len1=0
		for i in range(len(tableData)):
			for j in range(len(tableData[i])):
				if len(tableData[i][j])>colWidths[i]:colWidths[i]=len(tableData[i][j])
		printTable(tableData,colWidths)
		if photo_nums[0]<=500:print('训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。')
		if np.sum(nums)==0:print(_B);print(_B);print(_B);print('（重要的事情说三遍）。')