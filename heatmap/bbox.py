import pandas as pd 
import numpy as np 
import os
import cv2

image_list = []
df1 = pd.read_csv('BBox_List_2017.csv')
df2 = pd.read_csv('map_data.csv')
value1 = df1.values[:,0]
value2 = df2.values[:,0]
for i in range(len(value1)):
	for j in range(len(value2)):
		if value1[i] == value2[j]:
			image_list.append(list(value2).index(value2[j]))
image_list = list(set(image_list))
print len(image_list)

if not os.path.exists('./bbox'):
	os.mkdir('./bbox')

red = (0, 0, 225)
for i in range(len(image_list)):
	index = image_list[i]
	image_path = os.path.join('./CAM', str(index)+'.jpg')
	img = cv2.imread(image_path)
	origin = (int(df1.values[index, 2]), int(df1.values[index, 3]))
	end = (int(df1.values[index, 2]+df1.values[index, 4]), int(df1.values[index, 3]+df1.values[index, 5]))
	cv2.rectangle(img, origin, end, red, 5)
	write_path = os.path.join('./bbox', str(index)+'.jpg')
	cv2.imwrite(write_path, img)
	print i
