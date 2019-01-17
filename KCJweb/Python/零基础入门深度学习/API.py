# coding:UTF-8
# 2019-1-16

import os
import cv2
import math
import numpy as np 

__suffix = ['png', 'jpg']

def getFiles(dir_path):
	file = []
	for root, dirs, files in os.walk(dir_path, topdown=False):
		for name in files:
			path = os.path.join(root, name)
			if name.split('.')[-1] in __suffix:
				file.append(path)
	return file

def getTrainingData(dir_path):
	'''
	加载训练数据
	'''
	files = getFiles(dir_path)
	data, labels = [], []
	for f in files:
		img = cv2.imread(f, 0)
		m, n = img.shape
		img = np.array(img)
		year = int(os.path.basename(f).split('-')[1])
		label_temp = np.zeros(18) + 0.01
		label_temp[year] = 0.99

		data.append(img)
		labels.append(label_temp)

	return np.array(data), labels

if __name__ == '__main__':
	data, labels = getTrainingData('C:\\Users\\VISSanKCJ\\Desktop\\histogram')
	print(data[0], labels[0])




