#作业1

# lst = [1, 2, 3, 4]
# lst1 = []

# for i in lst:
# 	for j in lst:
# 		if i != j:
# 			lst1.append(str(i) + str(j))

# print(lst1)


# #作业2

# def f1():
# 	pre = True
# 	while pre:
# 		str1 = input('请输入三个整数，并用逗号隔开（如：12，22，11）：')
# 		lst = str1.split(',')

# 		if lst[0].isdigit() and lst[1].isdigit() and lst[2].isdigit():
# 			pre = False
# 			lst.sort(reverse = True)
# 			print(lst)
# 		else:
# 			print('输入不符合要求！')

# f1()



# def insert_px1(lst):
# 	print(lst)
# 	for i in range(1, len(lst)):
# 		lst1 = sorted(range(i+1), reverse=True)
# 		lst1.remove(0)
		
# 		# print(lst1)
# 		current_item = lst[i]    
# 		for j in lst1:

# 			if lst[j-1] > current_item:
# 				lst[j] = lst[j-1]
		
# 		lst[j-1] = current_item
		
# 	return lst

# lst1 = [1,3,2,6,8,5,4,2,5]
# lst_new = insert_px1(lst1)
# print(lst_new)

#2018-11-10
# def px(lst):
# 	j = 0
# 	for i in range(len(lst)):
# 		if lst[i] != 0:
# 			lst[j] = lst[i]
# 			j += 1
# 		else:
# 			continue

# 	lst[j:] = [0] * (i - j + 1)

# 	return lst
                         

# lst = [1,0,2,0,3,4,0,0,6,0]
# print(px(lst))

# import numpy as np
# ar = np.array([[1,2,3,4,5],[1,2,3,4,5]])
# print(ar)
# print(ar.ndim)
# print(ar.shape)
# print(ar.size)
# print(ar.dtype)
# print(ar.itemsize)

# print(np.array(range(10)))
# ar1 = np.array([[1,2,3],['a','s','d']])
# ar1 = np.linspace(1,5,9,endpoint=False,retstep=True)
# print(ar1)
# print(np.eye(6))


# import matplotlib.pyplot
# print (matplotlib.pyplot.version)

# weights = [0.0 for _ in range(2)]
# print(weights) #[0.0, 0.0]

# f = lambda x, y: x + y

# lst1 = [2 for i in range(3)]
# print(lst1)


# from functools import reduce
# print(reduce(lambda x, y: x * y, [1,2,3,4], 10))

# l1 = ['1','2','3','4']
# l2 = ['haha', 'xixi', 'lala', 'hehe']
# l4 = ['2','3','4','5']
# l3 = map(lambda x, y, z: x+y+z, l1, l2, l4)
# print(list(l3))

# print(list(zip(l1, l2)))


# l1 = [1, 2, 3, 4]
# l2 = [1, 1, 1, 1]
# b = 0
# print(reduce(lambda x, y: x + y, map(lambda x, y: x * y, l1, l2), b))
# input_data = [[0,0], [0,1], [1,0], [1,1]]
# target = [0, 0, 0, 1]

# epoch = 5
# for i in range(epoch):
#     for k,v in enumerate(input_data):
#         print(k)
#         print(v)

# import numpy as np
# l1 = np.array([1,1])
# print(3*l1)
# import numpy as np 


# def op(x):
# 	return x**2

# a = np.arange(6).reshape(2,3)
# print(a)
# for i in np.nditer(a,op_flags=['readwrite']):
#     i[...] = i + 1
# print(a)

# import numpy as np 

# weights = np.random.uniform(0, 10, (3, 3, 3))

# print(weights)
# flipped_weights= np.array(map(lambda i: np.rot90(i, 2),weights))
# print(np.array(flipped_weights))

import tensorflow