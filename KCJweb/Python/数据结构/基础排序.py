#2018-11-6
#for 循环每次自己更改变量i,而while语句需自己在语句内添加更新变量。确定i的取值范围时，先算出i的最大可能取值，则 i < max + 1.

#用while语句进行排序,


#交换列表元素位置
def swap(lst,i,j):

	temp = lst[i]
	lst[i] = lst[j]
	lst[j] = temp

	return(lst)


#选择排序:先将第一个数字与后面所有数字比较，每次将较小者放在第一个位置；同样方法再确定第二个位置的数字，以此类推。

def select_px(lst):
	leng = len(lst) - 1
	i = 0

	while i < leng:
		j = i + 1
		while j < leng + 1:
			if lst[i] > lst[j]:
				swap(lst, i, j)
			j += 1

		i += 1

	return lst


#冒泡排序：依次比较相邻两元素，将较大者排在后面。

def bubble_px(lst):

	leng = len(lst) - 1
	i = 0

	while i < leng:
		j = 0
		while j < leng - i:
			if lst[j] > lst[j+1]:
				swap(lst, j, j+1)
			j += 1

		i += 1

	return(lst)


# 插入排序，扑克牌排序：
def insert_px(lst):
	leng = len(lst)
	i = 0
	while i < leng:
		current_item = lst[i]
		j = i - 1
		while j >= 0:
			if current_item < lst[j]:
				lst[j+1] = lst[j]
				j -= 1
			else:
				break

		lst[j+1] = current_item
		i += 1

	return(lst)




#用for循环进行基础排序

#选择排序

def select_px1(lst):
	for i in range(len(lst) - 1):
		for j in range(i+1, len(lst)):
			if lst[i] > lst[j]:
				lst[i], lst[j] = lst[j], lst[i]

	return lst


#冒泡排序

# def bubble_px1(lst):
# 	for i in range(len(lst) - 1):
# 		for j in range(len(lst) - 1 - i):
# 			if lst[j] > lst[j+1]:
# 				lst[j], lst[j+1] = lst[j+1], lst[j]

# 	return lst



#插入排序


def insert_px1(lst):
	for i in range(1, len(lst)):
		lst1 = sorted(range(i+1), reverse=True)
		lst1.remove(0)
		current_item = lst[i]    
		for j in lst1:
			if lst[j-1] > current_item:
				lst[j] = lst[j-1]
		lst[j-1] = current_item

	return lst

lst1 = [1,3,2,6,8,5,4,2,5]
lst_new = insert_px1(lst1)
print(lst_new)

