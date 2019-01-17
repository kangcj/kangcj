#encoding:UTF-8
#栈实现

class Stack() :

	def __init__(self):
		self.items = []

	def isEmpty(self):
		return self.items == []

	def push(self, item):
		self.items.append(item)

	def pop(self):
		return self.items.pop()

	def peek(self):
		return self.items[len(self.items)-1]
	
	def size(self):
		return len(self.items)	


def test_stack():
	s1 = stack()
	s1.push(1)
	s1.push('haha')
	s1.push('lala')

	print(s1.isEmpty())
	print(s1.pop())
	print(s1.peek())
	print(s1.size())

	



if __name__ == '__main__':
	test_stack() 


#https://blog.csdn.net/brucewong0516/article/details/79120841 关于单个下划线和两个下划线的区别解释