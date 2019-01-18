
class Bag(object):

	def __init__(self, maxsize=10):
		self.maxsize = maxsize
		self._items = list()	#容器类型，这里用列表

	def add(self, item):
		if len(self) >= self.maxsize:
			raise Exception('Bag is Full')
		self._items.append(item)

	def remove(self, item):
		self._items.remove(item)

	def __len__(self):				 #魔术方法
		return len(self._items)		 #返回列表长度

	def __iter__(self):				#实现迭代器
		for item in self._items:
			yield item



def test_bag():
	bag = Bag()

	bag.add(1)
	bag.add(2)
	bag.add(3)

	print(len(bag))

if __name__ == '__main__':
	test_bag()



