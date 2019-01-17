#coding:UTF-8
#2018-12-6


# myTree = ['a',['b',['d',[],[]],['e',[],[]]],['c',['f',[],[]],[]]]
# print(myTree)
# print('left subtree = ', myTree[1])
# print('root = ', myTree[0])
# print('right subtree = ', myTree[2])


def BinaryTree(r):
	t = [r, [], []]
	return t

def insertLeft(root, newBranch):
	t = root.pop(1)
	if len(t) > 1:
		root.insert(1, [newBranch, t, []])
	else:
		root.insert(1, [newBranch, [], []])
	return root


def insertRight(root, newBranch):
	t = root.pop(2)
	if len(t) > 1:
		root.insert(2, [newBranch, t, []])
	else:
		root.insert(2, [newBranch, [], []])
	return root

def getRootVal(root):
	return root[0]

def setRootVal(root, newVal):
	root[0] = newVal

def getLeftChild(root):
	return root[1]

def getRightChild(root):
	return root[2]


T = BinaryTree('a')
insertLeft(T, 'd')
insertRight(T, 'f')
insertLeft(T, 'b')
insertRight(T, 'c')

print(getRootVal(T))
print(getLeftChild(T))
print(getRightChild(T))