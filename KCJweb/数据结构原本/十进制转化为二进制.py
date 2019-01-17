#2018-12-4
#栈的应用3

from stack import *


def divideBy2(decNumber):
	remstack = Stack()

	while decNumber > 0:
		rem = decNumber % 2
		remstack.push(rem)
		decNumber = decNumber // 2

	binstring = ''
	while not remstack.isEmpty():
		binstring += str(remstack.pop())

	return binstring



print(divideBy2(8))
print(divideBy2(255))