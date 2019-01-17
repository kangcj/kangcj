#2018-12-4
#栈的应用

from stack import *


def baseConverter(decNumber, base):
	digits = '0123456789ABCDEF'
	remstack = Stack()

	while decNumber > 0:
		rem = decNumber % base
		remstack.push(rem)
		decNumber = decNumber // base

	newString = ''
	while not remstack.isEmpty():
		newString += digits[remstack.pop()]

	return newString


print(baseConverter(255,2))
print(baseConverter(255,8))
print(baseConverter(255,16))
print(baseConverter(255,3))

for i in range(3):
	print(i)