#2018-12-4
#栈的运用1

from stack import *

def parChecker(symbolString):
	s = Stack()
	balanced = True
	index = 0

	while index < len(symbolString) and balanced:
		symbol = symbolString[index]
		if symbol == '(':
			s.push(symbol)
		else:							#其他符号时
			if s.isEmpty():             #此时遍历到后括号，而栈已是空，就没有前括号与之对应弹出，
				balanced = False		#此时即可知不平衡，作为跳出循环的出口
			else:
				s.pop()

		index += 1						
		
	if balanced and s.isEmpty():		#上边循环结束也可能多出一个前括号，故此处需要判断栈是否为空
		return True
	else:
		return False


print(parChecker('(())'))
print(parChecker('(()()))'))