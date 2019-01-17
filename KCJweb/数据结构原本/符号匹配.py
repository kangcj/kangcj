#2018-12-4
#栈的运用2

from stack import *

def parChecker(symbolString):
	s = Stack()
	balanced = True
	index = 0

	while index < len(symbolString) and balanced:
		symbol = symbolString[index]
		if symbol in '({[':
			s.push(symbol)
		else:							
			if s.isEmpty():             
				balanced = False		
			else:
				top = s.pop()
				if not matches(top,symbol):
					balanced = False

		index += 1	
		
	if balanced and s.isEmpty():		#上边循环结束也可能多出一个前括号，故此处需要判断栈是否为空
		return True
	else:
		return False


def matches(open,close):
	opens = "([{"
	closers = ")]}"
	return opens.index(open) == closers.index(close)


print(parChecker('{[](())}'))
print(parChecker('[(()()))'))
