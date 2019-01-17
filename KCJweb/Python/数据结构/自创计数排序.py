# 2018-11-4
lst = [1,3,2,67,43,23,44,33,55,77,23,31,42]

for i in range(len(lst)-1):
	if lst[i] > lst[i+1]:
		lst[i], lst[i+1] = lst[i+1], lst[i]
max_list = lst[len(lst)-1]

lst1 = [0] * (max_list + 1)
for j in lst:
	lst1[j] += 1

print(lst1)

lst2 = []
pre = 0
for k in range(len(lst1)):
	# for v in range(lst1[k]):
	# 	lst2.insert(pre, k)
	# 	pre += 1
	# 	# if v == lst1[k] - 1:
	# 	# 	pre += lst1[k]
	lst2.extend([k] * lst1[k])
print(lst2)