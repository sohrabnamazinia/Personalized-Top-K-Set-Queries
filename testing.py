# x = {i:i+1 for i in range(11)}
# print(max(x))
# for j in x:
#     print(x)
#     x.pop(j+1)
# def f1():
#     return 1, (2,3), 4

# x, y, z = f1()
# print(x, y, z)

# print(tuple(set((1,2,3)) - set((1,2))))
# def haha(x):
#     print(x.pop(2))
#     print(x)
# haha(x)
# print(len(list(range(21))))
import numpy as np

# # Define the size of the matrix
# rows, cols = 3, 3

# # Create a matrix with dtype=object and initialize with (False, False)
# default_value = (False, False)
# matrix = np.full(rows, cols, default_value, dtype=tuple)

# # Print the matrix
# print(matrix)

# element = 5
# list1 = [1, 2, 3, 4, 5]
# list2 = [5, 6, 7, 8, 9]

# # Check if the element is in both lists using set intersection
# is_in_both = {element} <= set(list1) & set(list2)

# print(is_in_both)  # Output: True
# mat = np.ones((3,3))
# print(mat)
# mat2 = np.expand_dims(mat, axis=0)

# print(np.shape(mat2)[0])
# print(mat2)
# mat = np.array()
import itertools
ln = [(0.1, 0.2, 0.3), (0.4,0.5,0.6), (0.7, 0.8, 0.9), (1.0, 1.1, 1.2)]
l = [round(i*0.1, 1) for i in range(1, 4)] + [round(i*0.1, 1) for i in range(2, 5)]
x = list(itertools.product(*ln))
# y = [ln[0]]
# for i in ln:
#     print(i)
#     y = list(itertools.product(y, i))
# print(x)
# flattened_y = [tuple(itertools.chain(*items)) for items in y]
# y_flattened = [item for sublist in y for item in sublist]

# Print the flattened result
# print(y_flattened)
# nl = []
# for i in x:
#     nl.extend(itertools.product(i, (4,5,6,7,8)))
# print(nl)
# Flatten the tuples to get the desired structure
# cartesian_product_2 = list(itertools.product(x, (4,5,6,7)))
# cartesian_product_2 = [item1 + (item2,) for item1, item2 in cartesian_product_2]
# print(cartesian_product_2)
# print(list(itertools.product((1,2,3), (2,3,4), (4,5,6,7))))
# print(tuple([round(i*0.1, 1) for i in range(1, 4)]))
# print((0,1,2,3,4,5)[2:])
# print(tuple(range(1,1)))
common = [2,3,4,5,6,7,8,9]
# for i in range(len(common)):
#     for j in common[i+1:]:
#         print((common[i],j,))
# x= (1,2)
# y = (2,3)
# print(type(x), [y])
# for i in common[1:-1]:
#     print(i)
# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(arr[(0,2)])
dt = {1:1, 2:2, 3:3}
dt2 = {4:4, 5:5, 6:6}
print(list(itertools.product(dt.keys(), dt2.keys())))