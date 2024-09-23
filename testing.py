x = {i:i+1 for i in range(11)}
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
mat = np.ones((3,3))
print(mat)
mat2 = np.expand_dims(mat, axis=0)

print(np.shape(mat2)[0])
print(mat2)
# mat = np.array()
