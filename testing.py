# x = {i:i+1 for i in range(11)}
# for j in x:
#     print(x)
#     x.pop(j+1)
def f1():
    return 1, (2,3), 4

x, y, z = f1()
print(x, y, z)