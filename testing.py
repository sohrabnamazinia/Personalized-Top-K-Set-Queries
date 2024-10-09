# x = {i:i+1 for i in range(11)}
# for j in x:
#     print(x)
#     x.pop(j+1)
# def f1():
#     return 1, (2,3), 4

# x, y, z = f1()
# print(x, y, z)
# Sample dictionary
my_dict = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5
}

# Group dictionary into sets of 2 key-value pairs
pairs = list(my_dict.items())  # Convert dictionary to a list of key-value tuples
grouped_pairs = [pairs[i:i+2] for i in range(0, len(pairs), 2)]

# Print the results
for group in grouped_pairs:
    print(group)
for i in my_dict:
    print(i)

for i in range(6, 16):
    print(i)