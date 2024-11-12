import numpy as np
import random

mean = 0.57
std_dev = 0.1
values_choices = [i * 0.1 for i in range(11)]
value = random.choice(values_choices)
time = np.random.normal(mean, std_dev)
time = min(max(0.1, time), 1.1)
print("Generated time:", time)
