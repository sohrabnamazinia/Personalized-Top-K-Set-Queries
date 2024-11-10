import matplotlib.pyplot as plt

# Data for the plots
n_values = [16, 32, 64]
times = [94, 510, 1903]
api_calls = [136, 248, 1008]

# Plotting time based on n
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(n_values, times, marker='o', linestyle='-', color='b')
plt.title('Time vs. n')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.grid(True)

# Plotting time based on number of API calls
plt.subplot(1, 2, 2)
plt.plot(api_calls, times, marker='o', linestyle='-', color='g')
plt.title('Time vs. Number of API Calls')
plt.xlabel('Number of API Calls')
plt.ylabel('Time (seconds)')
plt.grid(True)

plt.tight_layout()
plt.show()
