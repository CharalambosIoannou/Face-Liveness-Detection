import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('features.csv', sep=',')
mean = data.mean()
avg = []
# print(mean)
for i in range(0, len(mean)):
	avg.append((i,mean[i]))
	# print(i)

print(avg)
word, frequency = zip(*avg)
indices = np.arange(len(avg))
plt.bar(indices, frequency, color='r')
plt.xticks(indices, word, rotation='vertical')
plt.tight_layout()
plt.show()
