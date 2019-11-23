import pandas as pd
from matplotlib import pyplot as plt

#TODO histogram should have real and fake values. For each column have 2 bars one for positive and one for negative

data_features = pd.read_csv('features.csv', sep=',')
data_labels = pd.read_csv('labels.csv', sep=',')
live_face = []
fake_face = []
avg_live = []
avg_fake = []
mean_live = 0
mean_fake = 0
# print(mean)
for index, row in data_labels.iterrows():
	if (row[1] == 1):
		live_face.append(data_features.iloc[index])
	else:
		fake_face.append(data_features.iloc[index])


df = pd.DataFrame(live_face)
df1 = pd.DataFrame(fake_face)
df.to_csv('list1.csv', index=False)
df1.to_csv('list2.csv', index=False)

live = pd.read_csv('list1.csv', sep=',')
fake = pd.read_csv('list2.csv', sep=',')

mean_live = live.mean()
mean_fake = fake.mean()

avg_live = []
avg_fake = []
for i in range(0, len(mean_live)):
	avg_live.append((i,mean_live[i]))
	# print(i)
for i in range(0, len(mean_fake)):
	avg_fake.append((i,mean_fake[i]))
	
print(avg_live)
print(avg_fake)
	# print(i)
import matplotlib.transforms


items1, counts1 = zip(*avg_live)
items2, counts2 = zip(*avg_fake)
print(items1)
plt.plot(items1+items2, [0.25]*len(items1+items2), visible=False)

trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
trans2 = matplotlib.transforms.Affine2D().translate(+0.2,0)

plt.bar(items1, counts1, label="avg_live", width=0.4, transform=trans1+plt.gca().transData)
plt.bar(items2, counts2, label="avg_fake", width=0.4, transform=trans2+plt.gca().transData)
plt.legend()
plt.show()
