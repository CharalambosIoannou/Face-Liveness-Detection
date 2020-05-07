"""
To run file: type in a terminal python histogram_subtraction.py
The figures are saved to feature_extraction/imgs folder

"""

import matplotlib



def avg_and_sub_two_dataframes(df1,df2):
	list_subtracted = []
	for column in df1:
		for column1 in df2:
			a = df1[column] - df2[column1]
			list_subtracted.append(a.mean())
		break
	return list_subtracted

import pandas as pd
from matplotlib import pyplot as plt



original = pd.read_csv('../feature_extraction/features_Detectedface.csv')
no_right_eye = pd.read_csv('../feature_extraction/features_face_no_right_eye.csv')
no_left_eye = pd.read_csv('../feature_extraction/features_face_no_left_eye.csv')
no_mouth = pd.read_csv('../feature_extraction/features_face_no_mouth.csv')
no_nose = pd.read_csv('../feature_extraction/features_face_no_nose.csv')
no_both_eyes = pd.read_csv('../feature_extraction/features_face_both_eyes.csv')

labels_original = pd.read_csv('../feature_extraction/labels_Detectedface.csv')
labels_no_right_eye = pd.read_csv('../feature_extraction/labels_face_no_right_eye.csv')
labels_no_left_eye = pd.read_csv('../feature_extraction/labels_face_no_left_eye.csv')
labels_no_mouth = pd.read_csv('../feature_extraction/labels_face_no_mouth.csv')
labels_no_nose = pd.read_csv('../feature_extraction/labels_face_no_nose.csv')
labels_no_both_eyes = pd.read_csv('../feature_extraction/labels_face_both_eyes.csv')


live_face = []
fake_face = []

live_face_right_eye = []
fake_face_right_eye = []

live_face_left_eye = []
fake_face_left_eye = []

live_face_no_nose = []
fake_face_no_nose = []

live_face_no_mouth = []
fake_face_no_mouth = []

live_face_no_both_eyes = []
fake_face_no_both_eyes = []


avg_live = []
avg_fake = []
mean_live = 0
mean_fake = 0
# print(mean)
for index, row in labels_original.iterrows():
	if (row[1] == 1):
		live_face.append(original.iloc[index])
	else:
		fake_face.append(original.iloc[index])
		
for index, row in labels_no_right_eye.iterrows():
	if (row[1] == 1):
		live_face_right_eye.append(no_right_eye.iloc[index])
	else:
		fake_face_right_eye.append(no_right_eye.iloc[index])
		
for index, row in labels_no_left_eye.iterrows():
	if (row[1] == 1):
		live_face_left_eye.append(no_left_eye.iloc[index])
	else:
		fake_face_left_eye.append(no_left_eye.iloc[index])


for index, row in labels_no_nose.iterrows():
	if (row[1] == 1):
		live_face_no_nose.append(no_nose.iloc[index])
	else:
		fake_face_no_nose.append(no_nose.iloc[index])
		
for index, row in labels_no_mouth.iterrows():
	if (row[1] == 1):
		live_face_no_mouth.append(no_mouth.iloc[index])
	else:
		fake_face_no_mouth.append(no_mouth.iloc[index])

for index, row in labels_no_both_eyes.iterrows():
	if (row[1] == 1):
		live_face_no_both_eyes.append(no_both_eyes.iloc[index])
	else:
		fake_face_no_both_eyes.append(no_both_eyes.iloc[index])
		

# print("len: " , len(live_face))
# print("len: " , len(live_face_left_eye))
# print("len: " , len(live_face_right_eye))
# print("aaaaaa: " , live_face)
# # print("aaaaaa: " , live_face_left_eye)
# # print("aaaaaa: " , live_face_right_eye)
#
#
# print("len: " , live_face_right_eye == live_face)
# print("len: " , live_face_left_eye == live_face)



live = pd.DataFrame(live_face)
fake = pd.DataFrame(fake_face)

live_right_eye = pd.DataFrame(live_face_right_eye)
fake_right_eye = pd.DataFrame(fake_face_right_eye)

live_left_eye = pd.DataFrame(live_face_left_eye)
fake_left_eye = pd.DataFrame(fake_face_left_eye)

live_no_nose = pd.DataFrame(live_face_no_nose)
fake_no_nose = pd.DataFrame(fake_face_no_nose)

live_no_mouth = pd.DataFrame(live_face_no_mouth)
fake_no_mouth = pd.DataFrame(fake_face_no_mouth)

live_no_both_eyes= pd.DataFrame(live_face_no_both_eyes)
fake_no_both_eyes= pd.DataFrame(fake_face_no_both_eyes)

# df.to_csv('list1.csv', index=False)
# df1.to_csv('list2.csv', index=False)
#
# live = pd.read_csv('list1.csv', sep=',')
# fake = pd.read_csv('list2.csv', sep=',')


mean_live = live.mean()
mean_fake = fake.mean()

mean_live_right_eye = live_right_eye.mean()
mean_fake_right_eye = fake_right_eye.mean()

mean_live_left_eye = live_left_eye.mean()
mean_fake_left_eye = fake_left_eye.mean()

mean_live_no_nose = live_no_nose.mean()
mean_fake_no_nose = fake_no_nose.mean()

mean_live_no_mouth = live_no_mouth.mean()
mean_fake_no_mouth = fake_no_mouth.mean()

mean_live_no_both_eyes = live_no_both_eyes.mean()
mean_fake_no_both_eyes = fake_no_both_eyes.mean()


a = [mean_live_left_eye,mean_live_right_eye,mean_live_no_nose,mean_live_no_mouth,mean_live_no_both_eyes]
b = [mean_fake_left_eye,mean_fake_right_eye,mean_fake_no_nose,mean_fake_no_mouth,mean_fake_no_both_eyes]
c = ['No Left Eye','No Right Eye', 'No nose','No mouth', 'No both eyes']

d = []
d1 = []
sum_abs = []
for k in range(0,len(a)):
	avg_live = []
	avg_fake = []
	
	for i in range(0, len(mean_live)):
		sub = mean_live[i] - a[k][i]
		avg_live.append((i,sub))
		
	for i in range(0, len(mean_fake)):
		sub = mean_fake[i] - b[k][i]
		avg_fake.append((i,sub))
	
	print(avg_live)
	print(avg_fake)
	differences = []
	for p in range(0,len(avg_live)):
		diff = avg_live[p][1] - avg_fake[p][1]
		differences.append(abs(diff))
	print(differences)
	d.append((c[k], max(differences)))
	d1.append((c[k], min(differences)))
	sum_abs.append((c[k],sum(differences)))
	items1, counts1 = zip(*avg_live)
	items2, counts2 = zip(*avg_fake)
	plt.plot(items1+items2, [0.01]*len(items1+items2), visible=False)
	
	trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
	trans2 = matplotlib.transforms.Affine2D().translate(+0.2,0)
	plt.title("Original-" + c[k])
	plt.bar(items1, counts1, label="avg_live", width=0.4, transform=trans1+plt.gca().transData)
	plt.bar(items2, counts2, label="avg_fake", width=0.4, transform=trans2+plt.gca().transData)
	plt.legend()
	#
	# plt.show()
	plt.savefig(f'../feature_extraction/imgs/Original-{c[k]}.png', dpi=700)
	plt.clf()
	
	names = [j for j in range(0,128)]
	values = differences


	plt.title("Original-" + c[k] + "(Real - Fake)")
	plt.bar(names, values)
	plt.savefig(f'../feature_extraction/imgs/Original-{c[k]}(Real-Fake).png', dpi=700)
	plt.clf()
plt.rcParams.update({'font.size': 25})
# print(d)
# print(d1)
names1 = []
l1 = []

print("sum_abs: ",sum_abs)
for i in range(0,len(sum_abs)):
	l1.append(sum_abs[i][1])
	names1.append(sum_abs[i][0])
# highest = max(l1)
# print(highest)


plt.title("Sum of absolute Differences")
plt.bar(names1, l1)
plt.savefig(f'../feature_extraction/imgs/sum_abs_diff.png', dpi=700)
plt.show()
