import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import itertools


def avg_and_sub_two_dataframes(df1,df2):
	list_subtracted = []
	for column in df1:
		for column1 in df2:
			a = df1[column] - df2[column1]
			list_subtracted.append(a.mean())
		break
	return list_subtracted

original = pd.read_csv('feature_extraction/features_org.csv')
no_right_eye = pd.read_csv('feature_extraction/features_no_right_eye.csv')
no_left_eye = pd.read_csv('feature_extraction/features_no_left_eye.csv')
no_mouth = pd.read_csv('feature_extraction/features_no_mouth.csv')
no_nose = pd.read_csv('feature_extraction/features_no_nose.csv')
no_both_eyes = pd.read_csv('feature_extraction/features_no_both_eyes.csv')

org_w_right_eye = avg_and_sub_two_dataframes(original,no_right_eye)
print(org_w_right_eye)

org_w_left_eye = avg_and_sub_two_dataframes(original,no_left_eye)
print(org_w_left_eye)

org_w_mouth= avg_and_sub_two_dataframes(original,no_mouth)
print(org_w_mouth)

org_w_nose = avg_and_sub_two_dataframes(original,no_nose)
print(org_w_nose)

org_w_both_eyes = avg_and_sub_two_dataframes(original,no_both_eyes)
print(org_w_both_eyes)

print("max org_w_right_eye:" ,max(org_w_right_eye) )
print("max org_w_left_eye:" ,max(org_w_left_eye) )
print("max org_w_mouth:" ,max(org_w_mouth) )
print("max org_w_nose:" ,max(org_w_nose) )
print("max org_w_both_eyes:" ,max(org_w_both_eyes) )
print("***********************")
print("min org_w_right_eye:" ,min(org_w_right_eye) )
print("min org_w_left_eye:" ,min(org_w_left_eye) )
print("min org_w_mouth:" ,min(org_w_mouth) )
print("min org_w_nose:" ,min(org_w_nose) )
print("min org_w_both_eyes:" ,min(org_w_both_eyes) )

a = ['org_w_right_eye','org_w_left_eye','org_w_mouth','org_w_nose','org_w_both_eyes']
b = [org_w_right_eye,org_w_left_eye,org_w_mouth,org_w_nose,org_w_both_eyes]

for i in range(len(a)):
	items1 = b[i]
	counts1 = [i for i in range(128)]
	
	
	trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
	trans2 = matplotlib.transforms.Affine2D().translate(+0.2,0)
	
	plt.bar(counts1,items1 , width=0.8, transform=trans1+plt.gca().transData)
	plt.title(f"Original- Right Eye. Max val:{round(max(items1),2)} / Min val: {round(min(items1),2)}")
	plt.savefig(f'feature_extraction/imgs/{a[i]}.png', dpi=700)
	plt.legend()
	plt.show()


# This determines features in vector space. Understand which feature is related to where
#TODO check LSTM -> Time series model
