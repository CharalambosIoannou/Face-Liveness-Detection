import pandas as pd

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
