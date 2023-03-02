import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

#Creating DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/2_linear_reg_multivariate/Exercise/hiring.csv")
print(df)

#Filling NaN values to zeros
df["experience"] = df["experience"].fillna("zero")
print(df.experience)

#Converting word values to numeric with w2n
df.experience = df.experience.apply(w2n.word_to_num)
print(df)

#Filling NaN values in test_score columns
median = df['test_score(out of 10)'].median()
print(median)

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median)
print(df['test_score(out of 10)'])
print(df)

#Creating our model
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

print(reg.coef_)
print(reg.intercept_)

print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))




