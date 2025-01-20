#pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('train_data.csv')



df.replace({'France':0,'Spain':1,'Germany':2},inplace=True)

df.replace({'Male':0,'Female':1},inplace=True)


#print(df[['CreditScore','Geography','Age','Tenure','HasCrCard','IsActiveMember']].corr())

plt.hist(df['Age'])
plt.show()
#print(df['Geography'].value_counts())
#print(df.head())