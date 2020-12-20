import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
d=load_iris()
print(d)
df=pd.DataFrame(data=d.data,columns=d.feature_names)
print(df)
df.columns=['A','B','C','D']
X=df[['A','B','C']]
y=df['D']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
reg=LinearRegression()
reg.fit(X_train,y_train)
ans=reg.predict([[0.1,0.2,0.3]])
print(ans)
