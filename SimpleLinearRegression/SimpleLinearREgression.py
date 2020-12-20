import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
d=load_diabetes()
df=pd.DataFrame(data=d.data,columns=d.feature_names)

X=df[['age']]
y=df['s6']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
reg=LinearRegression()
reg.fit(X_train,y_train)
ans=reg.predict([[0.05]])
plt.xlabel('Age')
plt.ylabel('s6')
plt.scatter(X,y,marker='+',color='red')
plt.plot(X,reg.predict(X))
plt.show()
print(df)
print(ans)

