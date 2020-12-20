import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("sample1.csv")
print(df)
X=df.drop('class',axis=1)
print(X)
y=df['class']
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
obj=SVC(kernel='poly',degree=8,C=1,gamma=1)
obj.fit(X_train,y_train)
y_pred=obj.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
