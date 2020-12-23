import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

dataset=pd.read_csv("sample1.csv")
print(dataset.head())
x=dataset.drop('class',axis=1)
y=dataset['class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)

#FEATURE SCALING

scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)

#PRINT X _TRAIN

classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))



#JUDGING VALUE OF K BY ERROR METHOD

error=[]


for i in range(1,40):
    kmn=KNeighborsClassifier(n_neighbors=i)
    kmn.fit(x_train,y_train)
    pred_i=kmn.predict(x_test)
    error.append(np.mean(pred_i!=y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
  

