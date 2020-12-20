import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree

dataset=pd.read_csv("sample1.csv")
print(dataset.shape)
print(dataset.head())
X=dataset.drop('class',axis=1)
y=dataset['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
#classifier=DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3, min_samples_leaf=5)
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

col_name=[]
class_values=[]

for col in X.columns:
    col_name.append(col)
print(col_name)

for val in dataset['class'].unique():
    class_values.append((val))
print(class_values)

fig=plt.figure(figsize=(25,20))
tree.plot_tree(classifier,feature_names=col_name,class_names=class_values,filled=True)
fig.savefig("anshul.png")


