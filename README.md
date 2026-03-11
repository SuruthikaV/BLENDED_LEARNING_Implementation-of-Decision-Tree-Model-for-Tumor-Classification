# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Load and Prepare Dataset
2.  Split the Dataset
3.  Train the Decision Tree Model
4.  Evaluate and Visualize Results 

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: SURUTHIKA V
RegisterNumber:  212225040441
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('tumor.csv')

print(data.head())
print(data.columns)

x=data.drop(columns=['Class'])
y=data['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Name: SURUTHIKA V")
print("Register Number:212225040441")
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
classification=classification_report(y_test,y_pred)
print("Classification Report:",classification)
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:",confusion)

sns.heatmap(confusion,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
## Output:
<img width="844" height="324" alt="Screenshot 2026-03-11 090106" src="https://github.com/user-attachments/assets/a11e6f56-4bb2-4be0-99a8-c3761f551a6d" />

<img width="831" height="588" alt="Screenshot 2026-03-11 090253" src="https://github.com/user-attachments/assets/b316801f-67d8-423f-afd5-010bb64ddeb7" />


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
