# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect and load the dataset using the correct file encoding.
2. Preprocess the data by extracting features (v2) and labels (v1).
3. Split the dataset into training and testing sets.
4. Convert text data into numerical vectors using CountVectorizer.
5. Train the SVM model, make predictions, and evaluate accuracy.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Saileshwaran Ganesan
RegisterNumber:  212224230237
*/
```
```
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics
```
```
file = "/content/spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
```
```
data = pd.read_csv( "/content/spam.csv", encoding='windows-1252')
print(data.head())
```
```
print(data.info())
```
```
print(data.isnull().sum())
```
```
x = data["v2"].values 
y = data["v1"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
```
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
```
```
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)
```
```
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```


## Output:
encoding of the CSV file

![image](https://github.com/user-attachments/assets/366944a5-c76a-46fb-896d-b8c6360ce73a)

Display File

![image](https://github.com/user-attachments/assets/e40f4c7d-dbb7-45b8-a19f-5e15a8b51199)

display info

![image](https://github.com/user-attachments/assets/0e4896d2-3e4a-41f5-bb0f-91e4789a6dfa)

Display count of null values

![image](https://github.com/user-attachments/assets/92d94c7e-d728-440a-bfcf-c543c870483e)

Feature and label extraction

![image](https://github.com/user-attachments/assets/d3ae6a47-f681-48ac-801e-a4e79a5bcee5)


ACCURACY 

![image](https://github.com/user-attachments/assets/00c2a68f-5185-4156-8048-a69c3c38bf21)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
