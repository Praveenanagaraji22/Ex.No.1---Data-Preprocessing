# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#read the dataset
df=pd.read_csv('Churn_Modelling data.csv')
df
#drop unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
#checking for null, duplicates, outliers in lasrt column
df.isnull().sum()

df.duplicated()

df['Exited'].describe()
#normalising data to normal distribution
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
df2
#split dataset
x=df2.iloc[:,:-1].values #all rows from all except last column
x
y=df2.iloc[:,-1].values #all rows from only last column
y
##creating training and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
```
## OUTPUT:
## Dataset and Its Properties
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/cda1de41-ae0f-4dc8-9e5c-c16c9d7b237a)
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/8f0a6b7c-a8c5-4066-8aa4-f568ac17be0a)
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/2f4303db-cba2-4a1a-ae04-fd7f0df2bea3)

## Normalised Dataset
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/2b4ff128-8be3-477e-af45-1fba0df39c6a)

## X and Y Column Data
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/6cdc4fa5-ddb4-4208-86a9-2d3c7b2336ae)
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/c90404d9-73b2-43a7-95e5-66c02f5cb1df)

## Training Data
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/7183c948-c20a-4d84-ba42-c5b66020191a)

## Test Data
![image](https://github.com/Praveenanagaraji22/Ex.No.1---Data-Preprocessing/assets/119393514/673a2231-cb20-4cf0-8938-a92483a8017f)

## RESULT:
Thus, the Data preprocessing is performed over a data set successfully.
