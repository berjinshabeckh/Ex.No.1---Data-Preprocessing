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
1.Importing the libraries
2.Importing the dataset
3.Taking care of missing data
4.Encoding categorical data
5.Normalizing the data
6.Splitting the data into test and train

## PROGRAM:
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
df
#checking for null, duplicates, outliers in lasrt column
df.isnull().sum()

df.duplicated()

df['Exited'].describe()
#normalising data to normal distribution
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['CreditScore','Tenure','Balance',
'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
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
## OUTPUT:

DATASETS AND ITS PROPERTIES:

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/4ad81ba6-1687-420f-b821-8b23852ce90f)

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/f109cbd6-cbb3-4c4b-95fd-231de3a04d66)

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/ad135685-2d3d-4727-8790-98b0cdd9c8b8)

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/08a57ba8-a900-42e1-b9ec-131459313485)

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/62f29f40-68f1-4b44-8874-32310fb300fd)

NORMALISED DATASET:

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/5d8d2d65-1572-4c62-8507-085fae550361)

X AND Y COLOUMN DATA:

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/8468dee9-a151-410c-8026-5ac73b5bcc69)

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/97bc7a97-abc6-4d12-9907-fdf8eb2daa6f)

TRAINING DATA:

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/ce13e36e-c9ac-41e9-898d-956fa7c8947c)

TEST DATA:

![image](https://github.com/MUKESHPARTHASARATHY/Ex.No.1---Data-Preprocessing/assets/119393818/e802f748-bde5-4f78-b6f1-281760d58495)


## RESULT

Thus, the Data preprocessing is performed over a data set successfully.
