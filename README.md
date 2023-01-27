# :moneybag: Customer classification

In this project, we will focus on predicting the behavior of clients in a Bank dataset. 
Indeed, the classification goal is to predict if the client will subscribe to a term deposit.
The data set can be found in the website:  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing which is a very good site for open-source datasets. The focus will be put on preprocessing as this dataset has a lot of non numerical values and needs many preprocessing techniques to be used properly. 

## Using the Code

The dataset can be found in the file ```bank.csv```, and the python notebook used for its analysis is located through the file ```Training_on_Bank_Dataset.ipynb```.

## Preprocessing

For this project, we will use the most famous data science libraries which are **scikit-learn** and **pandas**. For visualization, we will use **matplotlib** and **seaborn**

```
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
```

We the display the dataset

```
df = pd.read_csv('bank.csv', sep = ';')
df
```

![image](https://user-images.githubusercontent.com/66775006/215159771-846681b0-2f87-4d59-9f1d-ab4e9e6ea9f3.png)

The dataset has a lot of columns that are not directly computable because they use labels. We have to change those chains of characters into numbers so that our model can learn the dataset and hopefully make a good prediction out of it. 

We will first make a verification of our dataset in order to understand what type of preprocessing do we need to apply

### Verification of the dataset

This step is really important and helps us spot any missing values in the data.

``` 
print(df.isnull().sum())
```

```
age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
contact      0
day          0
month        0
duration     0
campaign     0
pdays        0
previous     0
poutcome     0
y            0
dtype: int64
```
Our data set is ready: there are no missing values.
We will now start the encoding step by determining the number of different possibilities in every column of our dataset.

```
df.nunique()
```

```
age            67
job            12
marital         3
education       4
default         2
balance      2353
housing         2
loan            2
contact         3
day            31
month          12
duration      875
campaign       32
pdays         292
previous       24
poutcome        4
y               2
dtype: int64
```

### Ordinal Encoding for Education and Month columns

We will start the encoding with the easiest features. As there exists a hierarchical structure in the Education and Month columns, we need to encode those values while respecting the order of the values. For instance as a secondary education should be considered longer than a primary education, the preprocessed data should reflect this link.

```
df["education"].unique()
```
The Ordinal Encoder from **scikit-learn** helps us assign an integer proportional to the level of education of each client.

```
from sklearn.preprocessing import OrdinalEncoder
cat = ['unknown', 'primary', 'secondary', 'tertiary'] # [0, 1, 2, 3]
encoder = OrdinalEncoder(categories = [cat])
df["education"] = encoder.fit_transform(df[["education"]])
```

For encoding the month feature, we can simply assign the number of the month *(Jan = 1, Feb= 2, ...)* to each corresponding value. 

```
df["month"].unique()
cat = ['sep', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug'] #
encoder = OrdinalEncoder(categories = [cat])
df["month"] = encoder.fit_transform(df[["month"]])
df
```

This gives us the following dataset:

![image](https://user-images.githubusercontent.com/66775006/215163873-d52e39e2-d0db-4cd8-883e-3dd0a2af9fa6.png)


### Label Encoding for binary Columns

For binary features (columns that only take 2 different values), we can use the very simple **LabelEncoder** from the scikit-learn library. Those features are :
- "default" (the customer has credit in default)
- "housing" (the customer has a house)
- "load" (the customer contracted a loan)
- "y" (the target variable)

```
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
binaryVariables = ["default", "housing", "loan", "y"]
for var in binaryVariables:
  df[var] = encoder.fit_transform(df[var])
df
```

![image](https://user-images.githubusercontent.com/66775006/215189475-c381c1d7-b14c-4b22-8b24-171aed14b822.png)

### Splitting dataset

We're not done with the encoding step of the preprocessing. However, as we are going to implement what is called frequency encoding, we must split the dataset before doing this encoding, as we would cheat and transmit information that shouldn't not exist in the test set. 

```
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


### Frequency Encoding for job Column

The purpose of frequency encoding is to assign to each value a number proportional to its frequency in the dataset. We can do this technique when the values of the feature have no order and when the amount of different values is too big for one-hot-encoding to be practical. Moreover, it is important to underline here that we retrieve the frequency information from the train set and apply it to both the train and the test set. This is done so that the distribution of the test set remains unknown. 

```
enc_job = (X_train.groupby('job').size())/len(X_train)

for i in range(len(X_train)):
    X_train.iloc[i,1] = enc_job[X_train.iloc[i,1]]

for i in range(len(X_test)):
    X_test.iloc[i,1] = enc_job[X_test.iloc[i,1]]

X_test
```

![image](https://user-images.githubusercontent.com/66775006/215195738-4a9a2f9c-ee06-40b9-a359-fb1f876754be.png)

### Correlation Matrix

We can now plot the Correlation Matrix in order to display the different correlations between features:

```
import seaborn as sns
train_dataset = X_train.copy()
train_dataset["y"] = y_train
sns.heatmap(train_dataset.corr())
```

![image](https://user-images.githubusercontent.com/66775006/215199771-ff15e373-76c9-45cd-8e3c-7244c56843d9.png)

What is seen is that there exist no strong correlation between our features except for the couple **duration/y** and **previous/pdays**.
We will now begin the scaling of our dataset in order to standardize our data and reduce bias from differences in ranges between features.

### Scaling of relevant columns

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
indexesToScale = [0, 3, 5, 9, 10,11, 12, 13, 14]

scaler.fit(X_train.iloc[:,indexesToScale]) 

X_train.iloc[:,indexesToScale] = scaler.transform(X_train.iloc[:,indexesToScale])
X_test.iloc[:,indexesToScale] = scaler.transform(X_test.iloc[:,indexesToScale])
X_test
```

![image](https://user-images.githubusercontent.com/66775006/215201974-504c1ff1-9a90-4764-b392-53e5182c5588.png)

Now that we have our scaled data, we can do the final step of our preprocessing, which is the **One-Hot-Encoding** of the remaining variables. 

### One Hot Encoding for marital, contact and poutcome Columns

This technique is used when the feature does not take too many different values and that it represents different unordered categories.

```
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2, 8, 15])], remainder='passthrough')

X_train = pd.DataFrame(columnTransformer.fit_transform(X_train))
X_test = pd.DataFrame(columnTransformer.fit_transform(X_test))
X_test
```

![image](https://user-images.githubusercontent.com/66775006/215202958-abd273c6-2e74-4da8-a04e-6ca868b32f1c.png)

Now our data is correctly encoded, scaled and splitted in a Train and Test dataset. It is important that we did the right operations in the right order to prevent the introduction of biases in the dataset.

We will now start the modeling step which will be very short in comparison with all the preprocessing work that we have done so far.

## Model creation and results

### Logistic Regression

We will first try to solve this classification problem with the logistic regression algorithm from **scikit-learn**.

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
Classifier = LogisticRegression()
Classifier.fit(X_train, y_train)

y_pred = Classifier.predict(X_train)

print("Train Accuracy =", accuracy_score(y_pred, y_train))
print("Confusion Matrix:")
print(confusion_matrix(y_pred, y_train), '\n')

y_pred = Classifier.predict(X_test)

print("Test Accuracy =", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_pred, y_test))
```

```
Train Accuracy = 0.903
Confusion Matrix:
[[3144  288]
 [  63  121]] 

Test Accuracy = 0.893
Confusion Matrix:
[[769  73]
 [ 24  39]]
```

The model yields very good results in terms of accuracy *(~90% accuracy)* .However, we understand that the results are not very good.
Indeed as the dataset is unbalanced (the vast majority of customers didn't want to subscribe to the deposit), the number of **false positives** and **false negatives** is far greater than the number of **true negatives**. Lets try with another model: **random forest**.

### Random Forest

```
from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
Classifier.fit(X_train, y_train)

y_pred = Classifier.predict(X_train)

print("Train Accuracy =", round(accuracy_score(y_pred, y_train), 2))
print("Confusion Matrix:")
print(confusion_matrix(y_pred, y_train), '\n')

y_pred = Classifier.predict(X_test)

print("Test Accuracy =", round(accuracy_score(y_pred, y_test), 2))
print("Confusion Matrix:")
print(confusion_matrix(y_pred, y_test))
```

```
Train Accuracy = 1.0
Confusion Matrix:
[[3207    0]
 [   0  409]] 

Test Accuracy = 0.89
Confusion Matrix:
[[776  80]
 [ 17  32]]
 ```
 
As random forest is a much stronger algorithm, its capacity allows for an amazing 100% accuracy on the train set. However, results are not really improved on the test set when compared with the logistic regression model, and it is likely that our model overfitted the data. 
 
This is not really important as the goal of this project was not to achieve a perfect result, but rather to demonstrate the whole working process of a Data Scientist, from the preprocessing to the modeling of the AI. 

## Conclusion

This project is interesting from the preprocessing point of vue. 
Indeed, we applied various preprocessing techniques in order to extract useful information from the feature's labels. 

So far, we used:
- **Label Encoding**
- **One Hot Encoding**
- **Frequency Encoding**
- **Ordinal Encoding**


