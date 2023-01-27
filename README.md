# :moneybag: Customer classification

In this project, we will focus on predicting the behavior of clients in a Bank dataset. The data set can be found in the website:  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing which is a very good site for open-source dataset. The focus will be put on preprocessing as this dataset has a lot of non numerical values and need many preprocessing techniques to be used properly. 

## Using the Code

The dataset can be found in the file ```bank.csv```, and the python notebook used for its analysis is located through the file ```Training_on_Bank_Dataset.ipynb```.

## Preprocessing

For this project, we will use the most famous data science libraries which are **scikit-learn** and **pandas**. For visualisation, we will use **matplotlib** and **seaborn**

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

We will start the encoding with the easiest features. As there exists a hierarchical structure in the Education and Month columns, we need to encode those values while respecting the order of the values. For instance as a secondary education should be considered longer than a primairy education, the preprocessed data should reflects this link.

```
df["education"].unique()
```
The Ordinal Encoder from **scikit-learn** helps us assign an integer proportinal to the level of education of each client.

```
from sklearn.preprocessing import OrdinalEncoder
cat = ['unknown', 'primary', 'secondary', 'tertiary'] # [0, 1, 2, 3]
encoder = OrdinalEncoder(categories = [cat])
df["education"] = encoder.fit_transform(df[["education"]])
```

For encoding the month feature, we can simply assign the number of the month *(Septr = 1, Oct = 2, ...)* to each corresponding value. 

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



Correlation Matrix:

![image](https://user-images.githubusercontent.com/66775006/215153453-9d995ef8-256e-4310-818b-3fa164690961.png)
