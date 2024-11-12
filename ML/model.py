import pandas as data

dataset = data.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
# dataset.info()

# numerical summary statistics of the dataset
print("describe dataset:\n", dataset.describe())

# categorical summary statistics of the dataset
print("dataset number type: \n",dataset.describe(exclude = 'number'))

# check for null value and sum
print("amount null value: \n",dataset.isnull().sum())

# for imbalance case
print(dataset['Churn'].value_counts())