import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('D:/Python-master/DatasetFirst.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(x)

print('-----------------------------------------------------')

print(y)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x.iloc[:, 1:3])
x.iloc[:, 1:3] = imputer.transform(x.iloc[:, 1:3])

print(x)

print('-----------------------------------------------------')


from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()
x.iloc[:,0] = labelencoder_x.fit_transform(x.iloc[:,0])

print(x)

print('-----------------------------------------------------')


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

onehotencoder_x = OneHotEncoder()
x = onehotencoder_x.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


print(x)

print('-----------------------------------------------------')

print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

print(x)

print('-----------------------------------------------------')

print(y)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print(X_train)
print(Y_train)

print('-----------------------------------------------------')

print(X_test)
print(Y_test)

print('-----------------------------------------------------')























