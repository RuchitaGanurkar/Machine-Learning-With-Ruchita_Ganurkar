import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/Python-master/DatasetSecond.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, :-1]

print(x)

print('-----------------------------------------------------')

print(y)

print('-----------------------------------------------------')


from sklearn.model_selection  import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)
print(X_train)

print(X_test)

print(Y_train)

print(Y_test)


print('-----------------------------------------------------')


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print(X_train)

print(Y_train)

print('-----------------------------------------------------')

Y_predict = regressor.predict(X_test)

print(Y_predict)

print(Y_test)

print('-----------------------------------------------------')



# Graph For Tested Data

plt.scatter(X_test, Y_test, color = "blue")
plt.plot(X_test, regressor.predict(X_test), color = "black")
plt.title("Tested Data Of Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.show()
















