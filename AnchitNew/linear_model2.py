import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("parkinsons_updrs.data.csv")
X1 = data['RPDE']
X2 = data['HNR']

Y2 = data['motor_UPDRS']
X1 = np.array((X1 - X1.min())-(X1.max() - X1.min()))
X2 = np.array((X2 - X2.min())-(X2.max() - X2.min()))
Y2 = np.array((Y2 - Y2.min())-(Y2.max() - Y2.min()))

data['RPDE'] = X1
data['HNR'] = X2
data['motor_UPDRS'] = Y2

plt.scatter(X1, Y2, color='red')
plt.title('RPDE Vs motor_UPDRS', fontsize=14)
plt.xlabel('RPDE', fontsize=14)
plt.ylabel('motor_UPDRS', fontsize=14)
plt.grid(True)
plt.show()
 
plt.scatter(X2, Y2, color='green')
plt.title('motor_UPDRS Vs HNR', fontsize=14)
plt.xlabel('HNR', fontsize=14)
plt.ylabel('motor_UPDRS', fontsize=14)
plt.grid(True)
plt.show()


X = data[['RPDE','HNR']]
Y = data['motor_UPDRS']

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=0)

regrr = linear_model.LinearRegression()
regrr.fit(X_train, Y_train)

print('Intercept: \n', regrr.intercept_)
print('Coefficients: \n', regrr.coef_)


RPDE = 0.7
HNR = 0.11
Y_pred=regrr.predict(X_test)
#print ('Predicted motor UPDRS: \n', regr.predict([[RPDE ,HNR]]))
test_set_rmse = (np.sqrt(mean_squared_error(Y_test,Y_pred)))

test_set_r2 = r2_score(Y_test, Y_pred)
print(test_set_rmse)
print(test_set_r2)

X = sm.add_constant(Xx)
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)