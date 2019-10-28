from data_cleaning import get_clean_train_data
from data_cleaning import get_id
from data_cleaning import get_clean_test_data

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from pandas import concat
from pandas import DataFrame
from numpy import exp

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import sqrt


def ensemble(X_train, y_train, X_test):
    # Linear Regression Model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    sp_pred_lr = lr.predict(X_test)

    # Support Vector Regressor
    svr = SVR()
    svr.fit(X_train, y_train)
    sp_pred_svr = svr.predict(X_test)

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    sp_pred_gb = gbr.predict(X_test)

    # Creating an ensemble model
    sp_pred = 0.6 * sp_pred_gb + 0.2 * sp_pred_svr + 0.2 * sp_pred_lr
    return sp_pred


df_train = get_clean_train_data()
df_test = get_clean_test_data()
ids = get_id('test')

print(df_train.isna().any())
print(df_test.isna().any())

y_train = df_train['SalePrice']
X_train = df_train.loc[:, df_train.columns != 'SalePrice']

sp_pred_f = DataFrame(exp(ensemble(X_train, y_train, df_test)))

predictions = concat([ids, sp_pred_f], axis=1)
predictions.to_csv('Predictions.csv', header=['Id', 'SalePrice'], index=False)
print(predictions)



