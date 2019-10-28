
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

dataset = 'train'

# Here we obtain the data.
path = 'Datasets/' + dataset + '.csv'
df_train = pd.read_csv(path)

# ---------------------------- #

# We first observe the distribution of the SalePrice, which is the regressand.
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
# Observation: We observe Positive Skewness here

# So, we will know the skewness and Kurtosis.
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# ---------------------------- #


# Selecting the variable 'GrLivArea', i.e. Ground Living Area.
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice')
plt.show()
# Observation: A conical scatter plot, which means it's  heteroscedastic.

# ---------------------------- #

# Selecting the variable 'TotalBsmtSF'
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice')
plt.show()
# Observation: A conical scatter plot, which means it's  heteroscedastic.

# ---------------------------- #

# Selecting the variable 'OverallQual', i.e. Ground Living Area.
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
# Making a Box Whisker plot to observe dependency of quality along with outlier.
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis()
# Observation: There is a strong dependency of OverallQual and SalePrice, and very few outliers.

# ---------------------------- #

# Extracting YearBuilt and plotting Box Whisker diagram for it with Sale Price
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
# Observation: No Strong dependency, and many outliers.

# ---------------------------- #

# Obtain Correlation Matrix for top 10 variables, having correlation with SalePrice dataset.
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
num_variables = 10
cols = corrmat.nlargest(num_variables, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()

# ---------------------------- #

# Observing Missing Data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
# We observe that most of the missing data here is 'NA' which simply means Not Available

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()
#just checking that there's no missing data missing...

# ---------------------------- #
# Here we perform rectification of the data.

# Now we Rectify the Positive Skewness to remove the deviation from normal distribution.

# We are deleting the 2 outliers, we observed from the scatterplot.
df_train.sort_values(by='GrLivArea', ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# We plot the Normal Probability Plot and Histogram for the SalePrice.
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

# Now we apply Log Transformation to rectify the deviation from normality.
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# We can observe the change brought by Log Transformation
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

# We plot the Normal Probability Plot and Histogram for the GrLivArea.
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()

# Now we apply Log Transformation to rectify the deviation from normality.
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# We can observe the change brought by Log Transformation.
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()

# We plot the Normal Probability Plot and Histogram for the TotalBsmtSF.
sns.distplot(df_train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
plt.show()
# Observation: Deviation from Normality causing Hetereoscedacity.
# Problem: Here, we cant apply Log Transformation right away because, There are many values that are 0.
# We can observe this in the scatter plot plotted earlier, SalePrice vs. TotalBsmtSF

# Solution: We can create a new variable 'HasBsmt', which will indicate if there is a basement or not.
# Since, TotalBsmtSF = 0 represents no basement in the house.
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# Now we select TotalBsmtSF values that are non-zero i.e. where HasBsmt = 1.
# We apply Log Transformation to the selected values.
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# We can observe the change brought by Log Transformation.
sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
plt.show()

# Finally observe the scatter plot to observe if normality has homoscedasticity
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.show()

# Finally observe the scatter plot to observe if normality has homoscedasticity
plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])
plt.show()

