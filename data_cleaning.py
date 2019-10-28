from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def categorize_year(df, index):
    years = df.iloc[:, index].copy()
    bins = [1900, 1950, 1960, 1970, 1980, 1990, 2000, np.inf]
    names = ['<1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990',
             '1990-2000', '2000-2010']
    year_range = pd.cut(years, bins, labels=names)
    if year_range.isnull().values.any():
        year_range.replace(np.nan, 'N', inplace=True)
    return year_range.apply(str)


def get_clean_train_data():

    # Here we obtain the data.
    path = 'Datasets/' + 'train' + '.csv'
    df = pd.read_csv(path)

    df['SalePrice'] = np.log(df['SalePrice'])

    df['GrLivArea'] = np.log(df['GrLivArea'])

    df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
    df['HasBsmt'] = 0
    df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

    df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])

    overallqual = df['OverallQual']
    overallqual_d = pd.get_dummies(overallqual).reindex(columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    overallqual_d.drop(overallqual_d.columns[0], inplace=True, axis=1)

    yearbuilt = categorize_year(df, 19)
    yearbuilt_d = pd.get_dummies(yearbuilt).reindex(columns=['<1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990',
             '1990-2000', '2000-2010'])
    yearbuilt_d.drop(yearbuilt_d.columns[0], inplace=True, axis=1)

    neighborhood = df['Neighborhood']
    neighborhood_d = pd.get_dummies(neighborhood).reindex(columns=['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert',
                 'IDOTRR', 'MeadowV', 'Mitchel', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU',
                 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'])
    neighborhood_d.drop(neighborhood_d.columns[0], inplace=True, axis=1)

    external_qual = df['ExterQual']
    external_qual = pd.get_dummies(external_qual).reindex(columns=['Ex', 'Gd', 'TA', 'Fa'])
    external_qual.drop(external_qual.columns[0], inplace=True, axis=1)

    bsmt_qual = df['BsmtQual']
    bsmt_qual.replace(np.nan, 'N')
    bsmt_qual_d = pd.get_dummies(bsmt_qual).reindex(columns=['Ex', 'Gd', 'TA', 'Fa'])
    bsmt_qual_d.drop(external_qual.columns[0], inplace=True, axis=1)

    df_new = pd.concat([neighborhood_d, overallqual, external_qual, yearbuilt_d, bsmt_qual_d,
                        df['GrLivArea'], df['HasBsmt'], df['TotalBsmtSF'], df['SalePrice']], axis=1)

    df_new_d = pd.get_dummies(df_new)

    return df_new_d


def get_clean_test_data():
    path = 'Datasets/' + 'test' + '.csv'
    df = pd.read_csv(path)

    # total = df.isnull().sum().sort_values(ascending=False)
    # percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    # missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    # print(missing_data.head(20))

    df['GrLivArea'] = np.log(df['GrLivArea'])

    df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
    df['HasBsmt'] = 0
    df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean(), inplace=True)

    df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])

    overallqual = df['OverallQual']
    overallqual_d = pd.get_dummies(overallqual).reindex(columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    overallqual_d.drop(overallqual_d.columns[0], inplace=True, axis=1)

    yearbuilt = categorize_year(df, 19)
    yearbuilt_d = pd.get_dummies(yearbuilt).reindex(columns=['<1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990',
                                                             '1990-2000', '2000-2010'])
    yearbuilt_d.drop(yearbuilt_d.columns[0], inplace=True, axis=1)

    neighborhood = df['Neighborhood']
    neighborhood_d = pd.get_dummies(neighborhood).reindex(columns=['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr',
                                                                   'CollgCr', 'Crawfor', 'Edwards', 'Gilbert','IDOTRR',
                                                                   'MeadowV', 'Mitchel', 'NoRidge', 'NPkVill',
                                                                   'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer',
                                                                   'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'])
    neighborhood_d.drop(neighborhood_d.columns[0], inplace=True, axis=1)

    external_qual = df['ExterQual']
    external_qual = pd.get_dummies(external_qual).reindex(columns=['Ex', 'Gd', 'TA', 'Fa'])
    external_qual.drop(external_qual.columns[0], inplace=True, axis=1)

    bsmt_qual = df['BsmtQual']
    bsmt_qual.replace(np.nan, 'N')
    bsmt_qual_d = pd.get_dummies(bsmt_qual).reindex(columns=['Ex', 'Gd', 'TA', 'Fa'])
    bsmt_qual_d.drop(external_qual.columns[0], inplace=True, axis=1)

    df_new = pd.concat([neighborhood_d, overallqual, external_qual, yearbuilt_d, bsmt_qual_d,
                        df['GrLivArea'], df['HasBsmt'], df['TotalBsmtSF']], axis=1)

    df_new_d = pd.get_dummies(df_new)

    return df_new_d


def get_id(dataset):
    path = 'Datasets/' + dataset + '.csv'
    train_data = pd.read_csv(path)
    index = train_data.iloc[:, 0]
    del train_data
    return index
