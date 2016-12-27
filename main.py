# coding:utf8
import pandas as pd
import numpy as np
# because data has 'id' column so remove index, when read in
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# import seaborn as sea

train = pd.read_csv('./input/train.csv',index_col=0)
test = pd.read_csv('./input/test.csv',index_col=0)

###########################################################
# preprocessing

# seperate target (y_train)
y_train = np.log1p(train.pop('SalePrice'))

# combine train and text to preprocessing
all_df = pd.concat((train, test), axis=0)

# turn some column into category according to data_discription.txt
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

# encode categorical column using one-hot
# encode MSSubClass column first
pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass')
# encode others
# Todo:OverallQual column is also categorical, maybe also need to encoded in future
all_dummy_df = pd.get_dummies(all_df)

# preprocess the numerical column
# fill in the missing value using mean value
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)

# standardization for numerical columns
# get all numerical column name first
numeric_cols = all_df.columns[all_df.dtypes != 'object']
# standardization
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

# seperate data to train test
dummy_train_df = all_dummy_df.loc[train.index]
dummy_test_df = all_dummy_df.loc[test.index]
X_train = dummy_train_df.values
X_test = dummy_test_df.values

###########################################################
# modeling
# here use linear regression as benchmark
# lasso ,random forest ,and ensemble method(bagging adaboost XGBoost) will be tested
if __name__ == '__main__':
    # benchmark
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    test_score = np.sqrt(-cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    print np.mean(test_score)

    # lasso
    from sklearn.linear_model import Lasso
    # # best alpha is 0.000579 for lasso
    # alphas = np.logspace(-4, -3, 60)
    # para = {
    #     'alpha':alphas
    # }
    # lasso = Lasso(random_state=2,max_iter=2000)
    # grid = GridSearchCV(estimator=lasso,param_grid=para,scoring='neg_mean_squared_error',n_jobs=-1,cv=5)
    # grid.fit(X_train,y_train)
    # # print grid.cv_results_
    # print grid.best_params_
    # print np.sqrt(-grid.best_score_)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(alphas,np.sqrt(-grid.cv_results_['mean_test_score']))
    # plt.show()

    # random forest
    # from sklearn.ensemble import RandomForestRegressor
    #
    # # first tune max_features
    # # best is 0.37
    # # max_features = np.linspace(.1,1,11)
    #
    # # second tune max_depths and n_estimators
    # # 先粗调再细调
    # #  best is 14
    # max_depths = [13,14,15,16,17]
    # # max_depths = map(lambda x:int(x),max_depths)
    #
    # # best is 700
    # n_estimators = np.linspace(700,1000,5)
    # n_estimators = map(lambda x:int(x),n_estimators)
    #
    # para = {
    #     'max_depth':max_depths,
    #     'n_estimators':n_estimators
    # }
    # rf = RandomForestRegressor(random_state=2,max_features=0.37,n_estimators=700,max_depth=14)
    #
    # if type == 'evaluate':
    #     # 0.1370
    #     test_score = np.sqrt(-cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    #     print np.mean(test_score)
    #
    # grid = GridSearchCV(estimator=rf,param_grid=para,scoring='neg_mean_squared_error',n_jobs=-1,cv=5)
    # grid.fit(X_train, y_train)
    # print grid.best_params_
    # print np.sqrt(-grid.best_score_)

    # if tune one parameter once then can plot it
    # import matplotlib.pyplot as plt
    # plt.plot(min_samples_leaf, np.sqrt(-grid.cv_results_['mean_test_score']))
    # plt.show()


    # bagging
    # from sklearn.ensemble import BaggingRegressor
    # # best n_estimator is 579
    # n_estimators = np.linspace(1,1000,20)
    # n_estimators = map(lambda x:int(x),n_estimators)
    # para = {
    #     'n_estimators':n_estimators
    # }
    # bg = BaggingRegressor(base_estimator=Lasso(alpha=0.000579,random_state=2),random_state=2,n_jobs=-1)
    # grid = GridSearchCV(estimator=bg, param_grid=para, scoring='neg_mean_squared_error', n_jobs=1, cv=5)
    # grid.fit(X_train, y_train)
    # print grid.best_params_
    # print np.sqrt(-grid.best_score_)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(n_estimators, np.sqrt(-grid.cv_results_['mean_test_score']))
    # plt.show()

    # adaboost
    from sklearn.ensemble import AdaBoostRegressor

    # best n_estimators is 10 learning_rate is 1.0e-05
    n_estimators = np.linspace(10,400,5)
    n_estimators = map(lambda x: int(x), n_estimators)
    learning_rate = np.logspace(-5,-2,10)

    para = {
        # 'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'n_estimators':n_estimators
    }

    ada = AdaBoostRegressor(base_estimator=Lasso(alpha=0.000579,random_state=2),random_state=2,learning_rate=1.0e-05,n_estimators=10)
    type = 'evaluate'
    if type == 'evaluate':
        # 0.1346
        test_score = np.sqrt(-cross_val_score(ada, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        print np.mean(test_score)
    else:
        grid = GridSearchCV(estimator=ada,param_grid=para,scoring='neg_mean_squared_error',n_jobs=-1,cv=5)
        grid.fit(X_train, y_train)
        print grid.best_params_
        print np.sqrt(-grid.best_score_)

    # import matplotlib.pyplot as plt
    # plt.plot(learning_rate, np.sqrt(-grid.cv_results_['mean_test_score']))
    # plt.show()

    # xgboost
    # from xgboost import XGBRegressor
    # # best of max_depth is 5
    # max_depths = np.linspace(1,10,10)
    # max_depths = map(lambda x: int(x), max_depths)
    # para = {
    #     'max_depth':max_depths
    # }
    # xgb =  XGBRegressor()
    # grid = GridSearchCV(estimator=xgb, param_grid=para, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    # grid.fit(X_train, y_train)
    # print grid.best_params_
    # print np.sqrt(-grid.best_score_)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(max_depths, np.sqrt(-grid.cv_results_['mean_test_score']))
    # plt.show()