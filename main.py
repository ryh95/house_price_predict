# coding:utf8
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import seaborn as sea


def test_models(type='evaluate'):

    # modeling
    # here use linear regression as benchmark
    # lasso ,random forest ,and ensemble method(bagging adaboost XGBoost) will be tested

    # lasso
    # # best alpha is 0.000579 for lasso

    if type == 'evaluate':
        # 0.1351
        lasso = Lasso(alpha=0.000579, random_state=2, max_iter=2000)
        test_score = np.sqrt(-cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        print np.mean(test_score)
    else:
        alphas = np.logspace(-4, -3, 60)

        para = {
            'alpha': alphas
        }

        lasso = Lasso(random_state=2, max_iter=2000)
        grid = GridSearchCV(estimator=lasso, param_grid=para, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
        grid.fit(X_train, y_train)
        # print grid.cv_results_
        print grid.best_params_
        print np.sqrt(-grid.best_score_)

        import matplotlib.pyplot as plt
        plt.plot(alphas, np.sqrt(-grid.cv_results_['mean_test_score']))
        plt.show()

    # random forest
    if type == 'evaluate':
        # 0.1370
        rf = RandomForestRegressor(random_state=2, max_features=0.37, n_estimators=700, max_depth=14, n_jobs=-1)
        test_score = np.sqrt(-cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        print np.mean(test_score)
    else:
        # first tune max_features
        # best is 0.37
        # max_features = np.linspace(.1,1,11)

        # second tune max_depths and n_estimators
        # 先粗调再细调
        #  best is 14
        max_depths = [13, 14, 15, 16, 17]
        # max_depths = map(lambda x:int(x),max_depths)

        # best is 700
        n_estimators = np.linspace(700, 1000, 5)
        n_estimators = map(lambda x: int(x), n_estimators)

        para = {
            'max_depth': max_depths,
            'n_estimators': n_estimators
        }

        rf = RandomForestRegressor(random_state=2)
        grid = GridSearchCV(estimator=rf, param_grid=para, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
        grid.fit(X_train, y_train)
        print grid.best_params_
        print np.sqrt(-grid.best_score_)

        # if tune one parameter once then can plot it
        # import matplotlib.pyplot as plt
        # plt.plot(min_samples_leaf, np.sqrt(-grid.cv_results_['mean_test_score']))
        # plt.show()

    # bagging
    if type == 'evaluate':
        # 0.1349
        bg = BaggingRegressor(base_estimator=Lasso(alpha=0.000579, random_state=2), random_state=2, n_jobs=-1,
                              n_estimators=369, max_features=0.8)
        test_score = np.sqrt(-cross_val_score(bg, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        print np.mean(test_score)
    else:
        # best n_estimator is 369
        # n_estimators = np.linspace(1,1000,20)
        # n_estimators = map(lambda x:int(x),n_estimators)

        # best is 0.8
        max_features = np.linspace(0.1, 1, 10)

        para = {
            # 'n_estimators':n_estimators
            'max_features': max_features
        }
        bg = BaggingRegressor(base_estimator=Lasso(alpha=0.000579, random_state=2), random_state=2, n_jobs=-1)
        grid = GridSearchCV(estimator=bg, param_grid=para, scoring='neg_mean_squared_error', n_jobs=1, cv=5)
        grid.fit(X_train, y_train)
        print grid.best_params_
        print np.sqrt(-grid.best_score_)

        import matplotlib.pyplot as plt
        plt.plot(max_features, np.sqrt(-grid.cv_results_['mean_test_score']))
        plt.show()

    # adaboost
    if type == 'evaluate':
        # 0.1346
        ada = AdaBoostRegressor(base_estimator=Lasso(alpha=0.000579, random_state=2), random_state=2,
                                learning_rate=1.0e-05,
                                n_estimators=10)
        test_score = np.sqrt(-cross_val_score(ada, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        print np.mean(test_score)
    else:
        # best n_estimators is 10 learning_rate is 1.0e-05
        n_estimators = np.linspace(10, 400, 5)
        n_estimators = map(lambda x: int(x), n_estimators)
        learning_rate = np.logspace(-5, -2, 10)

        para = {
            # 'n_estimators':n_estimators,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators
        }
        ada = AdaBoostRegressor(base_estimator=Lasso(alpha=0.000579, random_state=2), random_state=2)
        grid = GridSearchCV(estimator=ada, param_grid=para, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
        grid.fit(X_train, y_train)
        print grid.best_params_
        print np.sqrt(-grid.best_score_)

        # import matplotlib.pyplot as plt
        # plt.plot(learning_rate, np.sqrt(-grid.cv_results_['mean_test_score']))
        # plt.show()

    # xgboost
    if type == 'evaluate':
        # 0.1265
        xgb = XGBRegressor(max_depth=2, learning_rate=0.2154, n_estimators=257, min_child_weight=3,
                           colsample_bytree=0.5, colsample_bylevel=0.6, reg_alpha=0.1, reg_lambda=0.3594)
        test_score = np.sqrt(-cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        print np.mean(test_score)
    else:
        # best of max_depth is 2 and learning rate is 0.2154
        max_depths = np.linspace(1, 10, 10)
        max_depths = map(lambda x: int(x), max_depths)
        learning_rate = np.logspace(-3, 0, 10)

        # best n_estimators is 257 and gamma is 0
        n_estimators = np.linspace(10, 1000, 5)
        n_estimators = map(lambda x: int(x), n_estimators)
        gamma = [i / 10.0 for i in range(0, 5)]

        # best min_child_weight is 3
        min_child_weight = np.linspace(1, 10, 9)
        min_child_weight = map(lambda x: int(x), min_child_weight)

        # best max_delta_step is 0
        max_delta_step = np.linspace(0, 10, 9)
        max_delta_step = map(lambda x: int(x), max_delta_step)

        # best subsample is 1.0
        subsample = np.linspace(0.1, 1, 10)

        # best colsample_bytree is 0.5 colsample_bylevel is 0.6
        colsample_bytree = np.linspace(0.1, 1, 10)
        colsample_bylevel = np.linspace(0.1, 1, 10)

        # best reg_alpha is 0.1 reg_lambda is 0.3594
        reg_alpha = [0, 1e-5, 1e-2, 0.1, 1, 100]
        reg_lambda = np.logspace(-2, 0, 10)

        # best scale_pos_weight is 1.0
        scale_pos_weight = np.linspace(0.1, 1, 10)

        para = {
            # 'max_depth':max_depths,
            # 'learning_rate':learning_rate
            # 'n_estimators':n_estimators,
            # 'gamma':gamma
            # 'min_child_weight':min_child_weight,
            # 'max_delta_step':max_delta_step
            # 'subsample':subsample
            # 'colsample_bytree':colsample_bytree,
            # 'colsample_bylevel':colsample_bylevel
            # 'reg_alpha':reg_alpha,
            # 'reg_lambda':reg_lambda
            'scale_pos_weight': scale_pos_weight
        }

        xgb = XGBRegressor(max_depth=2, learning_rate=0.2154, n_estimators=257, min_child_weight=3,
                           colsample_bytree=0.5, colsample_bylevel=0.6, reg_alpha=0.1, reg_lambda=0.3594)

        grid = GridSearchCV(estimator=xgb, param_grid=para, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
        grid.fit(X_train, y_train)
        print grid.best_params_
        print np.sqrt(-grid.best_score_)

        # import matplotlib.pyplot as plt
        # plt.plot(scale_pos_weight, np.sqrt(-grid.cv_results_['mean_test_score']))
        # plt.show()

def blending(X_train,X_test,y_train,id_test):

    # blending
    from sklearn.model_selection import KFold
    n_splits= 5

    skf = KFold(n_splits=n_splits,random_state=2)
    clfs = [BaggingRegressor(base_estimator=Lasso(alpha=0.000579, random_state=2), random_state=2, n_jobs=-1,
                             n_estimators=369, max_features=0.8),
            AdaBoostRegressor(base_estimator=Lasso(alpha=0.000579, random_state=2), random_state=2,
                              learning_rate=1.0e-05,
                              n_estimators=10),
            RandomForestRegressor(random_state=2, max_features=0.37, n_estimators=700, max_depth=14, n_jobs=-1),
            XGBRegressor(max_depth=2, learning_rate=0.2154, n_estimators=257, min_child_weight=3,
                         colsample_bytree=0.5, colsample_bylevel=0.6, reg_alpha=0.1, reg_lambda=0.3594)
            ]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_test.shape[0], n_splits))
        for i, (train, test) in enumerate(skf.split(X_train)):
            print "Fold", i
            X_train_tmp = X_train[train]
            y_train_tmp = y_train[train]
            X_test_tmp = X_train[test]
            y_test_tmp = y_train[test]
            clf.fit(X_train_tmp, y_train_tmp)
            y_submission = clf.predict(X_test_tmp)
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_test)
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    alphas = np.logspace(-5, -3, 60)

    para = {
        'alpha': alphas
    }

    lasso = Lasso(random_state=2, max_iter=2000,alpha=0.0001)
    # grid = GridSearchCV(estimator=lasso, param_grid=para, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    # grid.fit(dataset_blend_train, y_train)

    # print grid.cv_results_
    # print grid.best_params_
    # print np.sqrt(-grid.best_score_)

    # import matplotlib.pyplot as plt
    # plt.plot(alphas, np.sqrt(-grid.cv_results_['mean_test_score']))
    # plt.show()

    # result
    # 0.1249
    # test_score = np.sqrt(-cross_val_score(lasso, dataset_blend_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    # print np.mean(test_score)

    lasso.fit(dataset_blend_train,y_train)
    y_predict = lasso.predict(dataset_blend_test)

    #recover y_predict
    y_predict = np.expm1(y_predict)

    submission_df = pd.DataFrame(index=id_test)
    submission_df['SalePrice'] = y_predict
    submission_df.to_csv('submission.csv')

if __name__ == '__main__':

    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    id_train = train.pop('Id')
    id_test = test.pop('Id')

    # preprocessing

    # seperate target (y_train)
    y_train = np.log1p(train.pop('SalePrice'))

    # combine train and text to preprocessing
    all_df = pd.concat((train, test), axis=0)

    # turn some column into category according to data_discription.txt
    all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

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
    dummy_train_df = all_dummy_df.iloc[:len(train)]
    dummy_test_df = all_dummy_df.iloc[len(train):]
    X_train = dummy_train_df.values
    X_test = dummy_test_df.values

    # benchmark
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    test_score = np.sqrt(-cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    print np.mean(test_score)

    # test_models()

    blending(X_train=X_train,X_test=X_test,y_train=y_train,id_test=id_test)
