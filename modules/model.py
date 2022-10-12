from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
import pandas as pd
from .mlp_model import MLPRegression
import time
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR

class SimgleRegression(object):
    def __init__(self):
        self.model_dict = {
            "Ridge": "ridge",
            "XGBoost": "xgb",
            "Random Forest": "forest",
            "SVM": "svr",
            "KNN": "knn",
            "Multilayer Perceptron": "mlp",
            # "Stacking Regressor": "stacking"
        }

    def train(self, x_train,y_train, x_test, y_test):
        eval_cost = pd.DataFrame(columns=['model', 'time_cost'])
        r = 0
        for model_name in self.model_dict.keys():
            train_model_func = getattr(self, self.model_dict[model_name])
            time_cost = train_model_func(x_train,y_train, x_test, y_test)
            eval_cost.loc[r] = [model_name, time_cost]
            r+=1
        return eval_cost

    def test(self,  x_test, y_test):
        result_df = pd.DataFrame({"y_true":y_test})
        for model_name in self.model_dict.keys():
            if model_name == "Multilayer Perceptron":
                mlp = MLPRegression()
                y_pred = mlp.pred(x_test)
            else:
                # 加载机器学习模型 
                ml_model = pickle.load(open("./model/{}.dat".format(self.model_dict[model_name]),"rb"))
                y_pred = ml_model.predict(x_test)
            result_df['pred_{}'.format(self.model_dict[model_name])] = y_pred
        return result_df
    
    def ridge(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        model = make_pipeline(RobustScaler(), linear_model.Ridge(alpha = 1))
        model.fit(x_train,y_train)
        pickle.dump(model,open("./model/ridge.dat","wb"))
        # print('finished train {} model'.format('ridge'))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time

    def xgb(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        model =  make_pipeline(RobustScaler(),XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.82,
             enable_categorical=False, gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.3242105263157895,
             max_delta_step=0, max_depth=2, min_child_weight=4, 
             monotone_constraints='()', n_estimators=192, n_jobs=8,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=0.7736842105263158,
             tree_method='exact', validate_parameters=1, verbosity=None)) 
        model.fit(x_train,y_train)
        pickle.dump(model,open("./model/xgb.dat","wb"))
        # print('finished train {} model'.format('xgb'))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time

    def forest(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        model =  RandomForestRegressor(bootstrap=True, 
                      max_features=30, min_samples_leaf=6, n_estimators=250) 
        model.fit(x_train,y_train)
        pickle.dump(model,open("./model/forest.dat","wb"))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time
    
    def svr(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        model =  make_pipeline(RobustScaler(), SVR(kernel = 'linear'))
        model.fit(x_train,y_train)
        pickle.dump(model,open("./model/svr.dat","wb"))
        # print('finished train {} model'.format('SVM'))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time

    def knn(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        model =  make_pipeline(RobustScaler(), KNeighborsRegressor(n_neighbors=30, weights='distance'))
        model.fit(x_train,y_train)
        pickle.dump(model,open("./model/knn.dat","wb"))
        # print('finished train {} model'.format('KNN'))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time

    def mlp(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        mlp = MLPRegression()
        mlp.train(x_train,y_train, x_test, y_test)
        # print('finished train {} model'.format('Multilayer Perceptron'))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time

    def stacking(self, x_train,y_train, x_test, y_test):
        start_time = time.time()
        estimators = [
            ('ridge', make_pipeline(RobustScaler(),RidgeCV())),
            # ('lasso', make_pipeline(RobustScaler(),LassoCV(random_state=42))),
            # ('knr', make_pipeline(RobustScaler(),KNeighborsRegressor(n_neighbors=20, metric='euclidean'))),
            ('svr', make_pipeline(RobustScaler(),SVR(kernel = 'linear'))),
            ('xgboost', make_pipeline(RobustScaler(),XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.82,
             enable_categorical=False, gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.3242105263157895,
             max_delta_step=0, max_depth=2, min_child_weight=4, 
             monotone_constraints='()', n_estimators=192, n_jobs=8,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=0.7736842105263158,
             tree_method='exact', validate_parameters=1, verbosity=None)) ),
            ('randomForest',RandomForestRegressor(bootstrap=True, 
                      max_features=30, min_samples_leaf=6, n_estimators=250) )
        ]
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(n_estimators=100,
                                                random_state=42)
        )
        model.fit(x_train,y_train)
        pickle.dump(model,open("./model/stacking.dat","wb"))
        # print('finished train {} model'.format('KNN'))
        cost_time = round(time.time() - start_time,2)
        # print("--- cost %s seconds ---" % (cost_time))
        return cost_time