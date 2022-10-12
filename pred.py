import pickle
import pandas as pd
from modules.mlp_model import MLPRegression
import time
import os


class FatelWeightPred(object):
    def __init__(self):
        self.model_dict = {
            "Ridge": "ridge",
            "XGBoost": "xgb",
            "Random Forest": "forest",
            "SVM": "svr",
            "Multilayer Perceptron": "mlp",
        }
    def run(self, data_dict):
        x_data = self.feature_process(data_dict)
        result = self.pred(x_data)
        print('result',result)
        return result

    def feature_process(self, data_dict):
        start_time = time.time()
        data_df = pd.DataFrame(data_dict, index=[0])
        # BMI
        data_df.loc[:,'BMI_cat'] = data_df['pre_BMI'].apply(lambda x: 1 if x< 18.5 else 2 if x < 24 else 3 if x < 28 else 4)
        data_df.loc[:,'GWG_cat'] = data_df.apply(lambda row: 1 if ((row['BMI_cat']==1 and (row['GWG']<11)) or (row['BMI_cat']==2 and (row['GWG']<8)) or (row['BMI_cat']==3 and (row['GWG']<7)) or (row['BMI_cat']==4 and (row['GWG']<5))) else 2 if ((row['BMI_cat']==1 and (row['GWG']>=11.5 and row['GWG']<=16)) or (row['BMI_cat']==2 and (row['GWG']>=8 and row['GWG']<=14)) or (row['BMI_cat']==3 and (row['GWG']>=7 and row['GWG']<=11)) or (row['BMI_cat']==4 and (row['GWG']>=5 and row['GWG']<=9))) else 3 if ((row['BMI_cat']==1 and (row['GWG']>16)) or (row['BMI_cat']==2 and (row['GWG']>14)) or (row['BMI_cat']==3 and (row['GWG']>11)) or (row['BMI_cat']==4 and (row['GWG']>9))) else 4, axis=1)
        # 孕周
        data_df.loc[:, 'GA_last'] = data_df['Preg_Days']-data_df['GA_last_inspect']
        data_df.loc[:,'GA'] = data_df['Preg_Days'].apply(lambda x: round(x/7,1))
        data_df.loc[:,'GWG_Inspect_Preg_Days'] = data_df['GWG']/(data_df['GA_last']/data_df['Preg_Days'])
        data_df.loc[:,'GA_last_ul'] = data_df['Preg_Days']-data_df['days_last_ul_to_delivery']
        data_df.loc[:,'BPD_GA_last_ul'] = data_df['BPD']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'HC_GA_last_ul'] = data_df['HC']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'FL_GA_last_ul'] = data_df['FL']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'HL_GA_last_ul'] = data_df['HL']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'AC_GA_last_ul'] = data_df['AC']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'TTD_GA_last_ul'] = data_df['TTD']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'APTD_GA_last_ul'] = data_df['APTD']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df.loc[:,'AFI_Sum_GA'] = data_df['AFI_Sum']/(data_df['GA_last_ul']/data_df['Preg_Days'])
        data_df = data_df[['pre_weight', 'maternal_weight_last','days_last_ul_to_delivery', 'BMI_cat',
                'GWG_cat', 'GA', 'GA_last_inspect', 'GWG_Inspect_Preg_Days','GDM',
                'BPD_GA_last_ul', 'HC_GA_last_ul', 'FL_GA_last_ul', 'HL_GA_last_ul',
                'AC_GA_last_ul', 'TTD_GA_last_ul', 'APTD_GA_last_ul', 'Parity', 'AFI_Sum_GA']]
        end_time = time.time()
        cost_time = end_time - start_time
        print('feature process, cost_time: {}'.format(cost_time))
        return data_df

    def pred(self, x_data):
        result_df = pd.DataFrame()
        for model_name in self.model_dict.keys():
            start_time = time.time()
            if model_name == "Multilayer Perceptron":
                mlp = MLPRegression()
                y_pred = mlp.pred(x_data)
            else:
                # 加载机器学习模型 
                ml_model = pickle.load(open("{}/model/{}.dat".format(os.getcwd(),self.model_dict[model_name]),"rb"))
                y_pred = ml_model.predict(x_data)
            result_df['pred_{}'.format(self.model_dict[model_name])] = y_pred
            end_time = time.time()
            cost_time = end_time - start_time
            # print('model_name: {}, cost_time: {}'.format(model_name, cost_time))
        result_df['pred_vote'] = result_df[['pred_ridge','pred_xgb','pred_forest','pred_svr','pred_mlp']].mean(axis=1)
        return result_df['pred_vote'].values[0]

if __name__ == '__main__':
    start_time = time.time()
    data_dict = {
    'Preg_Days': 274, # 怀孕天数，用于计算孕周
    'pre_weight': 69.0, #孕前体重
    'maternal_weight_last': 81.3, #末次产检体重
    'days_last_ul_to_delivery': 7, # 末次超声距分娩天数
    'pre_BMI': 23.7, #孕前BMI
    'GWG': 12.3, #孕期体重增加，可以通过计算得出
    'GA_last_inspect': 7, #末次产检距分娩天数
    'GDM': 0, # 妊娠期糖尿病，1，0P
    'Parity': 1, #产次，整数
    'BPD': 95.0, # 双顶径，单位mm
    'HC': 335.0, # 头围，单位mm
    'FL': 70.0, # 股骨长，单位mm
    'HL': 60.0, # 肱骨长，单位mm
    'AC': 332.0, # 腹围，单位mm
    'TTD': 106.0, # 腹左右径，单位mm
    'APTD': 106.0, # 腹前后径，单位mm
    'AFI_Sum': 115.0, # 羊水指数合计
                }
    fw_pred = FatelWeightPred()
    fw_pred.run(data_dict)
    end_time = time.time()
    cost_time = end_time - start_time
    print('cost time', cost_time)