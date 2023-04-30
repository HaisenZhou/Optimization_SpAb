import numpy as np
import pandas as pd
def get_data(path='../Data/Data.csv'):
    f = open(path, 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    num_feature = RawData.shape[1]-3
    return RawData,num_feature

def prediction_result_output(candidates)->pd.DataFrame:

    '''
    Transform the prediction results from the botorch format
    '''

    output=candidates.param_df.round(4)
    prediction = pd.DataFrame(candidates.model_predictions_by_arm)
    activity = pd.DataFrame([prediction.iloc[0,i] for i in range(len(prediction.columns))])
    activity.rename({'control':'control_activity','target':'target_activity','blank':'blank_activity'},axis=1, inplace=True)
    covar = pd.DataFrame([prediction.iloc[1,i] for i in range(len(prediction.columns))])
    for i in range(len(prediction.columns)):
        covar.iloc[i]['control']= covar.iloc[i]['control']['control']
        covar.iloc[i]['target']= covar.iloc[i]['target']['target']
        covar.iloc[i]['blank']= covar.iloc[i]['blank']['blank']
    covar.rename({'control':'control_covar','target':'target_covar','blank':'blank_covar'},axis=1, inplace=True)
    result = pd.concat([output.reset_index(drop=True),activity,covar],axis=1)
    return result