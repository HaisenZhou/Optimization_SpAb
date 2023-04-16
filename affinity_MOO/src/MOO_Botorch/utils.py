import numpy as np
import pandas as pd

def prediction_result_output(candidates)->pd.DataFrame:

    '''
    Transform the prediction results from the botorch format
    '''

    output=candidates.param_df.round(4)
    prediction = pd.DataFrame(candidates.model_predictions_by_arm)
    activity = pd.DataFrame([prediction.iloc[0,i] for i in range(len(prediction.columns))])
    activity.rename({'GFP':'GFP_activity','IFN':'IFN_activity'},axis=1, inplace=True)
    covar = pd.DataFrame([prediction.iloc[1,i] for i in range(len(prediction.columns))])
    for i in range(len(prediction.columns)):
        covar.iloc[i]['GFP']= covar.iloc[i]['GFP']['GFP']
        covar.iloc[i]['IFN']= covar.iloc[i]['IFN']['IFN']
    covar.rename({'GFP':'GFP_covar','IFN':'IFN_covar'},axis=1, inplace=True)
    result = pd.concat([output.reset_index(drop=True),activity,covar],axis=1)
    return result