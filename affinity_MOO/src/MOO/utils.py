import numpy as np
import pandas as pd

def prediction_result_output(candidates)->pd.DataFrame:

    '''
    Transform the prediction results from the botorch format
    '''

    output=candidates.param_df.round(4)
    prediction = pd.DataFrame(candidates.model_predictions_by_arm)
    prediction = prediction.transpose()
    mean_list = []
    covar_list = []
    #Match the signature with the mean and covar
    for i in range(prediction.index.__len__()):
        prediction[0][i] = prediction[0][i]['GPx_metric']
        prediction[1][i] = prediction[1][i]['GPx_metric']['GPx_metric']
        if (output.index[i] in prediction.index[i]):
            mean_list.append(prediction.iloc[i , 0])
            covar_list.append(prediction.iloc[i , 1])
    output['Predict_Mean'] = mean_list
    output['Predict_Covar'] = covar_list
    return output