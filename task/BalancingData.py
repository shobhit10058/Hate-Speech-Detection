from sklearn.utils import resample
import pandas as pd

def balanceData(Train_data):
    train_majority = Train_data[Train_data['hateful']==0]
    train_minority = Train_data[Train_data['hateful']==1]
    train_minority_upsampled = resample(train_minority, 
                                    replace=True,    
                                    n_samples=len(train_majority),   
                                    random_state=123)
    train_upsampled = pd.concat([train_minority_upsampled, train_majority])
    return train_upsampled
