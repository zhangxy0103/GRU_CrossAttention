import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(data_paths):
    # 加载数据
    x1_df = pd.read_csv(data_paths['x1'])
    x2_df = pd.read_csv(data_paths['x2'])

    x1 = x1_df.drop(columns=['class']).values
    x2 = x2_df.drop(columns=['class']).values
    y = x1_df['class'].values

    scaler_x1 = StandardScaler()
    scaler_x2 = StandardScaler()
    x1 = scaler_x1.fit_transform(x1)
    x2 = scaler_x2.fit_transform(x2)

    return x1, x2, y