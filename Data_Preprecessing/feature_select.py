import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('path')

data = data.drop(columns=['id'])

target_column = 'class'
features_columns = data.columns.difference([target_column])

data.dropna(inplace=True)

X = data.drop(columns=[target_column])
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

importances = xgb_model.feature_importances_
for i, v in enumerate(importances):
    print(f'Feature: {X.columns[i]}, Score: {v}')

threshold = np.mean(importances) * 0.5# 使用均值的一半作为阈值
sfm = SelectFromModel(xgb_model, threshold=threshold, prefit=True)

X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

selected_features_indices = sfm.get_support(indices=True)
selected_features = X.columns[selected_features_indices]

print("Selected features:")
print(selected_features)

today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'selected_features_{today_date}_0.5.txt')

with open(output_file, 'w') as f:
    for feature, score in zip(selected_features, importances[selected_features_indices]):
        f.write(f'{feature}: {score}\n')

print(f'Selected features and their importance scores have been saved to {output_file}')

plt.figure(figsize=(10, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()

plot_file = os.path.join(output_dir, f'feature_importance_{today_date}_0.5.png')
plt.savefig(plot_file)
plt.close()

print(f'Feature importance plot has been saved to {plot_file}')
