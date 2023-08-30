import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기 및 오차 계산
merge_csv_files(file_lists, folder_path)
data = specific_dataframe
data['Residual'] = data['Power'] - data['Power_IV']
vehicle_type = 'kona EV'

# 필요한 피처들만 선택
features = data.drop(columns=['time', 'Residual', 'Power', 'Power_IV', 'Power_fit', 'A', 'B', 'C', 'D','E', 'chrg_cable_conn', 'pack_current', 'pack_volt'])

# NaN 값 처리 (평균으로 대체)
features = features.fillna(features.mean())

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, data['Residual'], test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 피처 중요도 추출
feature_importances = rf.feature_importances_

# 피처 중요도를 데이터프레임으로 변환
feature_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importances
})

# 중요도 순으로 정렬
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=sorted_feature_importance_df)
plt.title('Feature Importance based on Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True, axis='x')
plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()