import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 오차 계산
file_path = (r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\merged\01241248782.csv')
data = pd.read_csv(file_path)
data['Residual'] = data['Power'] - data['Power_IV']

# 2. 속도와 오차 관계 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='speed', y='Residual', alpha=0.5)
plt.title('Relationship between Speed and Residuals')
plt.xlabel('Speed (mps)')
plt.ylabel('Residual (Power - Power_IV)')
plt.grid(True)
plt.show()

# 3. 가속도와 오차 관계 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='acceleration', y='Residual', alpha=0.5)
plt.title('Relationship between Acceleration and Residuals')
plt.xlabel('Acceleration (mps^2)')
plt.ylabel('Residual (Power - Power_IV)')
plt.grid(True)
plt.show()

# 4. 외부 온도와 오차 관계 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='ext_temp', y='Residual', alpha=0.5)
plt.title('Relationship between External Temperature and Residuals')
plt.xlabel('External Temperature (°C)')
plt.ylabel('Residual (Power - Power_IV)')
plt.grid(True)
plt.show()