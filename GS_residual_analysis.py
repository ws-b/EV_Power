import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 오차 계산
file_path = (r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\merged\01241248726.csv')
data = pd.read_csv(file_path)
data['Residual'] = data['Power'] - data['Power_IV']
type = 2
if type == 1:
    vehicle_type = 'ioniq 5'
elif type == 2:
    vehicle_type = 'kona EV'
# Speed vs Residuals
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='speed', y='Residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship between Speed and Residuals')
plt.xlabel('Speed (mps)')
plt.ylabel('Residual (Power - Power_IV)')
plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
plt.grid(True)
plt.show()

# Acceleration vs Residuals
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='acceleration', y='Residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship between Acceleration and Residuals')
plt.xlabel('Acceleration (mps^2)')
plt.ylabel('Residual (Power - Power_IV)')
plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
plt.grid(True)
plt.show()

# External Temperature vs Residuals
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='ext_temp', y='Residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship between External Temperature and Residuals')
plt.xlabel('External Temperature (°C)')
plt.ylabel('Residual (Power - Power_IV)')
plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
plt.grid(True)
plt.show()