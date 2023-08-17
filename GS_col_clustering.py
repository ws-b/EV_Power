from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = (r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\merged\01241248782.csv')

data = pd.read_csv(file_path)
# Selecting relevant columns for clustering
selected_columns = ['Power_IV', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc']
data_selected = data[selected_columns]

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Applying KMeans clustering
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42, verbose=1)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Visualizing the clusters using a pairplot
sns.pairplot(data=data, hue='cluster', vars=selected_columns, palette='Set1')
plt.suptitle('KMeans Clustering (3 clusters)', y=1.02)
plt.show()