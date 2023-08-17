import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from GS_preprocessing import get_file_list
folder_path = (r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\merged')
file_lists = get_file_list(folder_path)

for file in tqdm(file_lists):
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # "Power"와 "Power_IV" 컬럼의 크로스-코릴레이션 계산
    lags = np.arange(-len(data) + 1, len(data))
    cc = np.correlate(data["Power"], data["Power_IV"], mode="full")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cc)
    plt.title("Cross-correlation between Model Power and Data Power")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

