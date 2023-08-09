import os
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from GS_Fitting import split_data
import matplotlib.pyplot as plt
def linear_func(v, T, a, b, c):
    return a + b * v + c * T

def objective(params, speed, temp, Power, Power_IV):
    a, b, c = params
    fitting_power = Power * linear_func(speed, temp, a, b, c)
    return ((fitting_power - Power_IV) ** 2).sum()
def fit_power(data):
    speed = data['speed']
    temp = data['ext_temp']
    Power = data['Power']
    Power_IV = data['Power_IV']

    # 초기 추정값
    initial_guess = [0, 1, 0]

    # 최적화 수행
    result = minimize(objective, initial_guess, args=(speed, temp, Power, Power_IV))

    a, b, c = result.x

    # 최적화된 Power 값을 별도의 컬럼으로 저장
    data['Power_fit'] = Power * linear_func(speed, temp, a, b, c)

    return a, b, c, data
def fit_parameters(train_files, folder_path):
    a_values = []
    b_values = []
    c_values = []
    for file in tqdm(train_files):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        a, b, c, _ = fit_power(data)  # Make sure fit_power returns a, b, c
        a_values.append(a)
        b_values.append(b)
        c_values.append(c)

    a_avg = sum(a_values) / len(a_values)
    b_avg = sum(b_values) / len(b_values)
    c_avg = sum(c_values) / len(c_values)

    return a_avg, b_avg, c_avg
def apply_fitting(test_files, folder_path, a_avg, b_avg, c_avg):
    for file in tqdm(test_files):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        temp = data['ext_temp']
        data['Power_fit'] = data['Power'] * linear_func(data['speed'], temp, a_avg, b_avg, c_avg)
        data.to_csv(os.path.join(folder_path, file), index=False)
def fitting(file_lists, folder_path):
    # 훈련 및 테스트 데이터 분리
    train_files, test_files = split_data(file_lists)
    # 훈련 데이터에서 a, b, c 파라미터 최적화
    a_avg, b_avg, c_avg = fit_parameters(train_files, folder_path)
    # 테스트 데이터에 대한 Power_fit 계산 및 저장
    apply_fitting(file_lists, folder_path, a_avg, b_avg, c_avg)
    print("Done")

def objective_with_callback(params, speed, temp, Power, Power_IV):
    cost = objective(params, speed, temp, Power, Power_IV)
    costs.append(cost)
    return cost

def fit_power_with_costs(data):
    global costs
    # Reset costs for each new optimization
    costs = []

    speed = data['speed']
    temp = data['ext_temp']
    Power = data['Power']
    Power_IV = data['Power_IV']

    # Initial guess
    initial_guess = [0, 0, 0]

    # Perform optimization
    result = minimize(objective_with_callback, initial_guess, args=(speed, temp, Power, Power_IV))

    a, b, c = result.x

    # Store the optimized Power values in a separate column
    data['Power_fit'] = Power * linear_func(speed, temp, a, b, c)

    return a, b, c, data, costs

def plot_costs(costs):
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Progress')
    plt.show()