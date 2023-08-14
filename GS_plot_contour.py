import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
def plot_contour(folder_path):
    file_path = folder_path
    data = pd.read_csv(file_path)
    speed = data['speed']
    temp = data['ext_temp']
    Power = data['Power']
    Power_IV = data['Power_IV']

    a_values = np.linspace(-10, 10, 100)
    b_values = np.linspace(-10, 10, 100)
    A, B = np.meshgrid(a_values, b_values)

    Z = np.zeros_like(A)
    c_mean = np.mean(data['Power_fit'])  # c의 평균값

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = objective([A[i, j], B[i, j], c_mean], speed, temp, Power, Power_IV)

    plt.contourf(A, B, Z, 20, cmap='RdGy')
    plt.colorbar()
    plt.title("Objective Function Contour Plot")
    plt.xlabel("a value")
    plt.ylabel("b value")
    plt.show()