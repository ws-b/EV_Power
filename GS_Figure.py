import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import plotly.graph_objects as go
from GS_Functions import get_vehicle_files, compute_rrmse, compute_rmse, compute_mape
from scipy.interpolate import griddata
from scipy.stats import linregress
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate, add_predicted_power_column as only_xgb_add_predicted_power_column
from GS_Train_LinearR import cross_validate as lr_cross_validate, add_predicted_power_column as lr_add_predicted_power_column
from GS_Train_LightGBM import cross_validate as lgbm_cross_validate, add_predicted_power_column as lgbm_add_predicted_power_column

# Function to get file lists for each vehicle based on vehicle_dict
def get_file_lists(directory):
    vehicle_file_lists = {vehicle: [] for vehicle in vehicle_dict.keys()}

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Match filename with vehicle IDs
            for vehicle, ids in vehicle_dict.items():
                if any(vid in filename for vid in ids):
                    vehicle_file_lists[vehicle].append(os.path.join(directory, filename))
                    break  # Stop searching once a match is found

    return vehicle_file_lists

# Example usage of the function
directory = r"D:\SamsungSTF\Processed_Data\TripByTrip"  # Change this to your actual directory
vehicle_file_lists = get_file_lists(directory)
#save_path
fig_save_path = r"C:\Users\BSL\Desktop\Figures"
# Now you can use the file lists for specific vehicles like EV6 and Ioniq5 in your figure1 function
file_lists_ev6 = vehicle_file_lists['EV6']
file_lists_ioniq5 = vehicle_file_lists['Ioniq5']
file_lists_konaEV = vehicle_file_lists['KonaEV']
file_lists_ioniq6 = vehicle_file_lists['Ioniq6']
file_lists_niroEV = vehicle_file_lists['NiroEV']
file_lists_GV60 = vehicle_file_lists['GV60']
def figure1(file_lists_ev6, file_lists_ioniq5):
    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [4.667, 5.371],
        'EV6': [4.524, 5.515]
    }

    # Function to process energy data
    def process_energy_data(file_lists):
        dis_energies_data = []
        for file in tqdm(file_lists):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            v = data['speed']
            v = np.array(v)

            distance = v * t_diff
            total_distance = distance.cumsum()

            Power_data = np.array(data['Power_data'])
            energy_data = Power_data * t_diff / 3600 / 1000

            dis_data_energy = ((total_distance[-1] / 1000) / (energy_data.cumsum()[-1])) if energy_data.cumsum()[
                                                                                              -1] != 0 else 0
            dis_energies_data.append(dis_data_energy)

        return dis_energies_data

    # Function to add official efficiency range for a specific car
    def add_efficiency_lines(selected_car):
        if selected_car in official_efficiency:
            eff_range = official_efficiency[selected_car]
            if len(eff_range) == 2:
                plt.fill_betweenx(plt.gca().get_ylim(), eff_range[0], eff_range[1], color='green', alpha=0.3, hatch='/')
                plt.text(eff_range[1] + 0.15, plt.gca().get_ylim()[1] * 0.8, 'Official Efficiency',
                         color='green', fontsize=12, alpha=0.7)

    # Process the data for EV6 and Ioniq5
    dis_energies_ev6 = process_energy_data(file_lists_ev6)
    dis_energies_ioniq5 = process_energy_data(file_lists_ioniq5)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Plot for EV6
    plt.sca(ax1)  # Set current axis to ax1
    mean_value_ev6 = np.mean(dis_energies_ev6)
    sns.histplot(dis_energies_ev6, bins='auto', color='gray', kde=False)
    plt.axvline(mean_value_ev6, color='red', linestyle='--')
    plt.text(mean_value_ev6 + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value_ev6:.2f}', color='red', fontsize=12, alpha=0.7)
    plt.xlabel('Official Efficiency in km/kWh')
    plt.xlim((0, 15))
    plt.ylabel('Number of trips')
    ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes, size=14, weight='bold', ha='left')  # Move (a) to top-left
    ax1.set_title("Energy Consumption Distribution : EV6", pad=10)  # Title below (a)
    add_efficiency_lines('EV6')
    plt.grid(False)

    # Plot for Ioniq5
    plt.sca(ax2)  # Set current axis to ax2
    mean_value_ioniq5 = np.mean(dis_energies_ioniq5)
    sns.histplot(dis_energies_ioniq5, bins='auto', color='gray', kde=False)
    plt.axvline(mean_value_ioniq5, color='red', linestyle='--')
    plt.text(mean_value_ioniq5 + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value_ioniq5:.2f}', color='red', fontsize=12, alpha=0.7)
    plt.xlabel('Official Efficiency in km/kWh')
    plt.xlim(0, 15)
    plt.ylabel('Number of trips')
    ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes, size=14, weight='bold', ha='left')  # Move (b) to top-left
    ax2.set_title("Energy Consumption Distribution : Ioniq5", pad=10)  # Title below (b)
    add_efficiency_lines('Ioniq5')
    plt.grid(False)

    # Save the figure with dpi 300
    save_path = os.path.join(fig_save_path, 'figure1_ev6_ioniq5.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def figure3(vehicle_files_ev6, vehicle_files_ioniq5):
    vehicle_file_sizes = [50, 1000]
    l2lambda = {'EV6': [], 'Ioniq5': []}
    results_dict = {'EV6': {}, 'Ioniq5': {}}

    # Function to process lambda and cross-validation results
    def process_lambda_cross_validation(vehicle_files, selected_car):
        l2lambda[selected_car] = []
        results_dict[selected_car] = {}
        max_samples = len(vehicle_files[selected_car])

        filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]

        # Perform cross-validation for Hybrid Model (XGBoost)
        _, _, lambda_XGB = xgb_cross_validate(vehicle_files, selected_car, None, None, save_dir=None)
        l2lambda[selected_car].append({
            'model': 'Hybrid Model(XGBoost)',
            'lambda': lambda_XGB
        })

        # Perform cross-validation for Only ML (XGBoost)
        _, _, lambda_ML = only_xgb_validate(vehicle_files, selected_car, None, None, save_dir=None)
        l2lambda[selected_car].append({
            'model': 'Only ML(XGBoost)',
            'lambda': lambda_ML
        })

        # Loop over vehicle file sizes
        for size in filtered_vehicle_file_sizes:
            if size not in results_dict[selected_car]:
                results_dict[selected_car][size] = []

            # Define number of samplings based on size
            if size < 20:
                samplings = 200
            elif 20 <= size < 50:
                samplings = 10
            elif 50 <= size <= 100:
                samplings = 5
            else:
                samplings = 1

            for _ in range(samplings):
                sampled_files = random.sample(vehicle_files[selected_car], size)
                sampled_vehicle_files = {selected_car: sampled_files}

                # Physics-based model RRMSE and MAPE
                mape_physics = compute_mape(sampled_vehicle_files, selected_car)
                rrmse_physics = compute_rrmse(sampled_vehicle_files, selected_car)
                if rrmse_physics is not None:
                    results_dict[selected_car][size].append({
                        'model': 'Physics-Based',
                        'rrmse': [rrmse_physics],
                        'mape': [mape_physics]
                    })

                # Hybrid Model (XGBoost)
                results, _, _ = xgb_cross_validate(sampled_vehicle_files, selected_car, lambda_XGB, None, save_dir=None)
                if results:
                    mape_values = [mape for _, _, _, mape in results]
                    rrmse_values = [rrmse for _, rrmse, _, _ in results]
                    results_dict[selected_car][size].append({
                        'model': 'Hybrid Model(XGBoost)',
                        'rrmse': rrmse_values,
                        'mape': mape_values
                    })

                # Hybrid Model (Linear Regression)
                results, _ = lr_cross_validate(sampled_vehicle_files, selected_car, None, save_dir=None)
                if results:
                    mape_values = [mape for _, _, _, mape in results]
                    rrmse_values = [rrmse for _, rrmse, _, _ in results]
                    results_dict[selected_car][size].append({
                        'model': 'Hybrid Model(Linear Regression)',
                        'rrmse': rrmse_values,
                        'mape': mape_values
                    })

                # Only ML (XGBoost)
                results, _, _ = only_xgb_validate(sampled_vehicle_files, selected_car, lambda_ML, None, save_dir=None)
                if results:
                    mape_values = [mape for _, _, _, mape in results]
                    rrmse_values = [rrmse for _, rrmse, _, _ in results]
                    results_dict[selected_car][size].append({
                        'model': 'Only ML(XGBoost)',
                        'rrmse': rrmse_values,
                        'mape': mape_values
                    })

        return l2lambda[selected_car], results_dict[selected_car]

    # Process results for EV6
    l2lambda_ev6, results_ev6 = process_lambda_cross_validation(vehicle_files_ev6, 'EV6')

    # Process results for Ioniq5
    l2lambda_ioniq5, results_ioniq5 = process_lambda_cross_validation(vehicle_files_ioniq5, 'Ioniq5')

    # Print results for lambda values
    print(f"Lambda values for EV6: {l2lambda_ev6}")
    print(f"Lambda values for Ioniq5: {l2lambda_ioniq5}")

    # Prepare plot data
    def prepare_plot_data(results_car):
        sizes = sorted(results_car.keys())
        phys_rrmse, xgb_rrmse, lr_rrmse, only_ml_rrmse = [], [], [], []

        for size in sizes:
            phys_values = [item for result in results_car[size] if result['model'] == 'Physics-Based' for item in
                           result['rrmse']]
            xgb_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)' for item in
                          result['rrmse']]
            lr_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(Linear Regression)'
                         for item in result['rrmse']]
            only_ml_values = [item for result in results_car[size] if result['model'] == 'Only ML(XGBoost)' for item in
                              result['rrmse']]

            # Append means
            if phys_values:
                phys_rrmse.append(np.mean(phys_values))
            if xgb_values:
                xgb_rrmse.append(np.mean(xgb_values))
            if lr_values:
                lr_rrmse.append(np.mean(lr_values))
            if only_ml_values:
                only_ml_rrmse.append(np.mean(only_ml_values))

        return sizes, phys_rrmse, xgb_rrmse, lr_rrmse, only_ml_rrmse

    sizes_ev6, phys_rrmse_ev6, xgb_rrmse_ev6, lr_rrmse_ev6, only_ml_rrmse_ev6 = prepare_plot_data(results_ev6)
    sizes_ioniq5, phys_rrmse_ioniq5, xgb_rrmse_ioniq5, lr_rrmse_ioniq5, only_ml_rrmse_ioniq5 = prepare_plot_data(
        results_ioniq5)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Plot for EV6
    ax1.plot(sizes_ev6, phys_rrmse_ev6, label='Physics-Based', linestyle='--', color='r')
    ax1.plot(sizes_ev6, xgb_rrmse_ev6, label='Hybrid Model(XGBoost)', marker='o')
    ax1.plot(sizes_ev6, lr_rrmse_ev6, label='Hybrid Model(Linear Regression)', marker='o')
    ax1.plot(sizes_ev6, only_ml_rrmse_ev6, label='Only ML(XGBoost)', marker='o')

    ax1.set_xlabel('Number of Trips')
    ax1.set_ylabel('RRMSE')
    ax1.set_title('RRMSE vs Number of Trips for EV6')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')

    # Plot for Ioniq5
    ax2.plot(sizes_ioniq5, phys_rrmse_ioniq5, label='Physics-Based', linestyle='--', color='r')
    ax2.plot(sizes_ioniq5, xgb_rrmse_ioniq5, label='Hybrid Model(XGBoost)', marker='o')
    ax2.plot(sizes_ioniq5, lr_rrmse_ioniq5, label='Hybrid Model(Linear Regression)', marker='o')
    ax2.plot(sizes_ioniq5, only_ml_rrmse_ioniq5, label='Only ML(XGBoost)', marker='o')

    ax2.set_xlabel('Number of Trips')
    ax2.set_ylabel('RRMSE')
    ax2.set_title('RRMSE vs Number of Trips for Ioniq5')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xscale('log')

    # Save the figure with dpi 300
    save_path = os.path.join(fig_save_path, 'figure3_rrmse_ev6_ioniq5.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# figure1(file_lists_ev6, file_lists_ioniq5)
figure3(file_lists_ev6, file_lists_ioniq5)