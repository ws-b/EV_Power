import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

def plot_scatter_all_trip(file_lists, folder_path):
    final_energy_data = []
    final_energy = []

    for file in tqdm(file_lists):
        # create file path
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # extract time, energy, CHARGE, DISCHARGE
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        power = data['Power_IV'].tolist()

        # calculate time differences in seconds
        time_diff = t.diff().dt.total_seconds().fillna(0)

        # calculate difference between CHARGE and DISCHARGE
        energy_data = power * time_diff / 3600 / 1000
        energy_data_cumulative = energy_data.cumsum()

        # convert energy data to kWh and perform cumulative calculation
        energy_kWh = data['Energy']
        energy_kWh_cumulative = energy_kWh.cumsum()
        final_energy_data.append(energy_data_cumulative.iloc[-1])
        final_energy.append(energy_kWh_cumulative.iloc[-1])

    # plot the graph
    fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

    ax.set_xlabel('Cumulative Energy (kWh)')  # changed
    ax.set_ylabel('BMS Energy (kWh)')  # changed
    ax.scatter(final_energy, final_energy_data, color='tab:blue')  # swapped athe x and y variables

    # Add trendline
    slope, intercept, r_value, p_value, std_err = linregress(final_energy, final_energy_data)
    ax.plot(np.array(final_energy), intercept + slope * np.array(final_energy), 'b', label='fitted line')

    # Create y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.title('BMS Energy(V*I) vs. Model Energy over Time')
    plt.show()

def plot_scatter_tbt(file_lists, folder_path):
    for file in file_lists:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # extract time, Energy, CHARGE, DISCHARGE
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        CHARGE = data['trip_chrg_pw'].tolist()
        DISCHARGE = data['trip_dischrg_pw'].tolist()

        # calculate difference between CHARGE and DISCHARGE
        net_charge = np.array(DISCHARGE) - np.array(CHARGE)

        # convert Energy data to kWh and perform cumulative calculation
        Energy_kWh = data['Energy']
        Energy_kWh_cumulative = Energy_kWh.cumsum()

        # plot the graph
        fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

        ax.set_xlabel('Cumulative Energy (kWh)')  # changed
        ax.set_ylabel('Net Charge (Discharge - Charge) (kWh)')  # changed
        ax.scatter(Energy_kWh_cumulative, net_charge, color='tab:blue')  # swapped the x and y variables

        # Create y=x line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: ' + file, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.title('Net Charge vs. Cumulative Energy')
        plt.show()
