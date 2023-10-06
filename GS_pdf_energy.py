import os
import platform

def main():
    print("1: Ioniq5")
    print("2: Kona_EV")
    print("3: Porter_EV")
    print("4: Quitting the program.")
    car = int(input("Select Car you want to calculate: "))
    if platform.system() == "Windows":
        folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
    elif platform.system() == "Darwin":
        folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터')
    else:
        print("Unknown system.")
        return
    folder_path = os.path.join(folder_path, 'trip_by_trip')

    if car == 1: #ioniq5
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if '01241248782' in file]
    elif car == 2: #kona_ev
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if '01241248726' in file]
    elif car == 3: #porter_ev
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if '01241228177' in file]
    elif car == 4:
        print("Quitting the program.")
        return
    else:
        print("Invalid choice. Please try again.")

    file_lists.sort()

def plot_bms_energy(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Energy (kWh)')
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:blue')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('BMS Energy')
        plt.tight_layout()
        plt.show()