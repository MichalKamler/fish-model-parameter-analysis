import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import csv
import os
import math

R = 1.0

def loadCsvData():

    # Initialize the lists to hold data
    ALP0, ALP1_LR, ALP1_UD = [], [], []
    BET0, BET1_LR, BET1_UD = [], [], []
    LAM0, LAM1_LR, LAM1_UD = [], [], []
    minDist, avgMinDist, polarization, avgDist = [], [], [], []

    # Open and read the CSV file
    with open('data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert the data from strings to floats
            values = list(map(float, row))
            ALP0.append(values[0])
            ALP1_LR.append(values[1])
            ALP1_UD.append(values[2])
            BET0.append(values[3])
            BET1_LR.append(values[4])
            BET1_UD.append(values[5])
            LAM0.append(values[6])
            LAM1_LR.append(values[7])
            LAM1_UD.append(values[8])
            minDist.append(values[9])
            avgMinDist.append(values[10])
            polarization.append(values[11])
            avgDist.append(values[12])

    return ALP0, BET0, LAM0, minDist, avgMinDist, polarization, avgDist

def makeHeatMap(ALP0, BET0, LAM0, data, dataName, label_for_indicator, upper_limit=None, lower_limit=None):
    # Ensure inputs are numpy arrays for easy manipulation
    root = os.getcwd()

    ALP0 = np.array(ALP0)
    BET0 = np.array(BET0)
    LAM0 = np.array(LAM0)
    data = np.array(data)
    data *= 1/R

    # Get unique ALP0 and BET0 values to define the grid
    unique_ALP0 = np.unique(ALP0)
    unique_BET0 = np.unique(BET0)
    
    # Create a grid for ALP0 and BET0
    ALP0_grid, BET0_grid = np.meshgrid(unique_ALP0, unique_BET0)

    # Initialize a grid for LAM0 values (heatmap data)
    heatmap = np.zeros_like(ALP0_grid, dtype=float)
    mask_invalid = np.zeros_like(ALP0_grid, dtype=bool)

    # Fill the heatmap with LAM0 values
    for i, alp in enumerate(unique_ALP0):
        for j, bet in enumerate(unique_BET0):
            # Find the corresponding data value
            mask = (ALP0 == alp) & (BET0 == bet)
            if mask.any():
                value = data[mask][0]
                if (upper_limit is not None and value > upper_limit) or (lower_limit is not None and value < lower_limit) or value is None:  # **Check upper limit**
                    mask_invalid[j, i] = True  # **Mark as invalid**
                else:
                    heatmap[j, i] = value

                # heatmap[j, i] = data[mask][0]  # Take the first match

    # Create the heatmap
    plt.figure(figsize=(12, 9))
    # plt.title(f"Heatmap of {dataName} based on ALP0, BET0 and for LAM0 =  {LAM0}, eq = 6.25 BL")
    plt.xlabel("ALP0 []", fontsize=27)
    plt.ylabel("BET0 []", fontsize=27)
    plt.xticks(ticks=np.arange(len(unique_ALP0)), labels=unique_ALP0, fontsize=22)
    plt.yticks(ticks=np.arange(len(unique_BET0)), labels=unique_BET0, fontsize=22)
    # plt.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis')
    cmap = plt.get_cmap("viridis")
    cmap.set_bad(color='gray')  # Set color for invalid (NaN) values
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap=cmap, vmin=lower_limit, vmax=upper_limit)

    if upper_limit is not None:
        for i in range(heatmap.shape[1]):
            for j in range(heatmap.shape[0]):
                if mask_invalid[j, i]:
                    plt.plot(i, j, marker='x', color='red', markersize=10, markeredgewidth=2)
    cbar = plt.colorbar(label=label_for_indicator)
    cbar.set_label(label_for_indicator, fontsize=20)  # Set colorbar label font size
    cbar.ax.tick_params(labelsize=20)
    # plt.colorbar(label=label_for_indicator)
    plt.grid(False)

    # Show the plot
    # plt.show()
    root = os.getcwd()
    plot_dir = os.path.join(root, 'plot_data', f"LAM0_{LAM0}")
    os.makedirs(plot_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define the file path
    file_path = os.path.join(plot_dir, f"{dataName}_heatmap.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def saveHeatMaps(LAM0, ALP0, BET0, minDist, avgMinDist, polarization, avgDist, banOne_banALL=False):
    side_len_a = 10
    side_len_b = 10
    grid_size = side_len_a * side_len_b

    minDistAllowed_up = 6.
    minDistAllowed_low = 0.
    avgMinDistAllowed_up = 20.
    avgMinDistAllowed_min = 0.
    avgDistAllowed = 20.

    for i in range(len(avgDist)):
        avgDist[i] /= 10




    # if banOne_banALL:
    #     for i in range(len(minDist)):
    #         if minDist[i]>minDistAllowed or avgMinDist[i]>avgDistAllowed or avgDist[i]>avgDistAllowed:
    #             minDist[i] = None
    #             avgMinDist[i] = None
    #             avgDist[i] = None
    #             polarization[i] = None


    for i in range(0, len(LAM0)//grid_size):
        max_value = max(minDist[i*grid_size:(i+1)*grid_size])
        max_value = math.ceil(max_value / 5) * 5
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], minDist[i*grid_size:(i+1)*grid_size], "Minimal observed distance during simulation", "Distance [BL]", max_value, 0.)

        max_value = max(avgMinDist[i*grid_size:(i+1)*grid_size])
        max_value = math.ceil(max_value / 5) * 5
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], avgMinDist[i*grid_size:(i+1)*grid_size], "Average minimal observed distance during simulation", "Distance [BL]", max_value, 0.)
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], polarization[i*grid_size:(i+1)*grid_size], "Polarization", "Polarization []", 1., 0.)

        max_value = max(avgDist[i*grid_size:(i+1)*grid_size])
        max_value = math.ceil(max_value / 5) * 5
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], avgDist[i*grid_size:(i+1)*grid_size], "Average distance between all UAVs", "avgDist [BL]", max_value, 0.)

def printOnlyValid(LAM0, ALP0, BET0, minDist, avgMinDist, polarization):

    for i in range(len(LAM0)):
        if minDist[i]>=3 and avgMinDist[i]<=10:
            print(f"LAM0: {LAM0[i]}, ALP0: {ALP0[i]}, BET0: {BET0[i]}, minDist: {minDist[i]}, avgMinDist: {avgMinDist[i]}, polarization: {polarization[i]})")


if __name__=="__main__":
    ALP0, BET0, LAM0, minDist, avgMinDist, polarization, avgDist = loadCsvData()
    sorted_data = sorted(zip(LAM0, ALP0, BET0, minDist, avgMinDist, polarization, avgDist), key=lambda x: x[0])

    LAM0, ALP0, BET0, minDist, avgMinDist, polarization, avgDist = map(list, zip(*sorted_data))

    # print(len(LAM0))

    saveHeatMaps(LAM0, ALP0, BET0, minDist, avgMinDist, polarization, avgDist, banOne_banALL=True)

    # printOnlyValid(LAM0, ALP0, BET0, minDist, avgMinDist, polarization)
        

