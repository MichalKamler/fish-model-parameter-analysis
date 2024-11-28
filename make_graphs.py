import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import csv
import os

R = 1.0

def loadCsvData():

    # Initialize the lists to hold data
    ALP0, ALP1_LR, ALP1_UD = [], [], []
    BET0, BET1_LR, BET1_UD = [], [], []
    LAM0, LAM1_LR, LAM1_UD = [], [], []
    minDist, avgMinDist, polarization = [], [], []

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

    return ALP0, BET0, LAM0, minDist, avgMinDist, polarization

def makeHeatMap(ALP0, BET0, LAM0, data, dataName, label_for_indicator):
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

    # Fill the heatmap with LAM0 values
    for i, alp in enumerate(unique_ALP0):
        for j, bet in enumerate(unique_BET0):
            # Find the corresponding data value
            mask = (ALP0 == alp) & (BET0 == bet)
            if mask.any():
                heatmap[j, i] = data[mask][0]  # Take the first match

    # Create the heatmap
    plt.figure(figsize=(12, 9))
    plt.title(f"Heatmap of {dataName} based on ALP0, BET0 and for LAM0 =  {LAM0}")
    plt.xlabel("ALP0 []")
    plt.ylabel("BET0 []")
    plt.xticks(ticks=np.arange(len(unique_ALP0)), labels=unique_ALP0)
    plt.yticks(ticks=np.arange(len(unique_BET0)), labels=unique_BET0)
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label=label_for_indicator)
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

def saveHeatMaps(LAM0, ALP0, BET0, minDist, avgMinDist, polarization):
    side_len_a = 10
    side_len_b = 10
    grid_size = side_len_a * side_len_b

    for i in range(0, len(LAM0)//grid_size):
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], minDist[i*grid_size:(i+1)*grid_size], "minimal observed distance", "Distance [BL]")
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], avgMinDist[i*grid_size:(i+1)*grid_size], "average minimal observed distance", "Distance [BL]")
        makeHeatMap(ALP0[i*grid_size:(i+1)*grid_size], BET0[i*grid_size:(i+1)*grid_size], LAM0[i*grid_size:(i+1)*grid_size][0], polarization[i*grid_size:(i+1)*grid_size], " polarization", "Polarization []")

def printOnlyValid(LAM0, ALP0, BET0, minDist, avgMinDist, polarization):

    for i in range(len(LAM0)):
        if minDist[i]>=3 and avgMinDist[i]<=10:
            print(f"LAM0: {LAM0[i]}, ALP0: {ALP0[i]}, BET0: {BET0[i]}, minDist: {minDist[i]}, avgMinDist: {avgMinDist[i]}, polarization: {polarization[i]})")


if __name__=="__main__":
    ALP0, BET0, LAM0, minDist, avgMinDist, polarization = loadCsvData()
    sorted_data = sorted(zip(LAM0, ALP0, BET0, minDist, avgMinDist, polarization), key=lambda x: x[0])

    LAM0, ALP0, BET0, minDist, avgMinDist, polarization = map(list, zip(*sorted_data))

    # print(len(LAM0))

    # saveHeatMaps(LAM0, ALP0, BET0, minDist, avgMinDist, polarization)

    printOnlyValid(LAM0, ALP0, BET0, minDist, avgMinDist, polarization)
        

