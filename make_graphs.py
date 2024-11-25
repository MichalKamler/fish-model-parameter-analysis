import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import csv

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

def makeHeatMap(ALP0, BET0, LAM0, data, dataName):
    # Ensure inputs are numpy arrays for easy manipulation
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
    plt.colorbar(label="Minimal distance [BL]")
    plt.grid(False)

    # Show the plot
    plt.show()











if __name__=="__main__":
    ALP0, BET0, LAM0, minDist, avgMinDist, polarization = loadCsvData()
    sorted_data = sorted(zip(LAM0, ALP0, BET0, minDist, avgMinDist, polarization), key=lambda x: x[0])

    LAM0, ALP0, BET0, minDist, avgMinDist, polarization = map(list, zip(*sorted_data))

    for i in range(0, len(LAM0)//25):
        makeHeatMap(ALP0[i*25:(i+1)*25], BET0[i*25:(i+1)*25], LAM0[i*25:(i+1)*25][0], minDist[i*25:(i+1)*25], "minimal observed distance")
        break





# # Example data generation (you will replace these with your actual data)
# alpha_0 = np.logspace(-2, 1, 20)  # y-axis values
# print(alpha_0)
# exit()
# beta_0 = np.logspace(-2, 1, 20)   # x-axis values
# data1 = np.random.rand(len(alpha_0), len(beta_0)) * 10  # First heatmap
# data2 = np.random.rand(len(alpha_0), len(beta_0)) * 2   # Second heatmap

# # Create the figure and axes
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot the first heatmap
# heatmap1 = ax.imshow(
#     data1,
#     extent=[beta_0[0], beta_0[-1], alpha_0[0], alpha_0[-1]],
#     aspect='auto',
#     origin='lower',
#     norm=LogNorm(vmin=1, vmax=10),  # Log scale for color normalization
#     cmap='inferno'
# )

# # Overlay hatching (adjust conditions as per your requirement)
# hatch_condition = (data1 < 1)  # Example condition for hatching
# for i in range(len(alpha_0) - 1):  # Exclude last index for alpha_0
#     for j in range(len(beta_0) - 1):  # Exclude last index for beta_0
#         if hatch_condition[i, j]:
#             ax.add_patch(plt.Rectangle(
#                 (beta_0[j], alpha_0[i]),  # Bottom-left corner
#                 beta_0[j+1] - beta_0[j],  # Width
#                 alpha_0[i+1] - alpha_0[i],  # Height
#                 fill=False, hatch='//', edgecolor='green'
#             ))

# # Add a colorbar for the first heatmap
# cbar1 = plt.colorbar(heatmap1, ax=ax, orientation='vertical', pad=0.02)
# cbar1.set_label("10BL")

# # Plot the second heatmap in the same region
# heatmap2 = ax.imshow(
#     data2,
#     extent=[beta_0[0], beta_0[-1], alpha_0[0], alpha_0[-1]],
#     aspect='auto',
#     origin='lower',
#     norm=LogNorm(vmin=1, vmax=2),  # Adjust limits as needed
#     cmap='inferno'
# )

# # Add a colorbar for the second heatmap
# cbar2 = plt.colorbar(heatmap2, ax=ax, orientation='vertical', pad=0.2)
# cbar2.set_label("2BL")

# # Format axes
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel(r"$\beta_0$")
# ax.set_ylabel(r"$\alpha_0$")
# ax.set_title("Minimal distance")

# plt.show()
