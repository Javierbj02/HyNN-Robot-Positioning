import numpy as np
import cmath
import csv
import os


def process_antennas(num_antennas, num_subcarriers, folder):
    folder_samples = folder + '/samples/'
    antenna_pos = np.load(folder + '/antenna_positions.npy')
    user_pos = np.load(folder + '/user_positions.npy')

    # Indices of the selected antennas (as the original paper does)
    if num_antennas == 64:
        selected_antennas = [x for x in range(64)]
    elif num_antennas == 32:
        if folder == "ULA_lab_LoS":
            selected_antennas = [x + 16 for x in range(32)]
    elif num_antennas == 16:
        if folder == "ULA_lab_LoS":
            selected_antennas = [x + 24 for x in range(16)]
    elif num_antennas == 8:
        if folder == "ULA_lab_LoS":
            selected_antennas = [x + 28 for x in range(8)]
    rows = []
    
    user_coordinates = list(user_pos)

    for filename in os.listdir(folder_samples):
        file_path = os.path.join(folder_samples, filename)
        measurements = np.load(file_path)


        # Create matrices to store the measurements
        mod = np.zeros((num_antennas, num_subcarriers))
        arg = np.zeros((num_antennas, num_subcarriers))

        # Fill the matrices with the measurements
        for antenna_idx, antenna_measurement in enumerate(selected_antennas):
            for subcarrier_idx in range(num_subcarriers):
                subcarrier_measurement = measurements[antenna_measurement][subcarrier_idx]
                mod[antenna_idx][subcarrier_idx] = cmath.polar(subcarrier_measurement)[0]
                arg[antenna_idx][subcarrier_idx] = cmath.polar(subcarrier_measurement)[1]


        pos_idx = int(filename.split('_')[2][:-4])
        user_coordinate = user_coordinates[pos_idx]

        row = []
        
        
        # Append the measurements to the row
        for antenna_idx in range(num_antennas):
            for subcarrier_idx in range(num_subcarriers):
                row.append(mod[antenna_idx][subcarrier_idx])
                row.append(arg[antenna_idx][subcarrier_idx])
                
        row.extend(user_coordinate[:2])
        rows.append(row)

        
    # Create csv
    headers = []
    for antenna_idx in range(num_antennas):
        for subcarrier_idx in range(num_subcarriers):
            headers.append(f'A{antenna_idx + 1}S{subcarrier_idx + 1}Mod')
            headers.append(f'A{antenna_idx + 1}S{subcarrier_idx + 1}Arg')
    headers.extend(['PositionX', 'PositionY'])

    with open(f'{folder}_{num_antennas}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

# ? AVAILABLE FOLDERS: 'DIS_lab_LoS', 'ULA_lab_LoS', 'URA_lab_LoS'
# To generate the .csv files, the data folders must be in the same directory as this file

process_antennas(num_antennas=8, num_subcarriers=100, folder='ULA_lab_LoS')
process_antennas(num_antennas=16, num_subcarriers=100, folder='ULA_lab_LoS')
process_antennas(num_antennas=32, num_subcarriers=100, folder='ULA_lab_LoS')
process_antennas(num_antennas=64, num_subcarriers=100, folder='ULA_lab_LoS')

