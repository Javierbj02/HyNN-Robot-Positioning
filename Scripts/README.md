# About the creation of the Datasets (/Create Datasets/)

The script "generate_scenarios.py" creates the datasets in .csv format that we will work with. To create these datasets, we use the measurement data that was collected. For the purposes of the project, we use the ULA scenario data. 

Within each folder, we find an .npy containing the coordinate point positions of the antennas and another .npy with the position of the "user". This "user" position refers to the coordinates at the coordinate point where each measurement is made. In addition, we find the "samples" folder, which contains the .npy with the CSI measurements of each position. As we have 252004 different positions, inside "samples" we find 252004 .npy files with the CSI measurements, each one defining a 64x100 matrix, according to the data in complex numbers collected from the 100 subcarriers of each of the 64 antennas. The "antenna_positions.py" file represents a 64x3 matrix, according to the 64 antennas and the coordinate position of each antenna (x, y, z). The "user_positions.npy" file represents a matrix of 252004x3, according to the coordinates (x, y, z) of each of the positions over which measurements are made.

In /ULA_lab_LoS/ folder, we find an example of the data that correspond to the first position measured.

As for the "generate_scenarios.py" file, it uses this data to create the .csv datasets. First, we convert the data from complex numbers to polar form. Secondly, we select the number of antennas to be taken into account to create the dataset: 8, 16, 32 and 64 antennas, so the fewer antennas, the less data there will be and the smaller the .csv dataset will be.

In short, we have 4 datasets in total. The dataset with 64 antennas have 252004 rows (all measured positions) x 12800 columns (64 antennas x 100 subcarriers x 2 (polar shape)).

# Size of Datasets:
8 antennas: 7.7GB
16 antennas: 15.4GB
32 antennas: 30.8GB
64 antennas: 61.6GB

Due to these sizes, it is not possible to store the datasets in Github. They can be found at: https://pruebasaluuclm-my.sharepoint.com/:f:/r/personal/javier_ballesteros_uclm_es/Documents/HyNN-CSI?csf=1&web=1&e=K7ahNy

# About HyNN Models (/HyNN Models/)

The ULA folder contains 4 scripts, each one referring to the models created from the datasets with 8, 16, 32 and 64 antennas.

Each script creates two models: one for predicting the X position and one for the Y position. In addition, it is necessary to save the model of the images when creating all the images.

The trained models can be found at: https://pruebasaluuclm-my.sharepoint.com/:f:/r/personal/javier_ballesteros_uclm_es/Documents/HyNN-CSI?csf=1&web=1&e=K7ahNy