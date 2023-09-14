""" Normalise intensity feature of the point cloud"""

import os
import numpy as np

# Function to normalize intensity values
def normalize_intensity(input_directory:str, min_intensity:float, max_intensity:float):
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.bin'):
            file_path = os.path.join(input_directory, filename)
    
    
        data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        data[:, 3] = (data[:, 3] - min_intensity) / (max_intensity - min_intensity)
        print(file_path)
        data.tofile(file_path)

    print(f'Finished Normalization:{len(os.listdir(input_directory))}')

def main():
    # Define the directory containing .bin files
    input_directory = '/work/akmt/dataset/SLAMANTIC/sequences/02/velodyne'
    max_intensity = 2876.0
    min_intensity =  1.0

    normalize_intensity(input_directory, min_intensity, max_intensity)



if __name__ == "__main__":
    main()