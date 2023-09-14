import os
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns


# Function to find min, max, mean, and standard deviation of features in the directory
def find_stats(directory):
    stats = {}
    x_values = []
    y_values = []
    z_values = []
    intensity_values = []
    range_values = []

    for filename in sorted(os.listdir(directory)):

        if filename.endswith('.bin'):
            file_path = os.path.join(directory, filename)
            
            data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

            # Append values for x, y, z, intensity, and range
            x_values.extend(data[:, 0])
            y_values.extend(data[:, 1])
            z_values.extend(data[:, 2])
            intensity_values.extend(data[:, 3])

            print(filename)

            for x, y, z  in zip(data[:, 0], data[:, 1], data[:, 2]):
                r = np.sqrt(x ** 2 + y ** 2 + z **2)

                print(x,y,z,r)
            # Calculate range for each point and append
            #range_data = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)
                range_values.append(r)

    # Calculate mean and standard deviation for each feature
    stats['min_x'] = np.min(x_values)
    stats['max_x'] = np.max(x_values)
    stats['mean_x'] = np.mean(x_values)
    stats['std_x'] = np.std(x_values)


    stats['min_y'] = np.min(y_values)
    stats['max_y'] = np.max(y_values)
    stats['mean_y'] = np.mean(y_values)
    stats['std_y'] = np.std(y_values)
    
    
    stats['min_z'] = np.min(z_values)
    stats['max_z'] = np.max(z_values)
    stats['mean_z'] = np.mean(z_values)
    stats['std_z'] = np.std(z_values)
    
    
    # Calculate mean and standard deviation for range
    stats['min_range'] = np.min(range_values)
    stats['max_range'] = np.max(range_values)
    stats['mean_range'] = np.mean(range_values)
    stats['std_range'] = np.std(range_values)

    stats['min_intensity'] = np.min(intensity_values)
    stats['max_intensity'] = np.max(intensity_values)
    stats['mean_intensity'] = np.mean(intensity_values)
    stats['std_intensity'] = np.std(intensity_values)

        # Combine features into a dictionary
    combined_features = {
        'X': x_values,
        'Y': y_values,
        'Z': z_values,
        'Intensity': intensity_values,
        'Range': range_values
    }

    # Create a DataFrame from the combined features
    df = pd.DataFrame(combined_features)
    
    return stats, df

def plot_hist(features_df:pd.DataFrame, save_figure:bool):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    for column in features_df.columns:
        plt.subplot(2, 3, features_df.columns.get_loc(column) + 1)
        sns.histplot(features_df[column], bins=50, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()

    # Save the figure (optional)
    save_figure = True
    if save_figure:
        plt.savefig('histograms.png', dpi=300)

    plt.show()


# for key, value in stats.items():
#     print(f'{key}: {value}')

# You can now use the 'stats' dictionary as needed.

def main():
    # Define the directory containing .bin files
    input_directory = '/work/akmt/dataset/SemanticKITTI/sequences/03/velodyne'
    
    # Main script
    stats, features_df = find_stats(input_directory)
    pprint.pprint(stats)

    # plot histogram
    #plot_hist(features_df, save_figure=False)

if __name__ == "__main__":
    main()