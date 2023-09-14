# Reference: https://github.com/PRBonn/lidar-bonnetal/issues/78
 
import numpy as np
import os


def convert2bin(root_path: str):

    pcds_path = [os.path.join(root_path, pcd) for pcd in os.listdir(root_path)]

    print(len(pcds_path))
    faulty_files = [] 
    for idx, pcd_path in enumerate(pcds_path):
        try:
            points_data = np.loadtxt(pcd_path, skiprows=1, max_rows=1000000)
            
            # Filter out points with zero x, y, and z values
            non_zero_points = points_data[(points_data[:, 0] != 0) & (points_data[:, 1] != 0) & (points_data[:, 2] != 0)]

            bin_filename = f"{pcd_path.split(sep='.')[0]}.bin"

            print(bin_filename)

            x = non_zero_points[:, 0]
            y = non_zero_points[:, 1]
            z = non_zero_points[:, 2]
            intensity = non_zero_points[:, 3]
            arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)
            arr[::4] = x
            arr[1::4] = y
            arr[2::4] = z
            arr[3::4] = intensity
            arr.astype('float32').tofile(bin_filename)
        except Exception as e:
            print(f"Error processing file {pcd_path}: {e}")
            faulty_files.append(pcd_path)

    return faulty_files


def main():
    root_path = "/work/SLAMantic/sequences/03/velodyne"

    # EMPTY FILE IN ORIGINAL PATH: /work/data/SLAMantic/2023_03_03_11_04_58_sync/scans/000925.txt:
    # IN KITTI FORMAT: /work/SLAMantic/sequences/01/velodyne/000925.txt

    # load pcd
    faulty_files = convert2bin(root_path)
    print("Faulty files:", faulty_files)


if __name__ == "__main__":
    main()
