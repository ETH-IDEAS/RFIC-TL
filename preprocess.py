import pandas as pd
import numpy as np
import random
import os

# print a random sample from the x, y, z arrays along with column names
def print_random_sample(x, y, z, x_columns, y_columns, z_columns):
    """Prints a random sample from the x, y, z arrays along with column names."""
    sample_index = random.randint(0, len(x) - 1)

    print(f"Sample Index: {sample_index}")

    # Print x data with column names
    print("x:", ', '.join([f"'{col}' = {value}" for col, value in zip(x_columns, x[sample_index])]))

    # Print y data with column names
    print("y:", ', '.join([f"'{col}' = {value}" for col, value in zip(y_columns, y[sample_index])]))

    # Print z data with column names
    print("z:", ', '.join([f"'{col}' = {value}" for col, value in zip(z_columns, z[sample_index])]))

def process_data(data):
        print("All columns:", data.columns)

        # Define column names
        x_columns = ['C2', 'C1', 'RealZin', 'ImageZin']
        y_columns = ['Lp', 'Ls', 'k', 'SRF', 'Qp', 'Qs']
        z_columns = ['wlow', 'wup', 'r0', 'r1', 'xgnd', 'lfeed']

        # Print the shape of the x, y, z arrays
        print("x:", data[x_columns].values.shape, end=" ")
        print("y:", data[y_columns].values.shape, end=" ")
        print("z:", data[z_columns].values.shape)

        # Extract the data arrays
        x = data[x_columns].values.astype(np.float32)
        y = data[y_columns].values.astype(np.float32)
        z = data[z_columns].values.astype(np.float32)


        # Print the mean and standard deviation of the x, y, z arrays
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        np.save(os.path.join(data_dir, 'x_mean.npy'), x_mean)
        np.save(os.path.join(data_dir, 'x_std.npy'), x_std)
        print("x_mean:", x_mean)
        print("x_std:", x_std)
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        np.save(os.path.join(data_dir, 'y_mean.npy'), y_mean)
        np.save(os.path.join(data_dir, 'y_std.npy'), y_std)
        print("y_mean:", y_mean)
        print("y_std:", y_std)
        z_mean = np.mean(z, axis=0)
        z_std = np.std(z, axis=0)
        np.save(os.path.join(data_dir, 'z_mean.npy'), z_mean)
        np.save(os.path.join(data_dir, 'z_std.npy'), z_std)
        print("z_mean:", z_mean)
        print("z_std:", z_std)

        # Save to .npy files
        np.save(os.path.join(data_dir, 'x_data.npy'), x)
        np.save(os.path.join(data_dir, 'y_data.npy'), y)
        np.save(os.path.join(data_dir, 'z_data.npy'), z)

        # Print a random sample
        print_random_sample(x, y, z, x_columns, y_columns, z_columns)


if __name__ == '__main__':
    # # downsample the data to imitate highly sparse data
    for sparsity in [0.005, 0.01, 0.05, 0.2, 1]:
        for frequency, process in [(FreqA, "NodeB_MetalOptionA"), (FreqB, "NodeB_MetalOptionA"), (FreqA, "NodeB_MetalOptionB"), (FreqB, "NodeB_MetalOptionB")]:
        # FreqA or Freq B is the frequency in GHz such as 10, 20, 30...
        # for frequency, process in [(FreqA, "GF4_MetalOptionA"), (FreqA, "NodeA_MetalOptionA_coarse")]:
            data_dir = 'data/'
            data = pd.read_csv(os.path.join(data_dir, f'{process}_{frequency}GHz.csv'))  # Replace with your file path XTF_V2.csv
            data = data.sample(frac=sparsity, random_state=42)
            data_dir = os.path.join(data_dir, f'{process}_{frequency}GHz_{sparsity}')
            os.makedirs(data_dir, exist_ok=True)
            process_data(data)

    # prepare the source data
    for sparsity in [0.25, 0.5, 0.75, 1]:
        for frequency, process in [(FreqA, "NodeA_MetalOptionA"), (FreqA, "NodeA_MetalOptionA_coarse")]:
            data_dir = 'data/'
            data = pd.read_csv(os.path.join(data_dir, f'{process}_{frequency}GHz.csv'))  # Replace with your file path XTF_V2.csv
            data = data.sample(frac=sparsity, random_state=42)
            data_dir = os.path.join(data_dir, f'{process}_{frequency}GHz_{sparsity}')
            os.makedirs(data_dir, exist_ok=True)
            process_data(data)
