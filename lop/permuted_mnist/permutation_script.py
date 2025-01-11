import numpy as np
import torch
import pickle


with open('data/mnist_', 'rb+') as f:
    x, y, _, _ = pickle.load(f)


num_tasks = 800
examples_per_task = 60000
input_size = 784


for task_idx in range(num_tasks):
    # pixel-wise permutation
    pixel_permutation = np.random.permutation(input_size)  # Random permutation 
    permuted_x = x[:, pixel_permutation]  # applying permutation to all images

    # shuffling the images
    data_permutation = np.random.permutation(examples_per_task)  
    permuted_x, permuted_y = permuted_x[data_permutation], y[data_permutation]

    # Saving
    task_filename = f'data/permutated_data/permuted_task_{task_idx + 1}.pkl'  
    with open(task_filename, 'wb') as task_file:
        pickle.dump([permuted_x, permuted_y], task_file)  # Saving images and corresponding labels

    print(f"Task {task_idx + 1} saved successfully!")

print("All tasks have been generated and saved.")
