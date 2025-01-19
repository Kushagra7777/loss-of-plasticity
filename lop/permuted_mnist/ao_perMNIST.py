import pickle
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import ao_core as ao
import ao_arch as ar


data_dir = 'data/permutated_data/'
task_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')])

agent = ao.Agent(...)

for task_idx, task_file in enumerate(task_files):
    print(f"Loading Task {task_idx + 1} from {task_file}...")

    # Load the task data
    with open(task_file, 'rb') as f:
        permuted_x, permuted_y = pickle.load(f)

    # Convert data to PyTorch tensors
    x_tensor = torch.tensor(permuted_x, dtype=torch.float32)
    y_tensor = torch.tensor(permuted_y, dtype=torch.long)


    train_dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #training
    agent.next_state(INPUT=..., LABEL=...)

    #testing
    agent.next_state

    


