import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from environments.robot_car.utils.dataset import CarDataset
from model.misc.robot_car.autoencoder_train import CarAutoencoder


# Precompute VAE embeddings for all images in the dataset
# Embeddings are then saved into a pickle file
if __name__ == "__main__":
    batch_size = 24
    data_root = "./car_data"
    train_root = "./autoencoder_training"
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    # check that checkpoint file exists
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    print(f"Loading model from {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarAutoencoder.load_from_checkpoint(args.checkpoint).to(device)

    print("Loading data...")
    dataset = CarDataset(data_root, max_k=1, resize=(256,256), cache_into_memory=False)
    num_samples = len(dataset.actions)
    assert len(dataset.pic_filenames) == num_samples
    assert sum(dataset.traj_lengths) == num_samples
    assert dataset.cumulative_lengths[-1] == num_samples
    assert len(dataset.traj_lengths) == len(dataset.cumulative_lengths)

    print("Generating embeddings...")
    traj_ends = dataset.cumulative_lengths - 1
    embeddings = []
    actions = []
    for i in tqdm(range(num_samples)):
        pics = dataset._load_pics_at_index(i)
        pics = torch.tensor(pics, dtype=torch.float32, device=device).unsqueeze(0)
        pics = pics / 256.0
        with torch.no_grad():
            z, _ = model.encode(pics)
        embeddings.append(z.squeeze().cpu().numpy())

        if i in traj_ends:
            action = np.array([0.5, 0.5, 0.5, 0.5])
        else:
            # get action that comes after this observation
            action = dataset.actions[i+1]
        actions.append(action)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    output = {
        "embeddings": embeddings,
        "actions": actions,
        "traj_lengths": dataset.traj_lengths,
        "total_samples": num_samples,
    }

    print("Saving pickle...")
    with open(args.output_file, "wb") as f:
        pickle.dump(output, f)       
