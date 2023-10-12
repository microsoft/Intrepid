import argparse
import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from environments.robot_car.utils.dataset import CarDataset
from model.misc.robot_car.autoencoder_train import CarAutoencoder


def interpolate(a, b, steps):
    return torch.stack([a + (b-a)*i/(steps+1) for i in range(steps+2)], dim=1)


if __name__ == "__main__":
    batch_size = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("test_data", type=str)
    args = parser.parse_args()

    # check that checkpoint file exists
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    
    # check that test data directory exists
    if not os.path.isdir(args.test_data):
        raise FileNotFoundError(args.test_data)

    train_root = os.path.dirname(args.checkpoint)

    print(f"Loading model from {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model = CarAutoencoder.load_from_checkpoint(args.checkpoint).to(device)
    model.eval()

    print("Loading data...")
    dataset = CarDataset(args.test_data, max_k=1, resize=(256,256), cache_into_memory=False)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print("Testing on random latent vectors")
    z = torch.randn(batch_size, 3, model.latent_dim).to(device)
    with torch.no_grad():
        pics = model.decode(z)
    pics = pics.reshape(-1, 3, 256, 256)
    save_image(pics, os.path.join(train_root, "test_decode_random_normal.jpg"), nrow=6)

    print("Testing image reconstruction")
    batch = next(iter(dataloader_train))
    st, stk, _, _ = batch
    with torch.no_grad():
        st_enc, _ = model.encode(st.to(device))
        stk_enc, _ = model.encode(stk.to(device))
        pics = model.decode(st_enc)
    pics = pics.reshape(-1, 3, 256, 256)
    save_image(pics, os.path.join(train_root, "test_reconstruct.jpg"), nrow=6)
    save_image(st.reshape(-1, 3, 256, 256), os.path.join(train_root, "test_original.jpg"), nrow=6)

    print("Testing step interpolation")
    n_interp = 10
    interp0 = interpolate(st_enc[:, 0], stk_enc[:, 0], steps=n_interp-2).reshape(-1, model.latent_dim)
    interp1 = interpolate(st_enc[:, 1], stk_enc[:, 1], steps=n_interp-2).reshape(-1, model.latent_dim)
    interp2 = interpolate(st_enc[:, 2], stk_enc[:, 2], steps=n_interp-2).reshape(-1, model.latent_dim)

    with torch.no_grad():
        pics = model.decode(torch.stack([interp0, interp1, interp2], dim=1).reshape(-1, 3, model.latent_dim).to(device))
    pics0 = pics[:, 0].reshape(-1, n_interp, 3, 256, 256)
    pics1 = pics[:, 1].reshape(-1, n_interp, 3, 256, 256)
    pics2 = pics[:, 2].reshape(-1, n_interp, 3, 256, 256)
    pics = torch.concatenate([pics0, pics1, pics2], dim=1).reshape(-1, 3, 256, 256)
    save_image(pics, os.path.join(train_root, "test_interpolate_step.jpg"), nrow=n_interp)

    print("Testing random interpolation")
    interp0 = interpolate(st_enc[:-1, 0], st_enc[1:, 0], steps=n_interp-2).reshape(-1, model.latent_dim)
    interp1 = interpolate(st_enc[:-1, 1], st_enc[1:, 1], steps=n_interp-2).reshape(-1, model.latent_dim)
    interp2 = interpolate(st_enc[:-1, 2], st_enc[1:, 2], steps=n_interp-2).reshape(-1, model.latent_dim)

    with torch.no_grad():
        pics = model.decode(torch.stack([interp0, interp1, interp2], dim=1).reshape(-1, 3, model.latent_dim).to(device))
    pics0 = pics[:, 0].reshape(-1, n_interp, 3, 256, 256)
    pics1 = pics[:, 1].reshape(-1, n_interp, 3, 256, 256)
    pics2 = pics[:, 2].reshape(-1, n_interp, 3, 256, 256)
    pics = torch.concatenate([pics0, pics1, pics2], dim=1).reshape(-1, 3, 256, 256)
    save_image(pics, os.path.join(train_root, "test_interpolate_random.jpg"), nrow=n_interp)

    print(f"Images saved to {train_root}")
