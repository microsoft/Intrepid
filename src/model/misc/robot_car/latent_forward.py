import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image as save_image
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from model.misc.robot_car.autoencoder_train import CarAutoencoder


# Dataset object for precomputed latent embeddings
class LatentDataset(Dataset):
    def __init__(self, filename, max_k=1):
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            self.data = pickle.load(f)
        self.embeddings = self.data["embeddings"]
        self.actions = self.data["actions"]
        self.traj_lengths = self.data["traj_lengths"]
        self.total_samples = self.data["total_samples"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32).to(device)
        self.actions = torch.tensor(self.actions, dtype=torch.float32).to(device)

        # in order to avoid crossing between trajectories, subtract max_k from each trajectory length
        # save both the original and the modified trajectory lengths
        # when indexing into actions and filenames, we must convert from subtracted to actual index
        self.traj_lengths_minus_k = self.traj_lengths - max_k
        self.cumulative_lengths = np.cumsum(self.traj_lengths)
        self.cumulative_lengths_minus_k = np.cumsum(self.traj_lengths_minus_k)
        self.max_k = max_k

        assert self.total_samples == self.cumulative_lengths[-1]

    def __len__(self):
        return self.cumulative_lengths_minus_k[-1]

    def __getitem__(self, idx):
        # find the trajectory that idx belongs to
        traj_idx = np.searchsorted(self.cumulative_lengths_minus_k, idx, side="right")
        if traj_idx >= len(self.cumulative_lengths_minus_k):
            raise IndexError(f"index {idx} out of range for dataset with length {len(self)}")
        if traj_idx > 0:
            # we need to re-map index to handle subtraction of max_k
            index_in_traj = idx - self.cumulative_lengths_minus_k[traj_idx - 1]
            actual_idx = self.cumulative_lengths[traj_idx - 1] + index_in_traj
        else:
            # we are in the first trajectory
            actual_idx = idx

        if self.max_k > 1:
            k = torch.randint(1, self.max_k + 1, (1,)).item()
        else:
            k = 1
        st_emb = self.embeddings[actual_idx]
        stk_emb = self.embeddings[actual_idx + k]
        action = self.actions[actual_idx]

        return (st_emb, stk_emb, k, action)


class LatentForward(pl.LightningModule):
    def __init__(self, input_dim=512, latent_dim=64, action_dim=4, kl=0.0001, lr=1e-4, vae=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.kl = kl
        self.lr = lr
        self.vae = vae
        if self.vae is not None:
            self.vae.freeze()

        d = 0.2
        self.encoder = nn.Sequential(
            nn.Linear(3 * input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(256, 2 * latent_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.LeakyReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(d),
            nn.Linear(2048, 3 * input_dim),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def encode(self, state):
        state = state.reshape(-1, 3 * self.input_dim)
        output = self.encoder(state)
        mu, std = torch.chunk(output, 2, dim=1)
        std = torch.exp(std / 2)
        dist = torch.distributions.Normal(mu, std)
        normal = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl = torch.distributions.kl_divergence(dist, normal).mean()
        return dist, kl

    def forward(self, state, action):
        # z_dist, _ = self.encode(state)
        # z = z_dist.rsample()
        state = state.reshape(-1, 3 * self.input_dim)
        output = self.encoder(state)
        mu, std = torch.chunk(output, 2, dim=1)
        z = mu
        a_enc = self.action_encoder(action)
        state_next = self.decoder(torch.cat((z, a_enc), dim=1))
        # return state_next.reshape(-1, 3, self.input_dim)
        return state_next.reshape(state_next.shape[0], -1)

    def augment(self, state, state_next, action, jitter, replace):
        # jitter all actions by small amount, except for direction
        if jitter:
            jitter_amount = 0.02
            rand_jitter = torch.randn_like(action, device=self.device) * jitter_amount
            action = action + rand_jitter
            action[:, 1] = torch.round(action[:, 1])

        # generate random actions that do nothing
        null_actions = torch.rand_like(action, device=self.device)
        # direction should be rounded to 0 or 1
        null_actions[:, 1] = torch.round(null_actions[:, 1])
        # randomly zero out either speed or time
        speed_zero = torch.randint(0, 2, (action.shape[0],), dtype=torch.bool, device=self.device)
        time_zero = torch.logical_not(speed_zero)
        null_actions[:, 2] *= speed_zero
        null_actions[:, 3] *= time_zero

        # replace some real actions with do-nothing actions
        # replace actual next state with original state
        mask = torch.rand(action.shape[0], device=self.device) < replace
        action = torch.where(mask.unsqueeze(1), null_actions, action)
        state_next = torch.where(mask.unsqueeze(1).unsqueeze(2), state, state_next)
        return state, state_next, action, mask

    def _run_step(self, batch, batch_idx, run_type="train"):
        state, state_next, _, action = batch
        state, state_next, action, mask = self.augment(
            state, state_next, action, jitter=(run_type == "train"), replace=(0.2 if run_type == "train" else 0.0)
        )

        z_dist, kl = self.encode(state)
        z = z_dist.rsample()
        a_enc = self.action_encoder(action)
        state_next_pred = self.decoder(torch.cat((z, a_enc), dim=1)).reshape(-1, 3, self.input_dim)
        state_pred_error = F.mse_loss(state_next_pred, state_next)

        if batch_idx == 0 and run_type == "val" and self.vae is not None:
            self.vae.to(self.device).eval()
            with torch.no_grad():
                img_orig = self.vae.decode(state)
                img_pred = self.vae.decode(state_next_pred)
                img_true = self.vae.decode(state_next)
            image_out_dir = self.trainer.default_root_dir
            num_img_per_row = 9
            output = torch.cat((img_orig, img_true, img_pred), dim=1).reshape(-1, 3, 256, 256)
            save_image(
                output[: 8 * num_img_per_row], os.path.join(image_out_dir, f"{run_type}_latent_fwd.jpg"), nrow=num_img_per_row
            )
            if run_type == "val":
                print("Actions:", action[:8])
                print("Action replaced with do-nothing:", mask[:8])

        self.log(f"state_pred_{run_type}", state_pred_error)
        self.log(f"kl_{run_type}", kl)
        return state_pred_error + self.kl * kl

    def training_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx, run_type="train")
        self.log("loss_train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx, run_type="val")
        self.log("loss_val", loss)
        return loss


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    dataset_pickle = "./dataset_embeddings.p"
    autoencoder_checkpoint = "autoencoder_training/autoencoder.ckpt"
    train_root = "latent_forward_training"
    train_split = 0.8
    batch_size = 256
    num_workers = 0

    print("Loading dataset pickle...")
    dataset = LatentDataset(dataset_pickle)
    torch.manual_seed(0)
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, [int(len(dataset) * train_split), len(dataset) - int(len(dataset) * train_split)]
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=(num_workers > 0)
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=(num_workers > 0)
    )

    print("Loading autoencoder...")
    vae = CarAutoencoder.load_from_checkpoint(autoencoder_checkpoint)

    print("Initializing model...")
    model = LatentForward(vae=vae)
    checkpoint_callback = ModelCheckpoint(dirpath=train_root, save_top_k=1, monitor="loss_val")
    trainer = pl.Trainer(
        default_root_dir=train_root,
        callbacks=[checkpoint_callback],
        max_epochs=1000,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=5,
    )

    print("Training...")
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
