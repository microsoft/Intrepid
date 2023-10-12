import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as save_image
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from model.misc.robot_car.autoencoder_train import CarAutoencoder
from model.misc.robot_car.latent_forward import LatentDataset


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        middle_dim = 256
        self.in0 = nn.Sequential(nn.Linear(in_dim, middle_dim), nn.LeakyReLU(), nn.BatchNorm1d(middle_dim))
        self.in1 = nn.Sequential(nn.Linear(in_dim, middle_dim), nn.LeakyReLU(), nn.BatchNorm1d(middle_dim))
        self.in2 = nn.Sequential(nn.Linear(in_dim, middle_dim), nn.LeakyReLU(), nn.BatchNorm1d(middle_dim))
        self.out = nn.Sequential(
            nn.Linear(middle_dim * 3, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x0 = self.in0(x[:, 0])
        x1 = self.in1(x[:, 1])
        x2 = self.in2(x[:, 2])
        return self.out(torch.cat((x0, x1, x2), dim=1))


class VariationalBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        middle_dim = 256
        self.in0 = nn.Sequential(nn.Linear(in_dim, middle_dim), nn.LeakyReLU(), nn.BatchNorm1d(middle_dim))
        self.in1 = nn.Sequential(nn.Linear(in_dim, middle_dim), nn.LeakyReLU(), nn.BatchNorm1d(middle_dim))
        self.in2 = nn.Sequential(nn.Linear(in_dim, middle_dim), nn.LeakyReLU(), nn.BatchNorm1d(middle_dim))
        self.mid = nn.Sequential(
            nn.Linear(middle_dim * 3, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
        )
        self.out_mu = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.BatchNorm1d(128), nn.Linear(128, out_dim))
        self.out_var = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.BatchNorm1d(128), nn.Linear(128, out_dim))

    def encode(self, x):
        x0 = self.in0(x[:, 0])
        x1 = self.in1(x[:, 1])
        x2 = self.in2(x[:, 2])
        xm = self.mid(torch.cat((x0, x1, x2), dim=1))

        mu = self.out_mu(xm)
        log_var = self.out_var(xm)
        return (mu, log_var)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kl = torch.distributions.kl_divergence(q, p).mean()
        return z, kl


class Unbottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net0 = self.make_nn(in_dim, out_dim)
        self.net1 = self.make_nn(in_dim, out_dim)
        self.net2 = self.make_nn(in_dim, out_dim)

    def make_nn(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            # nn.Linear(128, 128), nn.LeakyReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            # nn.Linear(256, 256), nn.LeakyReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x0 = self.net0(x)
        x1 = self.net1(x)
        x2 = self.net2(x)
        return torch.stack((x0, x1, x2), dim=1)


class ActionPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim)
            # nn.Linear(2*latent_dim, 256), nn.LeakyReLU(), nn.BatchNorm1d(256),
            # nn.Linear(256, 64), nn.LeakyReLU(), nn.BatchNorm1d(64),
            # nn.Linear(64, action_dim)
        )

    def forward(self, st, stk):
        return self.net(torch.cat((st, stk), dim=1))


class LatentInverse(pl.LightningModule):
    def __init__(
        self, embedding_dim=512, latent_dim=64, action_dim=4, train_on_reconstruction=False, variational=True, kl=1.0, vae=None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.train_on_reconstruction = train_on_reconstruction
        self.variational = variational
        self.kl = kl
        if variational:
            self.bottleneck = VariationalBottleneck(in_dim=embedding_dim, out_dim=latent_dim)
        else:
            self.bottleneck = Bottleneck(in_dim=embedding_dim, out_dim=latent_dim)
        self.unbottleneck = Unbottleneck(in_dim=latent_dim, out_dim=embedding_dim)
        self.action_predict = ActionPredictor(latent_dim=latent_dim, action_dim=action_dim)
        self.vae = vae
        if self.vae is not None:
            self.vae.freeze()

    def configure_optimizers(self):
        params = [self.bottleneck.parameters(), self.unbottleneck.parameters(), self.action_predict.parameters()]
        return torch.optim.Adam(itertools.chain(*params), lr=1e-5)

    def forward(self, st, stk):
        if self.variational:
            z1, kl1 = self.bottleneck(st)
            z2, kl2 = self.bottleneck(stk)
            apred = self.action_predict(z1, z2)
            return apred
        else:
            return self.action_predict(self.bottleneck(st), self.bottleneck(stk))

    def _run_step(self, batch, batch_idx, run_type="train"):
        state, state_next, _, action = batch

        # predict action from two states
        if self.variational:
            z1, kl1 = self.bottleneck(state)
            z2, kl2 = self.bottleneck(state_next)
            if self.train_on_reconstruction:
                action_pred = self.action_predict(z1.detach(), z2.detach())
            else:
                action_pred = self.action_predict(z1, z2)
            kl_loss = kl1  # (kl1 + kl2) / 2.0
        else:
            z1 = self.bottleneck(state)
            z2 = self.bottleneck(state_next)
            if self.train_on_reconstruction:
                action_pred = self.action_predict(z1.detach(), z2.detach())
            else:
                action_pred = self.action_predict(z1, z2)
        action_loss = F.mse_loss(action_pred, action)

        # reconstruct the first state
        if self.train_on_reconstruction:
            state_recon = self.unbottleneck(z1)
        else:
            state_recon = self.unbottleneck(z1.detach())
        recon_loss = F.mse_loss(state_recon, state)

        # reconstruct state back to image for batch 0
        if batch_idx == 0 and self.vae is not None:
            self.vae.to(self.device).eval()
            with torch.no_grad():
                img_true = self.vae.decode(state)
                img_pred = self.vae.decode(state_recon)
            image_out_dir = self.trainer.default_root_dir
            output = torch.cat((img_true, img_pred), dim=1).reshape(-1, 3, 256, 256)
            save_image(output[: 16 * 6], os.path.join(image_out_dir, f"{run_type}_latent_inv.jpg"), nrow=6)

        # compute total loss
        if self.variational:
            loss = action_loss + recon_loss + (self.kl * kl_loss)
        else:
            loss = action_loss + recon_loss

        log = {
            f"action_loss_{run_type}": action_loss,
            f"recon_loss_{run_type}": recon_loss,
            f"loss_{run_type}": loss,
        }
        if self.variational:
            log[f"kl_loss_{run_type}"] = kl_loss

        return loss, log

    def training_step(self, batch, batch_idx):
        loss, log = self._run_step(batch, batch_idx, run_type="train")
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log = self._run_step(batch, batch_idx, run_type="val")
        self.log_dict(log)
        return loss


if __name__ == "__main__":
    dataset_pickle = "./dataset_embeddings.p"
    autoencoder_checkpoint = "autoencoder_training/autoencoder.ckpt"
    train_root = "latent_inverse_training"
    train_split = 0.8
    batch_size = 256
    num_workers = 0
    torch.set_float32_matmul_precision("medium")

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
    model = LatentInverse(vae=vae)
    checkpoint_callback = ModelCheckpoint(dirpath=train_root, save_top_k=1, monitor="loss_val")
    trainer = pl.Trainer(
        default_root_dir=train_root,
        callbacks=[checkpoint_callback],
        max_epochs=500,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
    )

    print("Training...")
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
