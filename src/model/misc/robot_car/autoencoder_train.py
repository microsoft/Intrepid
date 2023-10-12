import argparse
import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from environments.robot_car.utils.dataset import CarDataset
from model.misc.robot_car.pl_vae import VAE


# Split pixel-wise into small square patches
# Find the patches with highest average loss
# Input argument losses = F.mse_loss(original, reconstruct, reduction="none")
def patch_loss(losses):
    # average over channels
    losses = losses.mean(dim=1)
    # split into 16x16 patches and flatten
    patch_size = 16
    n_patches = losses.shape[1] // patch_size
    patches = (
        losses.reshape(losses.shape[0], losses.shape[1], n_patches, patch_size)
        .permute(0, 2, 1, 3)
        .reshape(losses.shape[0], n_patches * n_patches, patch_size * patch_size)
    )
    # take the mean of each flattened patch
    patch_losses = patches.mean(dim=2)
    # return the mean of the top 4 patch means
    return patch_losses.topk(k=4, dim=1).values.mean()


class CarAutoencoder(pl.LightningModule):
    def __init__(self, latent_dim=512, kl_coeff=0.1, patch_coeff=1.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.patch_coeff = patch_coeff
        self.lr = lr
        self.max_kl_coeff = kl_coeff
        self.enc0 = VAE(
            input_height=256, first_conv=True, maxpool1=True, enc_out_dim=512, kl_coeff=kl_coeff, latent_dim=latent_dim
        )
        self.enc1 = VAE(
            input_height=256, first_conv=True, maxpool1=True, enc_out_dim=512, kl_coeff=kl_coeff, latent_dim=latent_dim
        )
        self.enc2 = VAE(
            input_height=256, first_conv=True, maxpool1=True, enc_out_dim=512, kl_coeff=kl_coeff, latent_dim=latent_dim
        )

    def forward(self, st):
        encoding = self.encode(st)
        # z = self.sample(encoding)
        # return z.reshape(z.shape[0], -1)
        mu = encoding[0]
        return mu.reshape(mu.shape[0], -1)

    def encode(self, st):
        # input is (batch_size, 3, ch, h, w)
        z0 = self.enc0.encoder(st[:, 0])
        z1 = self.enc1.encoder(st[:, 1])
        z2 = self.enc2.encoder(st[:, 2])
        mu0 = self.enc0.fc_mu(z0)
        mu1 = self.enc1.fc_mu(z1)
        mu2 = self.enc2.fc_mu(z2)
        # model outputs log(2*var)
        log_var0 = self.enc0.fc_var(z0)
        log_var1 = self.enc1.fc_var(z1)
        log_var2 = self.enc2.fc_var(z2)
        return (torch.stack([mu0, mu1, mu2], dim=1), torch.stack([log_var0, log_var1, log_var2], dim=1))

    def sample(self, encoding):
        mu, log_var = encoding
        _, _, z0 = self.enc0.sample(mu[:, 0], log_var[:, 0])
        _, _, z1 = self.enc1.sample(mu[:, 1], log_var[:, 1])
        _, _, z2 = self.enc2.sample(mu[:, 2], log_var[:, 2])
        return torch.stack([z0, z1, z2], dim=1)

    def decode(self, z):
        # input is (batch_size, 3, latent_dim)
        pic0 = self.enc0.decoder(z[:, 0])
        pic1 = self.enc1.decoder(z[:, 1])
        pic2 = self.enc2.decoder(z[:, 2])
        return torch.stack([pic0, pic1, pic2], dim=1)

    def set_kl_coeff(self, kl_coeff):
        self.enc0.kl_coeff = kl_coeff
        self.enc1.kl_coeff = kl_coeff
        self.enc2.kl_coeff = kl_coeff

    def get_kl_coeff(self):
        # returns the current kl_coeff, not the max
        return self.enc0.kl_coeff

    def _run_step(self, batch, batch_idx, run="train"):
        st, stk, k, a_true = batch
        pics = torch.cat([st, stk], dim=0)

        # VAE expects image labels but doesn't use them, so use None
        loss0, log0, recon_losses_0 = self.enc0.step((pics[:, 0], None), batch_idx)
        loss1, log1, recon_losses_1 = self.enc1.step((pics[:, 1], None), batch_idx)
        loss2, log2, _ = self.enc2.step((pics[:, 2], None), batch_idx)
        patch_loss_0 = patch_loss(recon_losses_0)
        patch_loss_1 = patch_loss(recon_losses_1)
        loss = loss0 + loss1 + loss2 + self.patch_coeff * patch_loss_0 + self.patch_coeff * patch_loss_1

        logs = {
            **{f"{k}_0": v for k, v in log0.items()},
            **{f"{k}_1": v for k, v in log1.items()},
            **{f"{k}_2": v for k, v in log2.items()},
            "patch_loss_0": patch_loss_0,
            "patch_loss_1": patch_loss_1,
            "kl_coeff": self.get_kl_coeff(),
        }

        if batch_idx == 0:
            # Save reconstructed images
            with torch.no_grad():
                pic0_recon = self.enc0(pics[:, 0])
                pic1_recon = self.enc1(pics[:, 1])
                pic2_recon = self.enc2(pics[:, 2])
                pics_recon = torch.stack([pic0_recon, pic1_recon, pic2_recon], dim=1).reshape(-1, 3, 256, 256)
                image_out_dir = self.trainer.default_root_dir
                save_image(pics.reshape(-1, 3, 256, 256), os.path.join(image_out_dir, f"{run}_original.jpg"), nrow=12)
                save_image(pics_recon, os.path.join(image_out_dir, f"{run}_recon.jpg"), nrow=12)

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._run_step(batch, batch_idx, run="train")
        self.log_dict({f"{k}_train": v for k, v in logs.items()})
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._run_step(batch, batch_idx, run="val")
        self.log_dict({f"{k}_val": v for k, v in logs.items()})
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)


class KLStepCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        kl_coeffs = [0.001, 0.025, 0.05, 0.075, 0.1, 0.1]
        index = epoch
        kl = kl_coeffs[index % len(kl_coeffs)]
        print(f"\nEpoch={epoch} Steps={trainer.global_step}: Setting KL coeff to {kl}")
        pl_module.set_kl_coeff(kl)


class KLRampCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        cycle_length = 10000
        kl_min = 0.01
        kl_max = pl_module.max_kl_coeff
        kl_ramp_end = cycle_length // 2
        if trainer.global_step < cycle_length:
            return kl_min
        step = trainer.global_step % cycle_length
        if step < kl_ramp_end:
            kl = (kl_max - kl_min) * step / kl_ramp_end + kl_min
        else:
            kl = kl_max
        pl_module.set_kl_coeff(kl)

    def on_validation_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        step = trainer.global_step
        kl = pl_module.get_kl_coeff()
        print(f"\nEpoch={epoch} Steps={step}: KL coeff is {kl}")


if __name__ == "__main__":
    data_root = "./car_data"
    train_root = "./autoencoder_training"
    max_k = 1
    num_workers = 8
    train_split = 0.8
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="?", default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()

    # Load model checkpoint if given on command line
    if args.checkpoint != "" and os.path.isfile(args.checkpoint):
        print(f"Loading model from {args.checkpoint}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CarAutoencoder.load_from_checkpoint(args.checkpoint).to(device)
        model.lr = args.lr
    else:
        print("Initializing model...")
        model = CarAutoencoder()
        model.lr = args.lr

    print("Loading data...")
    dataset = CarDataset(data_root, max_k=max_k, resize=(256, 256), cache_into_memory=True)
    torch.manual_seed(0)
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, [int(len(dataset) * train_split), len(dataset) - int(len(dataset) * train_split)]
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Training...")
    # save all checkpoints
    topk = -1
    checkpoint_callback = ModelCheckpoint(dirpath=train_root, save_top_k=topk, monitor="val_loss")
    kl_callback = KLStepCallback()
    trainer = pl.Trainer(
        default_root_dir=train_root,
        callbacks=[checkpoint_callback, kl_callback],
        max_epochs=100,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
