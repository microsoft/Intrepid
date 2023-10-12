import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from environments.robot_car.client.inference import ModelInferenceBase

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.LeakyReLU):
        super().__init__()
        c_hid = base_channel_size

        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(4 * c_hid, 8 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(8 * c_hid, 16 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(16 * c_hid, 16 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn())
            #nn.Flatten(),
            #nn.Linear(4 * 32 * c_hid, latent_dim))

    def forward(self, x):
        x = self.net(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, num_output_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.LeakyReLU):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
#            nn.Linear(latent_dim, 4 * 64 * c_hid),
#            nn.Unflatten(1, (64 * c_hid, 2, 2)),
            nn.ConvTranspose2d(16 * c_hid, 16 * c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(16 * c_hid, 16 * c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(16 * c_hid, 8 * c_hid, kernel_size=3, padding=1, stride=2,output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(8 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2,output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2,output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class AC(nn.Module):
    def __init__(self, indim, adim, maxk):
        super().__init__()
        self.fc_pre1 = nn.Sequential(nn.Linear(indim,2048))
        self.fc_pre2 = nn.Sequential(nn.Linear(indim,2048))
        self.embk = nn.Embedding(maxk,512)
        self.fc_post = nn.Sequential(nn.Linear(2048*2 + 512, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,512), nn.LeakyReLU(), nn.Linear(512, adim))

    def forward(self, st, stk, k):
        embedk = self.embk(k)
        ht = self.fc_pre1(st)
        htk = self.fc_pre2(stk)

        out = self.fc_post(torch.cat([embedk, ht, htk],dim=1))

        return out
        
def make_prod(S,A,maxk=64):
    row_indices = torch.arange(S.shape[0]).cuda()
    combinations = torch.cartesian_prod(row_indices, row_indices)
    st = S[combinations[:, 0]]
    stk = S[combinations[:, 1]]
    a = A[combinations[:,0]]

    k = combinations[:,1]-combinations[:,0]

    st = st[(k > 0) & (k <= maxk)]
    stk = stk[(k > 0) & (k <= maxk)]
    a = a[(k>0) & (k <= maxk)]
    k = k[(k>0) & (k <= maxk)]

    return st, stk, k, a


class AlexModelInference(ModelInferenceBase):
    def __init__(self, goal_state, model_path):
        self.goal_state = goal_state

        # seed
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = "cpu"
        #self.device = "cuda"
        if self.device == "cuda":
            torch.cuda.manual_seed(0)
        
        # load model
        self.encoder = Encoder(9, 32, 512).to(self.device)
        self.decoder = Decoder(9, 32, 512).to(self.device)
        self.acnet = AC(4*4*512, 4, 64).to(self.device)

        # hack to allow loading model file because it wasn't saved with state_dict
        import __main__
        setattr(__main__, 'Encoder', Encoder)
        setattr(__main__, 'Decoder', Decoder)
        setattr(__main__, 'AC', AC)
        model_filename = os.path.join(model_path, 'alex_car_model_06012023.pt')
        self.encoder, self.decoder, self.acnet = torch.load(model_filename)
        self.encoder = self.encoder.eval().to(self.device)
        self.decoder = self.decoder.eval().to(self.device)
        self.acnet = self.acnet.eval().to(self.device)

    def get_next_action(self, current_state, k=1, reconstruct_output=None):
        src_input = current_state.concat() / 256.0
        src_tensor = torch.FloatTensor(src_input).unsqueeze(0).to(self.device)
        target_input = self.goal_state.concat() / 256.0
        target_tensor = torch.FloatTensor(target_input).unsqueeze(0).to(self.device)

        # use the batch dimension to hold both source and target images
        # input has dimensions (batch 2, channels 3, height 250, width 750)
        xb = torch.cat((src_tensor, target_tensor), dim=0)

        # resize to (batch 2, channels 3, height 256, width 768)
        img_width = 256
        views = 3
        xb = F.interpolate(xb, (img_width, img_width*3))

        # shuffle dimensions into (batch 2, channels 9, height 256, width 256)
        bs,ch,w,h = xb.shape
        xbc = xb.reshape((bs,ch,w,h//views,views)).permute(0,1,4,2,3).reshape((bs,ch*views,w,h//views))
        xbc = xbc[:,:views*3]
        assert xbc.shape == (bs, 9, 256, 256), f"Expected shape (2, 9, 256, 256), got {xbc.shape}"

        with torch.no_grad():
            z = self.encoder(xbc)
            zflat = z.reshape((bs, -1))
            st = zflat[0].unsqueeze(0)
            stk = zflat[1].unsqueeze(0)
            k_tensor = torch.LongTensor([k]).to(self.device)
            action = self.acnet(st, stk, k_tensor)

        if not reconstruct_output:
            return self._unnormalize_action(action[0])
        
        # reconstruct the image
        with torch.no_grad():
            xrec = self.decoder(z)
            xrec = xrec[:,:views*3]
            xrec = xrec.reshape((bs,ch,views,w,h//views)).permute(0,1,3,4,2).reshape((bs,ch,w,h))
            assert xrec.shape == xb.shape, f"Expected reconstruction to have shape {xb.shape}, got {xrec.shape}"
        torchvision.utils.save_image(torch.cat((xb, xrec), dim=0), reconstruct_output, nrow=2)
        return self._unnormalize_action(action[0])
    