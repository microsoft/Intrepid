import json
import os
import math
import numpy as np
import cv2 as cv
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.utils import save_image as save_image


def normalize_action(a):
    ACTION_MIN = np.array([-10.0, 0.0, 0.0, 0.1])
    ACTION_MAX = np.array([50.0, 1.0, 0.5, 0.5])
    return (np.array(a, dtype=np.float32) - ACTION_MIN) / (ACTION_MAX - ACTION_MIN)

def unnormalize_action(a):
    ACTION_MIN = np.array([-10.0, 0.0, 0.0, 0.1])
    ACTION_MAX = np.array([50.0, 1.0, 0.5, 0.5])
    return np.array(a, dtype=np.float32) * (ACTION_MAX - ACTION_MIN) + ACTION_MIN


class CarDataset(Dataset):
    def __init__(self, root_dir, max_k=1, resize=(256,256), median_filter=False, load_into_memory=False, cache_into_memory=False):
        print(f"Loading dataset from {root_dir}")
        # get all subdirectories
        # each should contain a file called actions.txt
        subdirs = [d for d in os.listdir(root_dir) if
                   os.path.isdir(os.path.join(root_dir, d)) and
                   os.path.isfile(os.path.join(root_dir, d, 'actions.txt'))]
        assert len(subdirs) > 0, f"No subdirectories found in {root_dir}"

        actions = []
        pic_filenames = []
        traj_lengths = []
        max_filename_len = 0
        for dir in subdirs:
            # read log file for this subdirectory
            with open(os.path.join(root_dir, dir, 'actions.txt')) as f:
                log = f.readlines()
            log = [json.loads(a) for a in log if a.strip() != '']

            # make sure trajectory is long enough
            if len(log) < max_k + 1:
                continue

            # load a single trajectory
            traj_actions = []
            traj_pics = []
            for line in log:
                traj_actions.append(normalize_action([
                    line['angle'],
                    0. if line['direction'] == 'forward' else 1.,
                    line['speed'],
                    line['time']
                ]))
                traj_pics.append([
                    os.path.join(root_dir, dir, line['cam0']),
                    os.path.join(root_dir, dir, line['cam1']),
                    os.path.join(root_dir, dir, line['cam_car']),
                ])
                max_filename_len = max(max_filename_len, max([len(f) for f in traj_pics[-1]]))

            assert len(traj_actions) == len(traj_pics)
            actions.append(np.array(traj_actions))
            traj_lengths.append(len(traj_actions))
            pic_filenames.append(traj_pics)
        
        # actions and filenames are 3d arrays
        # dim 0 = trajectory (subdirectory of dataset)
        # dim 1 = action within a trajectory
        # dim 2 = action component (angle, direction, speed, time) or camera view
        # concatenate them into 2d arrays
        self.actions = np.concatenate(actions, dtype=np.float32)
        self.pic_filenames = np.concatenate(pic_filenames, dtype=np.dtype(('U', max_filename_len)))

        # in order to avoid crossing between trajectories, subtract max_k from each trajectory length
        # save both the original and the modified trajectory lengths
        # when indexing into actions and filenames, we must convert from subtracted to actual index
        self.traj_lengths = np.array(traj_lengths)
        self.traj_lengths_minus_k = self.traj_lengths - max_k
        self.cumulative_lengths = np.cumsum(self.traj_lengths)
        self.cumulative_lengths_minus_k = np.cumsum(self.traj_lengths_minus_k)
        self.max_k = max_k

        self.resize = resize
        self.median_filter = median_filter

        self.load_into_memory = load_into_memory
        if self.load_into_memory:
            print("Loading full dataset into memory...")
            self.pics = torch.empty((self.pic_filenames.shape[0], self.pic_filenames.shape[1], 3, self.resize[0], self.resize[1]), dtype=torch.uint8, device=torch.device("cpu")).share_memory_()
            for i in tqdm(range(self.pic_filenames.shape[0])):
                for j in range(self.pic_filenames.shape[1]):
                        self.pics[i,j] = torch.from_numpy(self._load_pic_from_file(self.pic_filenames[i,j]))

        self.cache_into_memory = cache_into_memory
        if self.cache_into_memory:
            print("Allocating shared memory for caching dataset...")
            self.pics = torch.empty((self.pic_filenames.shape[0], self.pic_filenames.shape[1], 3, self.resize[0], self.resize[1]), dtype=torch.uint8, device=torch.device("cpu")).share_memory_()
            self.pic_is_cached = torch.zeros(self.pic_filenames.shape[0], dtype=torch.uint8, device=torch.device("cpu")).share_memory_()

    def __len__(self):
        return self.cumulative_lengths_minus_k[-1]
    
    def __getitem__(self, args):
        # check if multiple arguments
        if isinstance(args, tuple):
            idx = args[0]
            k = args[1]
        else:
            idx = args
            k = 1

        # find trajectory and offset within trajectory
        traj_idx = np.searchsorted(self.cumulative_lengths_minus_k, idx, side="right")
        if traj_idx >= len(self.cumulative_lengths_minus_k):
            raise IndexError(f"index {idx} out of range for dataset with length {len(self)}")
        if traj_idx > 0:
            # we need to re-map index to handle subtraction of max_k
            index_in_traj = idx - self.cumulative_lengths_minus_k[traj_idx-1]
            actual_idx = self.cumulative_lengths[traj_idx-1] + index_in_traj
        else:
            # we are in the first trajectory
            actual_idx = idx

        # load pictures
        # dimensions are (3 images [cam0, cam1, car], 3 channels, 256 height, 256 width)
        if self.load_into_memory:
            st_pics = self.pics[actual_idx].float() / 256.0
            stk_pics = self.pics[actual_idx+k].float() / 256.0
        elif self.cache_into_memory:
            if not self.pic_is_cached[actual_idx]:
                self.pics[actual_idx] = torch.from_numpy(self._load_pics_at_index(actual_idx))
                self.pic_is_cached[actual_idx] = True
            if not self.pic_is_cached[actual_idx+k]:
                self.pics[actual_idx+k] = torch.from_numpy(self._load_pics_at_index(actual_idx+k))
                self.pic_is_cached[actual_idx+k] = True
            st_pics = self.pics[actual_idx].float() / 256.0
            stk_pics = self.pics[actual_idx+k].float() / 256.0
        else:
            st_pics = torch.from_numpy(self._load_pics_at_index(actual_idx)).float() / 256.0
            stk_pics = torch.from_numpy(self._load_pics_at_index(actual_idx+k)).float() / 256.0

        # add 1 to index because we want the action after the initial observation
        action = self.actions[actual_idx+1]
        return (st_pics, stk_pics, k, action)

    def _load_pics_at_index(self, idx):
        filename = self.pic_filenames[idx]
        return np.array([self._load_pic_from_file(file) for file in filename], dtype=np.uint8)

    def _load_pic_from_file(self, filename):
        pic = cv.imread(filename)
        pic = cv.resize(pic, self.resize, interpolation=cv.INTER_CUBIC)
        if self.median_filter:
            pic = cv.medianBlur(pic, 3)
            pic = cv.medianBlur(pic, 5)
        # OpenCV stores images in BGR format
        pic = cv.cvtColor(pic, cv.COLOR_BGR2RGB)
        # OpenCV stores images in shape (height, width, channels)
        # Convert to shape (channels, height, width)
        return np.transpose(pic, (2, 0, 1)).astype(np.uint8)


class CarDataSampler(Sampler):
    # returns an infinite iterator of (idx, k) pairs
    # idx is chosen uniformly from the dataset
    # k is chosen uniformly from [1, max_k]
    def __init__(self, data_source, max_k=1, shuffle=True):
        self.data_source = data_source
        self.len = len(data_source)
        self.max_k = max_k
        self.shuffle = shuffle
        self.randomize()
    
    def randomize(self):
        # store a list of indices into dataset
        if self.shuffle:
            self.idx_list = torch.randperm(self.len)
        else:
            self.idx_list = torch.arange(self.len)

        # store a list of k values
        self.k_list = torch.randint(1, self.max_k+1, (self.len,))

        # index into stored lists
        self.i = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        # if we have reached the end of the list, randomize again
        if self.i >= self.len:
            raise StopIteration
            #self.randomize()

        idx = self.idx_list[self.i]
        k = self.k_list[self.i]
        self.i += 1
        return (idx, k)

    def __len__(self):
        return self.len


# Checkpoint model was trained with wrong interleaving of the 3 camera views
# This will imitate the old behavior
def interleave(img):
    assert img.dim() == 4
    batch, ch_views, h, w = img.shape
    views = 3
    ch = ch_views // views
    img_concat = img.reshape((batch,views,ch,h,w)).permute((0,2,3,1,4)).reshape((batch,ch,h,w*views))
    xbc = img_concat.reshape((batch,ch,h,w,views)).permute(0,1,4,2,3).reshape((batch,ch*views,h,w))
    return xbc

def undo_interleave(img):
    assert img.dim() == 4
    batch, ch_views, h, w = img.shape
    views = 3
    ch = ch_views // views
    img_concat = img.reshape((batch,ch,views,h,w)).permute(0,1,3,4,2).reshape((batch,ch,h,w*views))
    return img_concat.reshape((batch,ch,h,views,w)).permute(0,3,1,2,4).reshape((batch,views*ch,h,w))


# These classes only needed for loading checkpoint used for testing
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
        

if __name__ == "__main__":
    print("Loading data...")
    data_root = "./car_data"
    batch_size = 64
    max_k = 1
    dataset = CarDataset(data_root, max_k=max_k, resize=(256,256))
    sampler = CarDataSampler(dataset, max_k=max_k, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)

    print("Loading model...")
    model_file = "./models/alex_car_model_06012023.pt"
    enc, dec, acnet = torch.load(model_file)
    enc.eval()
    dec.eval()
    acnet.eval()

    a_lst = []
    r_lst = []

    print("Sampling batch...")
    for st, stk, k, a_true in dataloader:
        assert st.shape == (batch_size, 3, 3, 256, 256)
        assert stk.shape == (batch_size, 3, 3, 256, 256)
        assert k.shape == (batch_size,)
        assert a_true.shape == (batch_size, 4)

        device = torch.device("cuda:0")
        st = st.to(device)
        stk = stk.to(device)
        k = k.to(device)
        a_true = a_true.to(device)

        print("Running model...")
        loss = 0.0
        with torch.no_grad():
            # imitate the old image processing behavior?
            do_interleave = True

            st = st.reshape((batch_size, 9, 256, 256))
            stk = stk.reshape((batch_size, 9, 256, 256))
            xbc = torch.concatenate([st, stk], dim=1)
            if do_interleave:
                st = interleave(st)
                stk = interleave(stk)

            st_emb = enc(st)
            stk_emb = enc(stk)
            a_pred = acnet(st_emb.flatten(start_dim=1), stk_emb.flatten(start_dim=1), k)
            ac_loss = (((a_pred - a_true)**2).sum(dim=(1)) / (torch.sqrt(k) * math.sqrt(a_true[0].numel()))).mean()
            loss += ac_loss

            st_dec = dec(st_emb.detach())
            stk_dec = dec(stk_emb.detach())
            if do_interleave:
                st_dec = undo_interleave(st_dec)
                stk_dec = undo_interleave(stk_dec)
            xrec = torch.concatenate([st_dec, stk_dec], dim=1)
            rec_loss = ((xrec - xbc)**2).sum(dim=(1,2,3)).mean() / math.sqrt(xrec[0].numel())
            loss += rec_loss

        print("Saving images...")
        xbc_save = xbc.reshape((-1, 3, 256, 256))
        xrec_save = xrec.reshape((-1, 3, 256, 256))
        save_image(xbc_save, 'test_data/orig.jpg', nrow=6)
        save_image(xrec_save, 'test_data/xrec.jpg', nrow=6)

        break

    a_lst.append(ac_loss.item())
    r_lst.append(rec_loss.item())

    print('a-true', a_true[0])
    print('a-pred', a_pred[0])
    print('k', k.cpu().numpy())

    print('a loss', sum(a_lst)/len(a_lst))
    print('r loss', sum(r_lst)/len(r_lst))
    a_lst = []
    r_lst = []
