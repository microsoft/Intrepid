import random
import torch
import multiprocessing as mp
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from utils.cuda import cuda_var
from copy import deepcopy


class Worker:

    def __init__(self):
        pass

    @staticmethod
    def forward(id, model, gpu_id): 

        # Set this process to use a different GPU
        torch.cuda.set_device(gpu_id)
        assert gpu_id == torch.cuda.current_device()
        
        a = np.eye(32)
        vector = cuda_var(torch.from_numpy(a)).float()

        output = None
        for i in range(0, 25000):  # A time consuming process
            output = model(vector)
        print("Client %r: Given GPU-ID to use %d, using GPU-ID %r out of %r, Output Sum is %r" % (id, gpu_id, torch.cuda.current_device(), torch.cuda.device_count(), output.sum()))

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.transform = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        return self.transform(x)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn')

    a = np.eye(32)
    a_v = cuda_var(torch.from_numpy(a)).float()

    model = Model()
    output = model(a_v)
    print("Master Output Sum is %r " % output.sum())

    # creating new process
    new_model = deepcopy(model)
    new_model.cuda(0)
    p1 = mp.Process(target=Worker.forward, args=(0, new_model, 0))
    p1.start()

    new_model = deepcopy(model)
    new_model.cuda(1)
    p2 = mp.Process(target=Worker.forward, args=(1, new_model, 1))
    p2.start()

    new_model = deepcopy(model)
    new_model.cuda(2)
    p3 = mp.Process(target=Worker.forward, args=(2, new_model, 2))
    p3.start()

    # wait until process is finished
    p1.join()
    p2.join()
    p3.join()
