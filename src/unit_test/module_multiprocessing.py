import random
import torch
import multiprocessing as mp
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from utils.cuda import cuda_var


class Worker:

    def __init__(self):
        pass

    @staticmethod
    def forward(id, model, vector):
        output = model(vector)
        print("Client: %r Output Sum is %r" % (id, output.sum()))

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

    a = np.random.rand(1, 32)
    a_v = cuda_var(torch.from_numpy(a)).float()

    model = Model()
    output = model(a_v)
    print("Master Output Sum is %r " % output.sum())

    # creating new process
    p1 = mp.Process(target=Worker.forward, args=(0, model, a_v))
    p1.start()

    p2 = mp.Process(target=Worker.forward, args=(1, model, a_v))
    p2.start()

    # wait until process is finished
    p1.join()
    p2.join()
