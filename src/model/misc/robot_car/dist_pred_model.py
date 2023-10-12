


'''
Given (x[t], x[t+k]) predict k.  

Have ability to find smallest k with probability above some epsilon.  


'''

import torch
import torch.nn as nn

def argmax_first(a):
    b = torch.stack([torch.arange(a.shape[1])] * a.shape[0])
    max_values, _ = torch.max(a, dim=1)
    b[a != max_values[:, None]] = a.shape[1]
    first_max, _ = torch.min(b, dim=1)

    if torch.cuda.is_available():
       first_max = first_max.cuda() 

    return first_max

class DistPred(nn.Module):

    def __init__(self, inp_size, maxk):
        super(DistPred, self).__init__()

        self.enc = nn.Sequential(nn.Linear(inp_size*2, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512,maxk))

    def forward(self, x, xk): 
        bs = x.shape[0]
        x = x.reshape((bs, -1))
        xk = xk.reshape((bs,-1))

        h = torch.cat([x,xk],dim=1)

        py = self.enc(h)

        return py

    def predict_k(self, x, xk):

        sm = nn.Softmax(dim=1)
        py = sm(self.forward(x,xk))
 
        cdf = torch.gt(torch.cumsum(py, dim=1), 0.5).float()

        first_max = argmax_first(cdf)

        return first_max

    def loss(self, x, xk, k):

        py = self.forward(x,xk)


        ce = nn.CrossEntropyLoss()
        loss = ce(py, k)

        return loss

if __name__ == "__main__":

    dp = DistPred(64, 32*32*3).cuda()

    x = torch.randn(1,3,32,32).repeat(100,1,1,1).cuda()
    xk = torch.randn(1,3,32,32).repeat(100,1,1,1).cuda()

    y = torch.zeros(100).long().cuda()

    #y[0:25] += 4
    #y[25:50] += 5
    #y[50:75] += 6
    #y[75:100] += 1

    #y += 3

    for i in range(0,1000):
        dp.train(x,xk,y)

    kpred = dp.predict_k(x[0:1],xk[0:1])

    print(kpred)





