import torch
from torch import nn


def modulelist2sequential(module:nn.ModuleList) -> nn.Sequential:
    out = nn.Sequential()
    for layer in module:
        out.append(layer)
    return out

class Radio_Emulator(nn.Module):
    def __init__(self, 
                 nlayers:list = [10,10,5, 3], 
                 nnodes:list = [1500,1000,500, 300],
                 out_len:list = [101,101,101,1],
                 input_len:int = 5):
        super().__init__()
        #shared_block = nn.Sequential(nn.Linear(nnodes[-1], nnodes[-1]),
        #                              nn.LeakyReLU())
        #shared_branch = nn.Sequential(nn.Linear(input_len, nnodes[-1]),
        #                              nn.LeakyReLU(),
        #                              )
        #shared_branch.append(modulelist2sequential(nn.ModuleList([shared_block for i in range(nlayers[0] - 1)])))
        #self.shared_branch = shared_branch
        
        inp_nnodes = input_len
        Tb_block = nn.Sequential(nn.Linear(nnodes[0], nnodes[0]),
                                      nn.LeakyReLU())
        Tb_branch = nn.Sequential(nn.Linear(inp_nnodes, nnodes[0]),
                                      nn.LeakyReLU(),
                                      )
        Tb_branch.append(modulelist2sequential(nn.ModuleList([Tb_block for i in range(nlayers[0] - 2)])))
        Tb_branch.append(nn.Sequential(nn.Linear(nnodes[0], out_len[0])))
        self.Tb_branch = Tb_branch
        
        Tr_block = nn.Sequential(nn.Linear(nnodes[1], nnodes[1]),
                                      nn.LeakyReLU())
        Tr_branch = nn.Sequential(nn.Linear(inp_nnodes, nnodes[1]),
                                      nn.LeakyReLU(),
                                      )
        Tr_branch.append(modulelist2sequential(nn.ModuleList([Tr_block for i in range(nlayers[1] - 2)])))
        Tr_branch.append(nn.Sequential(nn.Linear(nnodes[1], out_len[1])))
        self.Tr_branch = Tr_branch
        
        xHI_block = nn.Sequential(nn.Linear(nnodes[2], nnodes[2]),
                                      nn.LeakyReLU())
        xHI_branch = nn.Sequential(nn.Linear(inp_nnodes, nnodes[2]),
                                      nn.LeakyReLU(),
                                      )
        xHI_branch.append(modulelist2sequential(nn.ModuleList([xHI_block for i in range(nlayers[2] - 2)])))
        xHI_branch.append(nn.Sequential(nn.Linear(nnodes[2], out_len[2]), nn.Sigmoid()))
        self.xHI_branch = xHI_branch
        
        tau_block = nn.Sequential(nn.Linear(nnodes[3], nnodes[3]),
                                      nn.LeakyReLU())
        tau_branch = nn.Sequential(nn.Linear(inp_nnodes, nnodes[3]),
                                      nn.LeakyReLU(),
                                      )
        tau_branch.append(modulelist2sequential(nn.ModuleList([tau_block for i in range(nlayers[3] - 2)])))
        tau_branch.append(nn.Sequential(nn.Linear(nnodes[3], out_len[3])))
        self.tau_branch = tau_branch
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            # So that xHI starts around 0
            torch.nn.init.uniform_(m.bias, -4.,-1.)
                

    def forward(self, input_params):
        #shared = self.shared_branch(input_params)
        #self.xHI_branch.apply(self.init_weights)
        Tb_pred = self.Tb_branch(input_params)
        Tr_pred = self.Tr_branch(input_params)
        xHI_pred = self.xHI_branch(input_params)
        tau_pred = self.tau_branch(input_params)

        return torch.cat((Tb_pred, Tr_pred, xHI_pred, tau_pred),1)