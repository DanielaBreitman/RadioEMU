import torch
from torch import nn

def modulelist2sequential(module:nn.ModuleList) -> nn.Sequential:
    out = nn.Sequential()
    for layer in module:
        out.append(layer)
    return out

class CNN(nn.Module):
    def __init__(self, nconvs:int, 
                 in_ch: int, out_ch: int, dropout:bool=False,
                 hid_ch:int=None, kernel_size:int = 2, f_dropout:float=0.1,
                 stride:int=1, padding:int=0, final_act:bool=False,
                 batch_norm: bool = False, 
                 act_fn: object = nn.LeakyReLU, residual:bool=False):
        super().__init__()
        self.cnn = cnn_list(nconvs=nconvs, 
                 in_ch=in_ch, out_ch=out_ch, 
                 hid_ch=hid_ch, kernel_size=kernel_size, 
                 stride=stride, padding=padding, final_act=final_act,
                 batch_norm=batch_norm, act_fn=act_fn, residual=residual)
        self.residual = residual
        
    def forward(self, x):
        y = self.cnn(x)
        return y

def cnn_list(nconvs:int, 
                 in_ch: int, out_ch: int, dropout:bool=False,
                 hid_ch:int=None, kernel_size:int = 2, f_dropout:float=0.1,
                 stride:int=1, padding:int=0, final_act:bool=False,
                 batch_norm: bool = False, act_fn: object = nn.LeakyReLU, 
                 residual:bool = False) -> nn.ModuleList:

    act = act_fn()
    if hid_ch is None:
        hid_ch = out_ch
    conv_in = nn.Sequential(nn.ConvTranspose2d(in_ch, hid_ch, kernel_size = kernel_size, stride = stride, padding = padding), act_fn())
    conv_hid = nn.Sequential(nn.ConvTranspose2d(hid_ch, hid_ch, kernel_size = kernel_size, stride = stride, padding = padding), act_fn())
    if batch_norm:
        conv_in.append(nn.BatchNorm2d(hid_ch))
        conv_hid.append(nn.BatchNorm2d(hid_ch))
    if dropout:
        conv_in.append(nn.Dropout(f_dropout))
        conv_hid.append(nn.Dropout(f_dropout))
    cnn = modulelist2sequential(nn.ModuleList([conv_hid for i in range(nconvs-2)]))
    if final_act:
        conv_out = nn.Sequential(nn.ConvTranspose2d(hid_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding), act_fn())
    else:
        conv_out = nn.ConvTranspose2d(hid_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding)
    if residual:
        return ResNet(nn.Sequential(conv_in.extend(cnn.append(conv_out))))
    else:
        return conv_in.extend(cnn.append(conv_out))
    
class Radio_Emulator(nn.Module):
    def __init__(self, 
                 nlayers:list = [10,10,5, 5, 3], 
                 nnodes:list = [1500,1000,500, 64*20, 300],
                 out_len:list = [103,103,103,25*20,1],
                 input_len:int = 5,
                 ps_inp_shape:list=[64,5,4]):
        super().__init__()
        #shared_block = nn.Sequential(nn.Linear(nnodes[-1], nnodes[-1]),
        #                              nn.LeakyReLU())
        #shared_branch = nn.Sequential(nn.Linear(input_len, nnodes[-1]),
        #                              nn.LeakyReLU(),
        #                              )
        #shared_branch.append(modulelist2sequential(nn.ModuleList([shared_block for i in range(nlayers[0] - 1)])))
        #self.shared_branch = shared_branch
        self.ps_inp_shape = ps_inp_shape
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
        
        
        
        ps_branch = nn.Sequential(nn.Linear(inp_nnodes, nnodes[3]),
                                      nn.LeakyReLU(),
                                      )
        ps_block = nn.Sequential(nn.Linear(nnodes[3], nnodes[3]),
                                      nn.LeakyReLU())
        ps_branch.append(modulelist2sequential(nn.ModuleList([ps_block for i in range(nlayers[3] - 1)])))
        self.ps_fc = ps_branch
        
        self.cnn1 = CNN(nconvs = 2, in_ch = 64,
                              out_ch = 64, hid_ch = 64, kernel_size = (2,2), 
                              batch_norm = False, final_act = True, residual = False)
        self.cnn2 = CNN(nconvs = 2, in_ch = 64, 
                              out_ch = 64, hid_ch = 64, kernel_size = (3,2), 
                              batch_norm = False, final_act = True, residual = False)
        self.cnn2v2 = CNN(nconvs = 2, in_ch = 64, 
                              out_ch = 32, hid_ch = 32, kernel_size = (3,3), 
                              batch_norm = False, final_act = True, residual = False)
        self.cnn3 = CNN(nconvs = 2, in_ch = 32,
                              out_ch = 1, hid_ch = 16, kernel_size = (3,3), 
                              batch_norm = False, final_act = True, residual = False)
        
        tau_block = nn.Sequential(nn.Linear(nnodes[-1], nnodes[-1]),
                                      nn.LeakyReLU())
        tau_branch = nn.Sequential(nn.Linear(inp_nnodes, nnodes[-1]),
                                      nn.LeakyReLU(),
                                      )
        tau_branch.append(modulelist2sequential(nn.ModuleList([tau_block for i in range(nlayers[-1] - 2)])))
        tau_branch.append(nn.Sequential(nn.Linear(nnodes[-1], out_len[-1])))
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
        ps_1d = self.ps_fc(input_params)
        ps_2d = torch.reshape(ps_1d, (ps_1d.shape[0],)+tuple(self.ps_inp_shape))
        ps1 = self.cnn1(ps_2d)
        #print(ps1.shape)
        ps2 = self.cnn1(ps1)
        #print(ps2.shape)
        ps3 = self.cnn2(ps2)
        #print(ps3.shape)
        ps4 = self.cnn2(ps3)
        #print(ps4.shape)
        ps4 = self.cnn2v2(ps4)
        #print(ps4.shape)
        ps5 = self.cnn3(ps4)
        #print(ps5.shape)
        ps_pred = torch.reshape(ps5, (ps5.shape[0],ps5.shape[-2]*ps5.shape[-1]))
        #print(ps_pred.shape)

        return torch.cat((Tb_pred, Tr_pred, xHI_pred, ps_pred, tau_pred),1)