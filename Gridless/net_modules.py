import torch
import torch.nn as nn
import torch.functional as f
from torchinfo import summary

class ComplexSingleConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super().__init__()   
        self.conv2d_real = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias)
        self.conv2d_imag = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias)

    def forward(self, x):
        # (a+ib) * (x+iy) = (ax-by) + i(ay + xb)
        x_real=self.conv2d_real(x[..., 0]) - self.conv2d_imag(x[..., 1])
        x_imag=self.conv2d_real(x[..., 1]) + self.conv2d_imag(x[..., 0])
        # breakpoint()
        out=torch.stack((x_real, x_imag), dim=-1)
        return out

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn_real=nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, 
            track_running_stats=track_running_stats)
        self.bn_imag=nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, 
            track_running_stats=track_running_stats)
    def forward(self, x):
        x_real=self.bn_real(x[..., 0])
        x_imag=self.bn_imag(x[..., 1])
        out=torch.stack((x_real, x_imag), dim=-1)
        return out

class ComplexSingleConvBlock(nn.Module):
    def __init__(self, act_fn, c_in, c_out, k_size, stride=1, padding=0, dilation=1, 
            groups=1, use_bias=True, use_bn=True):
        super().__init__()
        self.convblock=[]
        self.convblock+=[
            ComplexSingleConv2d(c_in=c_in, c_out=c_out, k_size=k_size, stride=stride, 
                padding=padding, dilation=dilation, groups=groups, use_bias=use_bias)
            ]
        if use_bn:
            self.convblock+=[ComplexBatchNorm2d(num_features=c_out)]
        if act_fn:
            self.convblock+=[act_fn()]
        self.convblock=nn.Sequential(*self.convblock)
    def forward(self, x):
        x=self.convblock(x)
        return x

class ComplexInceptionBlock(nn.Module):
    def __init__(self, act_fn, c_in, c_red:dict, c_out:dict, k_size:dict, padding:dict, 
        stride=1, dilation=1, use_bias=True, use_bn=True, is_1dkernel:bool=True, 
        is_2dkernel:bool=True):
        
        super().__init__()
        self.is_1dkernel=is_1dkernel
        self.is_2dkernel=is_2dkernel

        self.cconv_1x1=nn.Sequential(
            ComplexSingleConvBlock(act_fn=act_fn, c_in=c_in, c_out=c_out['1x1'], 
                k_size=k_size['1x1'], stride=stride, padding=padding['1x1'], 
                dilation=dilation, use_bias=use_bias, use_bn=use_bn))
        if self.is_1dkernel:
            self.cconv_3x1=nn.Sequential(
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_in, c_out=c_red['3x1'], k_size=k_size['1x1'], 
                    stride=stride, padding=padding['1x1'], dilation=dilation, use_bias=use_bias, use_bn=use_bn),
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_red['3x1'], c_out=c_out['3x1'], k_size=k_size['3x1'], 
                    stride=stride, padding=padding['3x1'], dilation=dilation, use_bias=use_bias, use_bn=use_bn))
            self.cconv_5x1=nn.Sequential(
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_in, c_out=c_red['5x1'], k_size=k_size['1x1'], 
                    stride=stride, padding=padding['1x1'], dilation=dilation, use_bias=use_bias, use_bn=use_bn),
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_red['5x1'], c_out=c_out['5x1'], k_size=k_size['5x1'], 
                    stride=stride, padding=padding['5x1'], dilation=dilation, use_bias=use_bias, use_bn=use_bn))

        if self.is_2dkernel:
            self.cconv_1x3=nn.Sequential(
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_in, c_out=c_red['1x3'], k_size=k_size['1x1'], 
                    stride=stride, padding=padding['1x1'], dilation=dilation, use_bias=use_bias, use_bn=use_bn),
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_red['1x3'], c_out=c_out['1x3'], k_size=k_size['1x3'], 
                    stride=stride, padding=padding['1x3'], dilation=dilation, use_bias=use_bias, use_bn=use_bn))
            self.cconv_3x3=nn.Sequential(
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_in, c_out=c_red['3x3'], k_size=k_size['1x1'], 
                    stride=stride, padding=padding['1x1'], dilation=dilation, use_bias=use_bias, use_bn=use_bn),
                ComplexSingleConvBlock(act_fn=act_fn, c_in=c_red['3x3'], c_out=c_out['3x3'], k_size=k_size['3x3'], 
                    stride=stride, padding=padding['3x3'], dilation=dilation, use_bias=use_bias, use_bn=use_bn))
    
    def forward(self, x):
        x_1x1=self.cconv_1x1(x)
        if self.is_1dkernel:
            x_3x1=self.cconv_3x1(x)
            x_5x1=self.cconv_5x1(x)
            if self.is_2dkernel:
                x_1x3=self.cconv_1x3(x)
                x_3x3=self.cconv_3x3(x)
                x_real=torch.cat((x_1x1[..., 0], x_3x1[..., 0], x_5x1[..., 0], x_1x3[..., 0], x_3x3[..., 0]), dim=1)
                x_imag=torch.cat((x_1x1[..., 1], x_3x1[..., 1], x_5x1[..., 1], x_1x3[..., 1], x_3x3[..., 1]), dim=1)
            else:
                x_real=torch.cat((x_1x1[..., 0], x_3x1[..., 0], x_5x1[..., 0]), dim=1)
                x_imag=torch.cat((x_1x1[..., 1], x_3x1[..., 1], x_5x1[..., 1]), dim=1)
        elif self.is_2dkernel:
            x_1x3=self.cconv_1x3(x)
            x_3x3=self.cconv_3x3(x)
            x_real=torch.cat((x_1x1[..., 0], x_1x3[..., 0], x_3x3[..., 0]), dim=1)
            x_imag=torch.cat((x_1x1[..., 1], x_1x3[..., 1], x_3x3[..., 1]), dim=1)
        else:
            raise NotImplementedError(f'Make either is_2dkernel: {self.is_2dkernel} or is_1dkernel: {self.is_1dkernel} True.')
        out=torch.stack((x_real, x_imag), dim=-1)
        return out
    
class DoubleComplexInceptionBlock(nn.Module):
    def __init__(self, act_fn, c_in, c_red:dict, c_out1:int, c_out:dict, k_size:dict, 
        padding:dict, stride=1, dilation=1, use_bias:bool=True, use_bn:bool=True, 
        is_1dkernel:bool=True, is_2dkernel:bool=True):
        super().__init__()
        self.double_inception_block=nn.Sequential(
            ComplexInceptionBlock(act_fn=act_fn, c_in=c_in, c_red=c_red, c_out=c_out, k_size=k_size, 
                padding=padding, stride=stride, use_bias=use_bias, dilation=dilation, 
                use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel),
            ComplexInceptionBlock(act_fn=act_fn, c_in=c_out1, c_red=c_red, c_out=c_out, k_size=k_size, 
                padding=padding, stride=stride, dilation=dilation, use_bias=use_bias, use_bn=use_bn,
                is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel))
    def forward(self, x):
        x=self.double_inception_block(x)
        return x

class ComplexResNetBlock(nn.Module):
    def __init__(self, act_fn, c_in:int, c_out: int, k_size, stride1=1, stride2=(2, 1), 
        padding=0, dilation=1, use_bias=True, subsample=True):
        super().__init__()
        self.subsample=subsample
        self.net=nn.Sequential(
            ComplexSingleConv2d(c_in=c_in, c_out=c_out, k_size=k_size['3x1'], stride=stride1, 
                padding=padding['3x1'], dilation=dilation, use_bias=use_bias),
            ComplexBatchNorm2d(num_features=c_out),
            act_fn(),
            ComplexSingleConv2d(c_in=c_out, c_out=c_out, k_size=k_size['3x1'], stride=stride2, 
                padding=padding['3x1'], dilation=dilation, use_bias=use_bias),
            ComplexBatchNorm2d(num_features=c_out)
            )
        self.act_fn=act_fn()
        if self.subsample:
            self.subnet=nn.Sequential(
                ComplexSingleConv2d(c_in=c_in, c_out=c_out, k_size=k_size['3x1'], stride=stride2, 
                    padding=padding['3x1'], dilation=dilation, use_bias=use_bias))
    def forward(self, x):
        z=self.net(x)
        if self.subsample:
            x=self.subnet(x)
        out=self.act_fn(x+z)
        return out

class DoubleComplexResNetBlock(nn.Module):
    def __init__(self, act_fn, c_in, c_out, k_size, stride1=1, stride2=(2, 1),padding=0, dilation=1, 
        use_bias=True, subsample=True):
        super().__init__()
        self.double_resnet_block=[]
        for i in range(len(c_out)):
            if i == 0:
                self.double_resnet_block+=[
                    ComplexResNetBlock(act_fn=act_fn, c_in=c_in, c_out=c_out[i], k_size=k_size, 
                        stride1=stride1, stride2=stride2, padding=padding, dilation=dilation, 
                        use_bias=use_bias, subsample=subsample)]
            else:
                self.double_resnet_block+=[
                    ComplexResNetBlock(act_fn=act_fn, c_in=c_out[i-1], c_out=c_out[i], k_size=k_size, 
                        stride1=stride1, stride2=stride2, padding=padding, dilation=dilation, 
                        use_bias=use_bias, subsample=subsample)]
        self.double_resnet_block=nn.Sequential(*self.double_resnet_block)
    def forward(self, x):
        x=self.double_resnet_block(x)
        return x

class ComplexConvPool(nn.Module):
    def __init__(self, act_fn, c_in:int, c_out:int, k_size, stride, padding, dilation, 
            use_bias:bool=True, use_bn:bool=True):
        super().__init__()
        self.downnet = nn.Sequential(
            ComplexSingleConvBlock(act_fn=act_fn, c_in=c_in, c_out=c_out, k_size=k_size, stride=stride, padding=padding,
                dilation=dilation, use_bias=use_bias, use_bn=use_bn))
    def forward(self, x):
        x=self.downnet(x)
        return x

class AmpSqueezeNet(nn.Module):
    def __init__(self, act_fn1=nn.ReLU, act_fn2=nn.Sigmoid, squeeze_ratio=8, n_channels=64, use_bias:bool=True):
        super().__init__()
        self.n_channels=n_channels
        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.dense_block=nn.Sequential(
            nn.Linear(in_features=self.n_channels, out_features=self.n_channels//squeeze_ratio, bias=use_bias),
            act_fn1(),
            nn.Linear(in_features=self.n_channels//squeeze_ratio, out_features=self.n_channels, bias=use_bias),
            act_fn2()
            )
    def forward(self, x):
        x_energy=(x**2).sum(dim=-1).squeeze()
        x_avg=self.avg_pool(x_energy).squeeze()
        x_scale=self.dense_block(x_avg)
        x = x_scale[..., None, None, None] * x
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, c_in, c_out, hid_size, n_layers=3, use_bias:bool=True):
        super().__init__()
        self.lstm_block=nn.Sequential(
            nn.LSTM(c_in, hid_size, n_layers, bias=use_bias, batch_first=True, bidirectional=False, proj_size=c_out)
            )
    def forward(self, x):
        x=torch.cat((x[..., 0], x[..., 1]), dim=1).permute(0, 2, 1)
        out, _ =self.lstm_block(x)
        return out

if __name__=='__main__':
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # 1.
    model=ComplexSingleConv2d(c_in=1, c_out=16, k_size=(1, 1), stride=1, padding=1, dilation=1)
    summary(model=model.to(device), input_size=(32, 1, 8, 30, 2), depth=3)

    # 2.
    model=ComplexSingleConvBlock(act_fn=nn.ReLU, c_in=1, c_out=16, k_size=(1, 1))
    summary(model=model.to(device), input_size=(32, 1, 8, 30, 2), depth=3)

    # 3.
    model=ComplexInceptionBlock(act_fn=nn.ReLU, c_in=16, 
        c_red={'1x3': 12, '3x1': 12, '3x3': 16, '5x1': 12}, 
        c_out={'1x1': 6, '1x3': 6, '3x1': 6, '3x3': 8, '5x1': 6}, 
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]},
        padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]}, 
        stride=1, dilation=1, use_bias=True, use_bn=True, is_1dkernel=True, is_2dkernel=True)
    summary(model=model.to(device), input_size=(32, 16, 8, 30, 2), depth=3)

    # 4.
    model=DoubleComplexInceptionBlock(
        act_fn=nn.ReLU, c_in=16, c_red={'1x3': 12, '3x1': 12, '3x3': 16, '5x1': 12}, 
        c_out1=32, c_out={'1x1': 6, '1x3': 6, '3x1': 6, '3x3': 8, '5x1': 6},
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]},
        padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]},
        stride=1, dilation=1, use_bias=True, use_bn=True, is_1dkernel=True, is_2dkernel=True)
    summary(model=model.to(device), input_size=(32, 16, 8, 30, 2), depth=3, 
        col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])

    # 5.
    model=ComplexResNetBlock(act_fn=nn.ReLU, c_in=32, c_out=64, 
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]}, 
        stride1=1, stride2=(2, 1),
        padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]}, 
        dilation=1, use_bias=True, subsample=True
        )
    summary(model=model.to(device), input_size=(32, 32, 8, 30, 2), depth=3, 
        col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
    
    # 6.
    model=DoubleComplexResNetBlock(act_fn=nn.ReLU, c_in=48, c_out=[48, 64], 
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]}, 
        stride1=1, stride2=(2, 1),  
        padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]},
         dilation=1, use_bias=True, subsample=True)
    summary(model=model.to(device), input_size=(32, 48, 6, 30, 2), depth=3, 
        col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])

    # 7.
    model=ComplexConvPool(act_fn=nn.ReLU, c_in=32, c_out=32, k_size=(3, 1), stride=1, padding=(1, 0), 
        dilation=(2, 1))
    summary(model=model.to(device), input_size=(32, 32, 6, 30, 2), depth=3, 
        col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])

    # 8.
    model=RecurrentBlock(c_in=128, c_out=1, hid_size=64, n_layers=3, use_bias=True)
    summary(model=model.to(device), input_size=(512, 64, 30, 2), depth=3)

    # 9.
    model=AmpSqueezeNet(act_fn1=nn.ReLU, act_fn2=nn.Sigmoid, squeeze_ratio=8, n_channels=64, use_bias=True)
    summary(model=model.to(device), input_size=(512, 64, 1, 30, 2), depth=3)
    
    