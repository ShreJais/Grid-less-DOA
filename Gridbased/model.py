import torch
import torch.nn as nn
import torch.nn.functional as f

from torchinfo import summary

class single_conv_block(nn.Module):
    def __init__(self, act_fn, c_in:int, c_out:int, conv_stride:int=1, dilation:int=1, 
        filter_size:int=3, padding:int=1, bias:bool=True, use_bn:bool=True):
        super().__init__()
        self.conv_block = []
        self.conv_block += [nn.Conv2d(in_channels=c_in, out_channels=c_out, 
            kernel_size=filter_size, stride=conv_stride, padding=padding, 
            dilation=dilation, bias=bias)]
        if use_bn:
            self.conv_block += [nn.BatchNorm2d(num_features=c_out)]
        if act_fn != None:
            self.conv_block += [act_fn()]    
        self.conv_block = nn.Sequential(*self.conv_block)
    
    def forward(self, x):
        return self.conv_block(x)
    
class single_convt_block(nn.Module):
    def __init__(self, act_fn, c_in:int, c_out:int, filter_size:int, stride:int, 
        padding:int=1, bias:bool=True) -> None:
        super().__init__()
        self.convt_block = []
        self.convt_block += [nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, 
            kernel_size=filter_size, stride=stride, padding=padding, bias=bias)]
        self.convt_block = nn.Sequential(*self.convt_block)

    def forward(self, x):
        return self.convt_block(x)
    
class down_block(nn.Module):
    def __init__(self, act_fn, c_in, c_out, conv_stride:int=1, dilation:int=1, filter_size:int=3, 
        padding:int=1, bias:bool=True, use_bn:bool=True, pool_size: int=2, pool_stride:int=2,
        double_conv: bool=True):
        super().__init__()
        self.maxpool_double_conv = []
        self.maxpool_double_conv += [
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)]
        if double_conv:
            self.maxpool_double_conv += [
                single_conv_block(act_fn=act_fn, c_in=c_in, c_out=c_out, conv_stride=conv_stride, 
                    dilation=dilation, filter_size=filter_size, padding=padding, use_bn=use_bn, bias=bias),
                single_conv_block(act_fn=act_fn, c_in=c_out, c_out=c_out, conv_stride=conv_stride, 
                    dilation=dilation, filter_size=filter_size, padding=padding, use_bn=use_bn, bias=bias)
                    ]
        self.maxpool_double_conv = nn.Sequential(*self.maxpool_double_conv)
    
    def forward(self, x):
        return self.maxpool_double_conv(x)
    
class up_block(nn.Module):
    def __init__(
        self, act_fn, c_in, c_out, convt_filter_size, u_conv_stride, conv_filter_size, u_padding, 
        convt_stride, use_bn=False, bias=True, is_concat=True):
        super().__init__()
        self.convt_conv_block = []
        self.convt_conv_block += [ 
            single_convt_block(
                act_fn=act_fn, c_in=c_in, c_out=c_out, filter_size=convt_filter_size,
                stride=convt_stride, padding=u_padding, bias=bias)]
        if is_concat:
            self.convt_conv_block += [
                single_conv_block(
                    act_fn=act_fn, c_in=c_in, c_out=c_out, conv_stride=u_conv_stride,
                    filter_size= conv_filter_size, padding=u_padding, bias=bias, use_bn=use_bn)
            ]
        else:
            self.convt_conv_block += [
                single_conv_block(
                    act_fn=act_fn, c_in=c_out, c_out=c_out, conv_stride=u_conv_stride,
                    filter_size= conv_filter_size, padding=u_padding, bias=bias, use_bn=use_bn)
                    ]

        self.convt_conv_block = nn.Sequential(*self.convt_conv_block)
    
    def forward(self, x1, x2=None):
        x1=self.convt_conv_block[0](x1)
        if x2 != None:
            diff_y = x2.shape[2] - x1.shape[2]
            diff_x = x2.shape[3] - x1.shape[3]
            # print(diff_x, diff_y)
            x1 = f.pad(input=x1, pad=[diff_x//2, (diff_x - diff_x//2), diff_y//2, (diff_y - diff_y//2)])
            x = torch.cat([x2, x1], dim=1)
            x = self.convt_conv_block[1](x)
        else:
            x = self.convt_conv_block[1](x1)
        return x

class unet2(nn.Module):
    def __init__(
        self, act_fn, c_in=1, c_out=[32, 64, 128, 256, 512], 
        d_conv_stride:int=1, u_conv_stride:int=1, d_dilation:int=2, u_dialtion:int=1, conv_filter_size:int=3,
        convt_filter_size:list=[4,5,4,4], convt_stride:int=2, u_padding:int=1,
        d_padding:int=1, bias:bool=True, use_bn:bool=True, pool_size: int=2, pool_stride:int=2, 
        double_conv: bool=True, is_concat=True):
        super().__init__()

        self.act_fn = act_fn
        self.c_in=c_in
        self.c_out=c_out
        self.d_conv_stride=d_conv_stride
        self.u_conv_stride=u_conv_stride
        self.d_dilation=d_dilation
        self.u_dilation=u_dialtion
        self.conv_filter_size=conv_filter_size
        self.convt_filter_size=convt_filter_size
        self.u_padding=u_padding
        self.d_padding=d_padding
        self.bias=bias
        self.use_bn=use_bn
        self.pool_size=pool_size
        self.pool_stride=pool_stride
        self.double_conv=double_conv
        self.convt_stride=convt_stride
        self.is_concat=is_concat

        self.config ={
            'act_fn': act_fn.__name__, 'c_in': c_in, 'c_out': c_out, 'd_conv_stride': d_conv_stride,
            'u_conv_stride': u_conv_stride, 'd_dilation': d_dilation, 'u_dilation': u_dialtion,
            'conv_filter_size': conv_filter_size, 'convt_filter_size': convt_filter_size,
            'u_padding': u_padding, 'd_padding': d_padding, 'bias': bias, 'use_bn': use_bn,
            'pool_size': pool_size, 'pool_stride': pool_stride, 'double_conv': double_conv,
            'convt_stride': convt_stride, 'is_concat': is_concat
            }

        self.create_network()
        self._init_params()
    
    def create_network(self):
        self.input_net = nn.Sequential(
            single_conv_block(act_fn=self.act_fn, c_in=self.c_in, c_out=self.c_out[0], 
                conv_stride=self.d_conv_stride, dilation=self.d_dilation, 
                filter_size=self.conv_filter_size, padding=self.d_padding, 
                bias=self.bias, use_bn=self.use_bn),
            single_conv_block(act_fn=self.act_fn, c_in=self.c_out[0], c_out=self.c_out[0], 
                conv_stride=self.d_conv_stride, dilation=self.d_dilation,
                filter_size=self.conv_filter_size, padding=self.d_padding, 
                bias=self.bias, use_bn=self.use_bn))
        
        down_channels = self.c_out[:-1]
        # print(down_channels)
        self.down_blocks = []
        for i in range(1, len(down_channels)):
            self.down_blocks += [
                down_block(act_fn=self.act_fn, c_in=down_channels[i-1], c_out=down_channels[i],
                    conv_stride=self.d_conv_stride, dilation=self.d_dilation, 
                    filter_size=self.conv_filter_size, padding=self.d_padding, bias=self.bias, 
                    use_bn=self.use_bn, pool_size=self.pool_size, pool_stride=self.pool_stride, 
                    double_conv=self.double_conv)]
        self.double_conv=False
        self.down_blocks+=[
            down_block(act_fn=self.act_fn, c_in=down_channels[i-1], c_out=down_channels[i],
                conv_stride=self.d_conv_stride, dilation=self.d_dilation, 
                filter_size=self.conv_filter_size, padding=self.d_padding, bias=self.bias, 
                use_bn=self.use_bn, pool_size=self.pool_size, pool_stride=self.pool_stride, 
                double_conv=self.double_conv)]
    
        self.down_blocks = nn.Sequential(*self.down_blocks)

        self.mid_net=nn.Sequential(
            single_conv_block(act_fn=self.act_fn, c_in=down_channels[-1], c_out=self.c_out[-1], 
                conv_stride=self.d_conv_stride, dilation=self.u_dilation, 
                filter_size=self.conv_filter_size, padding=self.u_padding, bias=self.bias,
                use_bn=False))
        
        self.use_bn=False
        up_channels = [self.c_out[-1]] + down_channels[::-1]
        # print(up_channels)
        self.up_blocks = []
        count = 0
        for i in range(1, len(up_channels)):
            if count >= 2:
                self.is_concat=False
            self.up_blocks += [
                up_block(act_fn=self.act_fn, c_in=up_channels[i-1], c_out=up_channels[i], 
                    u_conv_stride=self.u_conv_stride, convt_filter_size=self.convt_filter_size[i-1], 
                    conv_filter_size=self.conv_filter_size, u_padding=self.u_padding, 
                    convt_stride=self.convt_stride, use_bn=False, bias=self.bias, 
                    is_concat=self.is_concat)]
            
            count += 1
        self.up_blocks = nn.Sequential(*self.up_blocks)
        if self.act_fn == nn.ReLU:
            self.output_net = nn.Sequential(
                single_conv_block(act_fn=self.act_fn, c_in=up_channels[-1], c_out=self.c_in,
                    conv_stride=self.u_conv_stride, padding=0, filter_size=1, bias=self.bias, 
                    use_bn=False))
        else:
            self.output_net = nn.Sequential(
                single_conv_block(
                    act_fn=None, c_in=up_channels[-1], c_out=self.c_in, conv_stride=self.u_conv_stride, 
                    padding=0, filter_size=1, bias=self.bias, use_bn=False),
                nn.Sigmoid()
            )
         
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # breakpoint()
        x = self.input_net(x)
        # print(x.shape)
        skip_x = []
        skip_x.append(x)
        # breakpoint()
        # print(len(self.down_blocks))
        for i, down_layers in enumerate(self.down_blocks):
            x = down_layers(x)
            # print(x.shape)
            if i!= len(self.down_blocks)-1:
                skip_x.append(x)
        # print(len(skip_x))
        # print(skip_x[0].shape)
        # print(skip_x[1].shape)
        # print(skip_x[2].shape)
        x = self.mid_net(x)
        # print(x.shape)
        count = 0
        # print(len(self.up_blocks))
        for i, up_layers in enumerate(self.up_blocks):
            if count < 2:
                x=up_layers(x,skip_x[-(i+1)])
                # print(x.shape)
            else:
                x=up_layers(x,None)
            count += 1
        x = self.output_net(x)
        return x
    
if __name__ == "__main__":
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    
    # exp1
    net=unet2(
        act_fn=nn.ReLU, c_in=1, c_out=[6, 12, 24, 48], d_conv_stride=1, 
        u_conv_stride=1, conv_filter_size=3, convt_filter_size=[4,5,5], convt_stride=2,
        d_dilation=2, u_dialtion=1, u_padding=1, d_padding=2, bias=True, use_bn=True, pool_size=2, 
        pool_stride=2, double_conv=True, is_concat=True)
    
    summary(model=net.to(device), input_size=(512, 1, 11, 161), depth=4)

    # # exp2
    # net=unet2(
    #     act_fn=nn.ReLU, c_in=1, c_out=[8, 16, 32, 64], d_conv_stride=1, 
    #     u_conv_stride=1, conv_filter_size=3, convt_filter_size=[4,5,5], convt_stride=2,
    #     d_dilation=2, u_dialtion=1, u_padding=1, d_padding=2, bias=True, use_bn=True, pool_size=2, 
    #     pool_stride=2, double_conv=True, is_concat=True)
    # summary(model=net.to(device), input_size=(1, 15, 181))


    # # exp3
    # net=unet2(
    #     act_fn=nn.ReLU, c_in=1, c_out=[4, 8, 16, 32], d_conv_stride=1, 
    #     u_conv_stride=1, conv_filter_size=3, convt_filter_size=[4,5,5], convt_stride=2,
    #     d_dilation=2, u_dialtion=1, u_padding=1, d_padding=2, bias=True, use_bn=True, pool_size=2, 
    #     pool_stride=2, double_conv=True, is_concat=True)
    # summary(model=net.to(device), input_size=(1, 11, 161))
