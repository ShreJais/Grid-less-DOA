import torch
import torch.nn as nn
import torch.nn.functional as f
from torchinfo import summary
import net_modules

class ComplexEncoderBlock(nn.Module):
    def __init__(self, act_fn, c_in:int, c_outsum:list, c_out, c_red, k_size:dict, conv_stride, conv_padding,  
        conv_dilation, pool_stride, pool_dilation, pool_padding, 
        use_bias:bool=True, use_bn:bool=True, is_1dkernel=True, is_2dkernel:bool=True,
        n_layers=2, n_inception=2):
        
        super().__init__()
        self.n_layers=n_layers
        self.n_inception=n_inception

        self.input_net=nn.Sequential(
            net_modules.ComplexSingleConvBlock(
                act_fn=act_fn, c_in=c_in, c_out=c_outsum[0], k_size=k_size['1x1'], stride=conv_stride, 
                padding=conv_padding['1x1'], dilation=conv_dilation, use_bias=use_bias, use_bn=use_bn))
        self.encoder_block=[]
        self.pool_block=[]

        for i in range(n_layers):
            if self.n_inception == 2:
                self.encoder_block+=[
                    net_modules.DoubleComplexInceptionBlock(
                        act_fn=act_fn, c_in=c_outsum[i], c_red=c_red[i], c_out1=c_outsum[i+1], c_out=c_out[i], 
                        k_size=k_size, padding=conv_padding, dilation=conv_dilation, stride=conv_stride, 
                        use_bias=use_bias, use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel)]
            else:
                self.encoder_block+=[
                    net_modules.ComplexInceptionBlock(
                        act_fn=act_fn, c_in=c_outsum[i], c_red=c_red[i], c_out=c_out[i], 
                        k_size=k_size, padding=conv_padding, dilation=conv_dilation, stride=conv_stride, 
                        use_bias=use_bias, use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel)]
            self.pool_block+=[
                net_modules.ComplexConvPool(act_fn=act_fn, c_in=c_outsum[i+1], c_out=c_outsum[i+1], k_size=k_size['3x1'], 
                    stride=pool_stride, padding=pool_padding, dilation=pool_dilation, 
                    use_bias=use_bias, use_bn=use_bn)]
    
        self.encoder_block=nn.Sequential(*self.encoder_block)
        self.pool_block=nn.Sequential(*self.pool_block)
    def forward(self, x):
        skip_x = []
        x = self.input_net(x)
        for i in range(self.n_layers):
            x=self.encoder_block[i](x)
            skip_x.append(x)
            x=self.pool_block[i](x)
        return x, skip_x[0], skip_x[1]

class AmpDecoderBlock(nn.Module):
    def __init__(self, act_fn1, act_fn2, c_in, c_outsum, c_out, c_red, k_size:dict, conv_stride, conv_dilation, 
        conv_padding, pool_stride, pool_dilation, pool_padding, 
        use_bias:bool=True, use_bn:bool=True, is_1dkernel=True, is_2dkernel:bool=True, 
        is_concat=True, squeeze_ratio: int=8, is_skip:bool=True, is_se_block:bool=True, n_inception=2):
        super().__init__()
        self.is_concat=is_concat
        self.is_skip=is_skip
        self.is_se_block=is_se_block
        self.n_inception=n_inception

        if self.n_inception==2:
            self.decoder_block1=net_modules.DoubleComplexInceptionBlock(
                act_fn=act_fn1, c_in=c_in, c_red=c_red, c_out1=c_outsum, c_out=c_out, k_size=k_size, 
                padding=conv_padding, dilation=conv_dilation, stride=conv_stride, use_bias=use_bias, 
                use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel)
        else:
            self.decoder_block1=net_modules.ComplexInceptionBlock(
                act_fn=act_fn1, c_in=c_outsum, c_red=c_red, c_out=c_out, 
                k_size=k_size, padding=conv_padding, dilation=conv_dilation, stride=conv_stride, 
                use_bias=use_bias, use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel)
        
        if self.is_skip:
            pool_c_in=int(2*c_outsum) if self.is_concat else c_outsum
        else:
            pool_c_in=c_outsum
        
        self.pool_block=net_modules.ComplexConvPool(act_fn=act_fn1, c_in=pool_c_in, c_out=c_outsum, k_size=k_size['3x1'], 
            stride=pool_stride, padding=pool_padding, dilation=pool_dilation, 
            use_bias=use_bias, use_bn=use_bn)
        self.decoder_block2=net_modules.ComplexSingleConvBlock(
            act_fn=nn.Identity, c_in=pool_c_in, c_out=c_outsum, k_size=k_size['2x1'], stride=conv_stride, 
            padding=conv_padding['2x1'], dilation=conv_dilation, use_bias=use_bias, use_bn=use_bn)
        
        self.squeeze_layer=net_modules.AmpSqueezeNet(act_fn1=act_fn1, act_fn2=act_fn2, squeeze_ratio=squeeze_ratio,
            n_channels=c_outsum, use_bias=use_bias)
        self.final_layer=nn.Sequential(
            net_modules.ComplexSingleConv2d(c_in=c_outsum, c_out=1, k_size=k_size['1x1'], stride=conv_stride, 
                padding=conv_padding['1x1'], dilation=conv_dilation, use_bias=use_bias))
        
    def forward(self, x, x0=None, x1=None):
        x=self.decoder_block1(x)
        if self.is_skip:
            x=torch.cat((x, x0), dim=1) if self.is_concat else x+x0
        x=self.pool_block(x)
        if self.is_skip:
            x=torch.cat((x, x1), dim=1) if self.is_concat else x+x1
        x=self.decoder_block2(x)
        if self.is_se_block:
            x=self.squeeze_layer(x)
        x=self.final_layer(x).squeeze()
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

class DoaTrajDecoderBlock(nn.Module):
    def __init__(self, act_fn1, act_fn3, c_in, c_outsum, c_out, c_red, k_size:dict, conv_stride, conv_padding, 
        conv_dilation, pool_stride, pool_dilation,  pool_padding, 
        use_bias:bool=True, use_bn:bool=True, is_1dkernel=True, is_2dkernel:bool=True, 
        is_concat=False, is_skip:bool=False, rnn_hid_size=64, 
        rnn_nlayers=3, n_snap=30, n_inception=2):
        super().__init__()
        self.is_concat=is_concat
        self.is_skip=is_skip
        self.n_inception =n_inception
        
        if self.n_inception==2:
            self.decoder_block1=net_modules.DoubleComplexInceptionBlock(
                act_fn=act_fn1, c_in=c_in, c_red=c_red, c_out1=c_outsum, c_out=c_out, k_size=k_size, 
                padding=conv_padding, dilation=conv_dilation, stride=conv_stride, use_bias=use_bias, 
                use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel)
        else:
            self.decoder_block1=net_modules.ComplexInceptionBlock(
                act_fn=act_fn1, c_in=c_outsum, c_red=c_red, c_out=c_out, 
                k_size=k_size, padding=conv_padding, dilation=conv_dilation, stride=conv_stride, 
                use_bias=use_bias, use_bn=use_bn, is_1dkernel=is_1dkernel, is_2dkernel=is_2dkernel)
        
        if self.is_skip:
            pool_c_in=int(2*c_outsum) if self.is_concat else c_outsum
        else:
            pool_c_in=c_outsum
        self.pool_block=net_modules.ComplexConvPool(act_fn=act_fn1, c_in=pool_c_in, c_out=c_outsum, 
            k_size=k_size['3x1'], stride=pool_stride, padding=pool_padding, dilation=pool_dilation, 
            use_bias=use_bias, use_bn=use_bn)
        self.decoder_block2=net_modules.ComplexSingleConvBlock(
            act_fn=act_fn3, c_in=c_outsum, c_out=c_outsum, k_size=k_size['2x1'], stride=conv_stride, 
            padding=conv_padding['2x1'], dilation=conv_dilation, use_bias=use_bias, use_bn=use_bn)
        self.recurrent_layer=RecurrentBlock(c_in=2*c_outsum, c_out=1, hid_size=rnn_hid_size, 
            n_layers=rnn_nlayers, use_bias=use_bias)
        self.dense_layer=nn.Linear(in_features=30, out_features=2, bias=use_bias)
        self.act_fn3=act_fn3()

    def forward(self, x, x0=None, x1=None):
        x=self.decoder_block1(x)
        if self.is_skip:
            x=torch.cat((x, x0), dim=1) if self.is_concat else x+x0
        x=self.pool_block(x)
        x=self.decoder_block2(x).squeeze()
        x_rnn=self.recurrent_layer(x).squeeze()
        x=self.act_fn3(self.dense_layer(x_rnn))
        return x, x_rnn

class GridlessModel(nn.Module):
    def __init__(self, act_fn1, act_fn2, act_fn3, 
            c_in: int, c_outsum:list, c_out:list, c_red: list, k_size: dict, 
            conv_stride, conv_padding: dict, conv_dilation, 
            pool_stride, pool_padding, pool_dilation, 
            use_bias:bool=True, use_bn:bool=True, is_1dkernel=True, is_2dkernel:bool=True,
            n_layers=2, n_inception=2, 
            resnet_stride1=[1, 1], resnet_stride2=[2, 1], resnet_subsample:bool=True,
            is_concat=True, is_skip=True, is_ampskip=True, is_doaskip=False, squeeze_ratio=8, 
            is_se_block=True, rnn_hid_size=64, rnn_nlayers=3, n_snap=30):
        super().__init__()
        self.act_fn1=act_fn1
        self.act_fn2=act_fn2 # nn.Identity
        self.act_fn3=act_fn3 # nn.Tanh
        self.c_in=c_in
        self.c_out=c_out
        self.c_red=c_red
        self.c_outsum=c_outsum
        self.k_size=k_size
        self.conv_stride=conv_stride
        self.conv_padding=conv_padding
        self.conv_dilation=conv_dilation
        self.pool_stride=pool_stride
        self.pool_padding=pool_padding
        self.pool_dilation=pool_dilation
        self.use_bias=use_bias
        self.use_bn=use_bn
        self.is_1dkernel=is_1dkernel
        self.is_2dkernel=is_2dkernel
        self.n_layers=n_layers
        self.n_inception=n_inception
        self.resnet_stride1=resnet_stride1
        self.resnet_stride2=resnet_stride2
        self.resnet_subsample=resnet_subsample
        self.is_concat=is_concat
        self.is_skip=is_skip
        self.is_ampskip=is_ampskip
        self.is_doaskip=is_doaskip
        self.is_se_block=is_se_block
        self.squeeze_ratio=squeeze_ratio
        self.rnn_hid_size=rnn_hid_size
        self.rnn_nlayers=rnn_nlayers
        self.n_snap=n_snap

        self.config ={
            'act_fn_used_throught': self.act_fn1.__name__, 
            'act_fn1_for_se_net': self.act_fn2.__name__,
            'act_fn2_for_trajectory_output': self.act_fn3.__name__, 
            'c_in': self.c_in, 'c_outsum': self.c_outsum, 'c_out': self.c_out, 'c_red': self.c_red,
            'kernel_size_used': self.k_size, 'convolution_stride': self.conv_stride, 
            'convolution_padding': self.conv_padding, 'convolution_dilation': self.conv_dilation, 
            'pool_stride': self.pool_stride, 'pool_padding': self.pool_padding, 
            'pool_dilation': self.pool_dilation,
            'use_bias_used_throught': self.use_bias, 'use_bn_used_throught': self.use_bn, 
            'is_1dkernel_for_inception': self.is_1dkernel, 'is_2dkernel_for_inception': self.is_2dkernel, 
            'n_layers_for_encoder': self.n_layers, 'num_inception_blocks': self.n_inception, 
            'is_concat': self.is_concat, 'is_skip_connection': self.is_skip,  
            'is_ampskip': self.is_ampskip, 'is_doaskip': self.is_doaskip, 'is_se_block': self.is_se_block,
            'squeeze_ratio_for_se_net': self.squeeze_ratio, 'rnn_hid_size': self.rnn_hid_size, 
            'rnn_nlayers': self.rnn_nlayers,'n_snap': self.n_snap, 
            'resnet_stride1': self.resnet_stride1, 'resnet_stride2': self.resnet_stride2,
            'subsample': self.resnet_subsample
            }
        
        self.create_network()
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def create_network(self):
        #########################################################################################################
        # Encoder Block.
        self.encoder_block=ComplexEncoderBlock(act_fn=self.act_fn1, c_in=self.c_in, c_outsum=self.c_outsum, 
            c_out=self.c_out, c_red=self.c_red, k_size=self.k_size, conv_stride=self.conv_stride, 
            conv_padding=self.conv_padding, conv_dilation=self.conv_dilation, pool_stride=self.pool_stride, 
            pool_padding=self.pool_padding, pool_dilation=self.pool_dilation, use_bias=self.use_bias, 
            use_bn=self.use_bn, is_1dkernel=self.is_1dkernel, is_2dkernel=self.is_2dkernel, 
            n_layers=self.n_layers, n_inception=self.n_inception)
        
        ###########################################################################################################
        # Resnet blocks.
        if self.is_skip:
            c_in_res1, c_out_res1=self.c_outsum[1], self.c_outsum[3]
            self.resnet_block1=net_modules.ComplexResNetBlock(
                act_fn=self.act_fn1, c_in=c_in_res1, c_out=c_out_res1, k_size=self.k_size, 
                stride1=self.resnet_stride1, stride2=self.resnet_stride2, padding=self.conv_padding, 
                dilation=self.conv_dilation, use_bias=self.use_bias, subsample=self.resnet_subsample)
            
            c_in_res2, c_out_res2 = self.c_outsum[2], self.c_outsum[2:]
            self.resnet_block2=net_modules.DoubleComplexResNetBlock(
                act_fn=self.act_fn1, c_in=c_in_res2, c_out=c_out_res2, k_size=self.k_size, 
                stride1=self.resnet_stride1, stride2=self.resnet_stride2, padding=self.conv_padding, 
                dilation=self.conv_dilation, use_bias=self.use_bias, subsample=self.resnet_subsample,)        
        
        ############################################################################################################
        # Amp Decoder Block
        amp_c_in, amp_c_outsum = self.c_outsum[2], self.c_outsum[-1] 
        amp_c_out, amp_c_red=self.c_out[-1], self.c_red[-1]
        self.amp_decoder_block=AmpDecoderBlock(
            act_fn1=self.act_fn1, act_fn2=self.act_fn2, c_in=amp_c_in, c_outsum=amp_c_outsum,
            c_out=amp_c_out, c_red=amp_c_red, k_size=self.k_size, 
            conv_stride=self.conv_stride, conv_padding=self.conv_padding, 
            conv_dilation=self.conv_dilation, pool_stride=self.pool_stride, 
            pool_padding=self.pool_padding, pool_dilation=self.pool_dilation, 
            use_bias=self.use_bias, use_bn=self.use_bn, is_1dkernel=self.is_1dkernel, 
            is_2dkernel=self.is_2dkernel, is_concat=self.is_concat,
            squeeze_ratio=self.squeeze_ratio, is_skip=(self.is_ampskip and self.is_skip), 
            is_se_block=self.is_se_block, n_inception=self.n_inception)
        
        #############################################################################################################
        # DOA trajectory Block.
        doa_c_in, doa_c_outsum=self.c_outsum[2], self.c_outsum[-1]
        doa_c_out, doa_c_red=self.c_out[-1], self.c_red[-1]
        
        self.doatraj_decoder_block=DoaTrajDecoderBlock(
            act_fn1=self.act_fn1, act_fn3=self.act_fn3, c_in=doa_c_in, c_outsum=doa_c_outsum, 
            c_out=doa_c_out, c_red=doa_c_red, k_size=self.k_size, 
            conv_stride=self.conv_stride, conv_padding=self.conv_padding, 
            conv_dilation=self.conv_dilation, pool_stride=self.pool_stride, 
            pool_padding=self.pool_padding, pool_dilation=self.pool_dilation, 
            use_bias=self.use_bias, use_bn=self.use_bn, is_1dkernel=self.is_1dkernel, 
            is_2dkernel=self.is_2dkernel, 
            is_concat=self.is_concat, is_skip=(self.is_doaskip and self.is_skip), 
            rnn_hid_size=self.rnn_hid_size, rnn_nlayers=self.rnn_nlayers, n_snap=self.n_snap, 
            n_inception=self.n_inception, )
    def forward(self, x):
        # breakpoint()
        x, x0, x1=self.encoder_block(x)
        x_res1_out, x_res2_out=self.resnet_block1(x0), self.resnet_block2(x1)
        x_amp=self.amp_decoder_block(x, x_res1_out, x_res2_out)
        x_doa_param, x_doa_track=self.doatraj_decoder_block(x)
        return x_amp, x_doa_param, x_doa_track

if __name__=='__main__':
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # 1.
    model=ComplexEncoderBlock(act_fn=nn.ReLU, c_in=1, c_outsum=[16, 32, 48, 64], 
        c_out=[{'1x1': 6, '1x3': 6, '3x1': 6, '3x3': 8, '5x1': 6},
                    {'1x1': 8, '1x3': 8, '3x1': 8, '3x3': 16, '5x1': 8},
                        {'1x1': 10, '1x3': 10, '3x1': 10, '3x3': 24, '5x1': 10}],
        c_red=[{'1x3': 12, '3x1': 12, '3x3': 16, '5x1': 12},
                    {'1x3': 16, '3x1': 16, '3x3': 32, '5x1': 16},
                        {'1x3': 20, '3x1': 20, '3x3': 48, '5x1': 20}], 
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]},
        conv_padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]},
        conv_stride=(1, 1), conv_dilation=(1, 1), pool_stride=(1, 1), pool_dilation=(2, 1), 
        pool_padding=(1, 0), use_bias=True, use_bn=True, is_1dkernel=True, is_2dkernel=True, 
        n_layers=2, n_inception=2)
    
    summary(model=model.to(device), input_size=(512, 1, 8, 30, 2), depth=8, 
        col_names=['input_size', 'output_size', 'num_params'])
    total_params=sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    
    # 2.
    model=AmpDecoderBlock(
        act_fn1=nn.ReLU, act_fn2=nn.Sigmoid, c_in=48, c_outsum=64, 
        c_out={'1x1': 10, '1x3': 10, '3x1': 10, '3x3': 24, '5x1': 10},
        c_red={'1x3': 20, '3x1': 20, '3x3': 48, '5x1': 20},
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]},
        conv_stride=(1, 1), 
        conv_padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]}, 
        conv_dilation=(1, 1), pool_stride=(1, 1), pool_padding=(1, 0), pool_dilation=(2, 1), use_bias=True, 
        use_bn=True, is_1dkernel=True, is_2dkernel=True, is_concat=True, is_skip=True, squeeze_ratio=8, is_se_block=True, 
        n_inception=2)
    summary(model=model.to(device), input_size=((512, 48, 4, 30, 2), (512, 64, 4, 30, 2), (512, 64, 2, 30, 2)), depth=8, 
        col_names=['input_size', 'output_size', 'num_params'])

    # 3.
    model=DoaTrajDecoderBlock(
        act_fn1=nn.ReLU, act_fn3=nn.Tanh, c_in=48, c_outsum=64, 
        c_out={'1x1': 10, '1x3': 10, '3x1': 10, '3x3': 24, '5x1': 10}, 
        c_red={'1x3': 20, '3x1': 20, '3x3': 48, '5x1': 20}, 
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '5x1': [5, 1]},
        conv_stride=(1, 1), 
        conv_padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]}, 
        conv_dilation=(1, 1), pool_stride=(1, 1), pool_padding=(1, 0), pool_dilation=(2, 1), 
        use_bias=True, use_bn=True, is_1dkernel=True, is_2dkernel=True, is_concat=False, is_skip=False, 
        rnn_hid_size=64, rnn_nlayers=3, n_snap=30, n_inception=2)
    summary(model=model.to(device), input_size=(512, 48, 4, 30, 2), depth=5)

    # 4.
    model=GridlessModel(
        act_fn1=nn.ReLU, act_fn2=nn.Sigmoid, act_fn3=nn.Tanh, 
        c_in=1, c_outsum=[16, 32, 48, 64],
        c_out=[{'1x1': 6, '1x3': 6, '3x1': 6, '3x3': 8, '5x1': 6},
                {'1x1': 8, '1x3': 8, '3x1': 8, '3x3': 16, '5x1': 8},
                    {'1x1': 10, '1x3': 10, '3x1': 10, '3x3': 24, '5x1': 10}],
        c_red=[{'1x3': 12, '3x1': 12, '3x3': 16, '5x1': 12},
                    {'1x3': 16, '3x1': 16, '3x3': 32, '5x1': 16},
                        {'1x3': 20, '3x1': 20, '3x3': 48, '5x1': 20}],
        k_size={'1x1': [1, 1], '1x3': [1, 3], '2x1': [2, 1], '3x1': [3, 1], '3x3': [3, 3], '3x5': [3, 5],'5x1': [5, 1]},
        conv_stride=1,
        conv_padding={'1x1': [0, 0], '1x3': [0, 1], '2x1': [0, 0], '3x1': [1, 0], '3x3': [1, 1], '5x1': [2, 0]},
        conv_dilation=1, pool_stride=(1, 1), pool_padding=(1, 0), pool_dilation=(2, 1), 
        use_bias=True, use_bn=True, is_1dkernel=True, is_2dkernel=True, n_layers=2, n_inception=2, 
        resnet_stride1=(1, 1), resnet_stride2=(2, 1), resnet_subsample=True, is_concat=True, 
        is_skip=True, is_ampskip=True, is_doaskip=False, squeeze_ratio=8, is_se_block=True, rnn_hid_size=64, 
        rnn_nlayers=3, n_snap=30, 
        )

    summary(model=model.to(device), input_size=(512, 1, 8, 30, 2), depth=5,
            col_names=['input_size', 'output_size', 'num_params'])
    
