import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
#from vit_pytorch import ViT

toPIL = transforms.ToPILImage()
 
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
   #model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model = ConvNeXt(depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], **kwargs)
    return model


class Encode_t(nn.Module):
    def __init__(self,input_shape,img_size):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.input_shape = input_shape
        self.num_inputs = self.input_shape- self.img_size
        self.convx = convnext_tiny()
        self.linear0 = nn.Linear(1000,448)
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        #self.mean_linear = nn.Linear(64, self.actions_num)
        #self.value_linear = nn.Linear(64, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        img_obs = obs[:,:self.img_size]
        data_obs = obs[:,self.img_size:]
        img_obs = img_obs.view(-1,300,300,4)
        img_obs = img_obs.permute((0, 3, 1, 2))
        g = self.convx(img_obs)
        g = F.elu(self.linear0(g))
        x = F.elu(self.linear1(data_obs))
        x = torch.cat((g,x),1)
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))
        
        return x

class MLP_S(nn.Module):
    def __init__(self,input_shape,img_size):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.input_shape = input_shape
        self.num_inputs = self.input_shape- self.img_size
        
        self.linear0 = nn.Linear(1000,448)
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        #self.mean_linear = nn.Linear(64, self.actions_num)
        #self.value_linear = nn.Linear(64, 1)

    def is_rnn(self):
        return False

    def forward(self, obs,img_out):
        data_obs = obs[:,self.img_size:]

        g = F.elu(self.linear0(img_out))
        x = F.elu(self.linear1(data_obs))
        x = torch.cat((g,x),1)
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))
        
        return x



class SimpleCovnEncode(nn.Module):
    def __init__(self,input_shape,img_size):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.input_shape = input_shape
        self.num_inputs = self.input_shape- self.img_size
        """ self.covEncoder = nn.Sequential(nn.Conv2d(3, 24, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(16),
                            nn.MaxPool2d(2),
                            nn.Conv2d(24, 48, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(32),
                            nn.MaxPool2d(2),
                            nn.Conv2d(48, 96, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(64),
                            nn.MaxPool2d(2),
                            nn.Conv2d(96, 192, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(128),
                            nn.MaxPool2d(2),
                            #nn.Conv2d(192, 3, 1),
                        ) """

        self.covEncoder = nn.Sequential(nn.Conv2d(3, 8, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(16),
                            nn.MaxPool2d(2),
                            nn.Conv2d(8, 16, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(32),
                            nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(64),
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3),
                            #nn.ELU(),
                            #nn.BatchNorm2d(128),
                            nn.MaxPool2d(2),
                            #nn.Conv2d(192, 3, 1),
                        )
        self.linear0 = nn.Linear(16384,768)
        self.linear1 = nn.Linear(9, 256)
        #self.linear0 = nn.Linear(49152,768)
        #self.linear1 = nn.Linear(10, 256)
        #self.bn1 = nn.BatchNorm1d(64)
        #self.linear11 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 512)
        
        self.linear3 = nn.Linear(512, 256)
        #self.bn2 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, 128)
        #self.bn3 = nn.BatchNorm1d(128)
        self.linear5 = nn.Linear(128, 64)
        #self.mean_linear = nn.Linear(64, self.actions_num)
        #self.value_linear = nn.Linear(64, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        img_obs = obs[:,:self.img_size]
        data_obs = obs[:,self.img_size:]
        img_obs = img_obs.view(-1,300,300,3)
        img_obs = img_obs.permute((0, 3, 1, 2))
        """ pic = toPIL(img_obs[0])
        pic.save("/home/roland/MyNewTask/rgb111.png")
        print('Saved the img from the  camera..................') """
        g = self.covEncoder(img_obs)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',g.shape)
        g = g.flatten(1)
        g = F.elu(self.linear0(g))
        #g = self.bn0(g)
        
        x = F.elu(self.linear1(data_obs))
        #x = self.bn1(x)
        x = torch.cat((g,x),1)
        #x = F.elu(self.linear11(x))
        x = F.elu(self.linear2(x))
        #x = self.bn2(x)
        #x, hn  = self.gru(x)
        x = F.elu(self.linear3(x))
        #x = self.bn3(x)
        x = F.elu(self.linear4(x))

        x = F.elu(self.linear5(x))
        return x


class TestNet(nn.Module):   
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        self.actions_num = kwargs.pop('actions_num')
        self.input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)
        self.img_size = 270000
        self.num_inputs = self.input_shape[0]- self.img_size
        #assert(type(input_shape) is dict)
        #for k,v in input_shape.items():
        self.convx = convnext_tiny()
        

        self.central_value = params.get('central_value', False)


        # need to change when the output number is not 64
        self.mu = torch.nn.Linear(64, self.actions_num)
        #self.sigma = torch.nn.Linear(64, self.actions_num)
        self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.value = torch.nn.Linear(64, self.value_size)
        self.encode = SimpleCovnEncode(self.input_shape[0],self.img_size).cuda()
        self.act = nn.Identity()
    def is_rnn(self):
        return False

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        
        """ img_obs = obs[:,:self.img_size]
        data_obs = obs[:,self.img_size:]
        img_obs = img_obs.view(-1,300,300,3)
        img_obs = img_obs.permute((0, 3, 1, 2))
        img_out = self.convx(img_obs) """
        
        states = obs_dict.get('rnn_states', None)

        a_out = c_out = obs

        a_out = self.encode.forward(a_out)
        c_out = self.encode.forward(c_out)


        
        mu = self.act(self.mu(a_out))
        #sigma = self.act(self.sigma(a_out))
        sigma = mu * 0.0 + self.act(self.sigma)
        value = self.act(self.value(c_out))
        #print('shape of mu sigma value:.........................',mu.shape,sigma.shape,value.shape)
        return mu, sigma, value, states


from rl_games.algos_torch.network_builder import NetworkBuilder

class TestNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TestNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
