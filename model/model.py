import torch.nn as nn
from torch.nn.modules import dropout
import torchvision.models as models
from torch.nn import functional as F

class ReNet(nn.modules):

    def __init__(self, n_input, n_units, patch_size=(1, 1)):
        super(ReNet, self).__init__()

        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])

        # Horizontal RNN
        self.rnn_hor = nn.GRU(input_size = n_input*self.patch_size_width*self.patch_size_height, hidden_size=n_units,
        num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True)

        # Vertical RNN
        self.rnn_ver = nn.GRU(input_size = 2 * n_units, hidden_size=n_units, num_layer=1, bias=True, batch_first=True,
        dropout=0, bidirectional=True)
    
    def rnn_forward(self, x, hor_or_ver):
        assert hor_or_ver in ['hor', 'ver']

        b, n_height, n_width, n_filters = x.size()

        x = x.view(b * n_height, n_width, n_filters)
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        else:
            x, _ = self.rnn_ver(x)
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)

        return x
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)      
        x = x.contiguous()
        x = self.rnn_forward(x, 'hor') 
        x = x.permute(0, 2, 1, 3)      
        x = x.contiguous()
        x = self.rnn_forward(x, 'ver') 
        x = x.permute(0, 2, 1, 3)      
        x = x.contiguous()
        x = x.permute(0, 3, 1, 2)      
        x = x.contiguous()

        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = models.__dict__['vgg16'](pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[0]) 
        self.model = nn.Sequential(*list(self.model.children())[:16])
    
    def forward(self, x):

        b, n_channel, n_height, n_width = x.size()
        x = self.model(x)

        return x

class Architecture(nn.Module):

    def __init__(self, n_classes, usegpu=True):
        super(Architecture, self).__init__()

        self.n_classes = n_classes

        self.cnn = CNN(usegpu=usegpu)
        self.renet1 = ReNet(256, 100, usegpu=usegpu)
        self.renet2 = ReNet(100 * 2, 100, usegpu=usegpu)
        self.upsampling1 = nn.ConvTranspose2d(100 * 2, 50, kernel_size=(2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.upsampling2 = nn.ConvTranspose2d(50, 50, kernel_size=(2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU()
        self.output = nn.Conv2d(50, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.renet1(x)
        x = self.renet2(x)
        x = self.relu1(self.upsampling1(x))
        x = self.relu2(self.upsampling2(x))
        x = self.output(x)
        return x


