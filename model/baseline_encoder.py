import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def get_encoder(encoder_name, input_size, input_dims, input_channels):
    try:
        if encoder_name == "CBR":
            encoder = Encoder(input_size, in_channels=input_channels)
            encoder_out_dims = encoder.output_dims
        else:
            encoder = FCEncoder(input_dims)
            encoder_out_dims = encoder.hidden_dims[-1]
        return encoder, encoder_out_dims
    except:
        raise Exception("Invalid model name. Check the config file and pass one of: resnet18 or resnet50 or CBR or "
                        "FC")


class Encoder(nn.Module):
    def __init__(self, input_size, in_channels=1, hidden_dims=None):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.in_channels = in_channels
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()
            ))
            in_channels = h_dim
        modules.append(nn.Flatten())

        self.encoder = nn.Sequential(*modules)
        tmp = torch.zeros((2, self.in_channels, self.input_size, self.input_size))
        self.output_dims = self.encoder.forward(tmp).shape[1]

    def forward(self, x):
        h = self.encoder(x)
        return h


class FCEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims=None):
        nn.Module.__init__(self)

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 512]
        self.hidden_dims = hidden_dims
        modules = []

        in_dim = in_features
        for dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ))
            in_dim = dim
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)
