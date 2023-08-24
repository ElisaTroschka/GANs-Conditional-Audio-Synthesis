import torch
import torch.nn as nn
import torch.nn.init as init

class MelClassifier(nn.Module):
    def __init__(self, out_dim=1):
        super(MelClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 5, 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 5, 2, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(1024, out_dim)
        self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        output = x.view(-1, 1, 128, 128)
        output = self.conv(output)
        output.squeeze_()
        output = self.fc(output)
        return output
    
    
    
class AudioClassifier(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(AudioClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 25, stride=4, padding=11, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 25, stride=4, padding=11, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 25, stride=4, padding=11, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 25, stride=4, padding=11, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, 25, stride=4, padding=11, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 2048, 25, stride=4, padding=11, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.linear = nn.Linear(8192, out)
        self.apply(self.init_weights)
        

    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
    def forward(self, x):
        output = srlf.conv_layers(x)
        output = output.reshape(-1, 8192)
        output = self.linear(output).squeeze()
        return output