import torch.nn as nn
import torch.nn.init as init


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


class AudioClassifier(nn.Module):
    """
    Audio classifier used for computation on IS
    """
    def __init__(self, out_dim=1):
        """
        Audio classifier initialization
        :param out_dim: number of output classes
        """
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
        self.linear = nn.Linear(8192, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.apply(init_weights)

    def forward(self, x):
        output = self.conv_layers(x.unsqueeze(1))
        output = output.reshape(-1, 8192)
        output = self.linear(output).squeeze()
        output = self.softmax(output)
        return output
