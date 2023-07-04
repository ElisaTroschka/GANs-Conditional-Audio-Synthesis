import torch
import torch.nn as nn


class WaveGANGenerator(nn.Module):

    def __init__(self, in_dim, cond_dim, out_dim, d=1):
        super(WaveGANGenerator, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim

        self.generator = nn.Sequential(
            nn.Linear(self.in_dim + self.cond_dim, 100),
            self._make_conv_block(100, 64, kernel_size=25),
            self._make_conv_block(64, 256, kernel_size=25),
            self._make_conv_block(256, 1024, kernel_size=25),
            self._make_conv_block(1024, 4096, kernel_size=25),
            self._make_conv_block(4096, 16384, kernel_size=25),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.generator(x)
        return output

    def _make_conv_block(self, input_dim, output_dim, kernel_size):
        return nn.Sequential(
            nn.ConvTranspose1d(input_dim, output_dim, kernel_size, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(),
            nn.ReLU()
        )


class WaveGANDiscriminator(nn.Module):
    def __init__(self, kernel_len=25, dim=64, phaseshuffle_rad=0):
        super(WaveGANDiscriminator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2)
        self.phaseshuffle_rad = phaseshuffle_rad

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, dim, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(dim, dim * 2, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(dim * 2, dim * 4, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(dim * 4, dim * 8, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(dim * 8, dim * 16, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(dim * 16, dim * 32, kernel_len, stride=2 if phaseshuffle_rad > 0 else 4, padding=kernel_len//2)
        )

        self.linear = nn.Linear(dim * 32, 1)

    def forward(self, x):
        output = x
        for layer in self.conv_layers:
            output = layer(output)
            output = self.lrelu(output)
            output = self.phaseshuffle(output)

        output = output.view(output.size(0), -1)
        output = self.linear(output)
        output = output.squeeze()

        return output

    def phaseshuffle(self, x):
        if self.phaseshuffle_rad > 0:
            rad = torch.randint(-self.phaseshuffle_rad, self.phaseshuffle_rad + 1, (1,))
            pad_l = max(rad, 0)
            pad_r = max(-rad, 0)
            phase_start = pad_r.item()

            pad = nn.ReflectionPad1d((pad_l.item(), pad_r.item()))
            x = pad(x)
            x = x[:, phase_start:phase_start + x.size(2)]

        return x
