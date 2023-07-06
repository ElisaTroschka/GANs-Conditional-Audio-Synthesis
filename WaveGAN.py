import torch
import torch.nn as nn


class WaveGANGenerator(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim, sr, duration, kernel_len=25):
        super(WaveGANGenerator, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.sr = sr
        self.duration = duration
        self.slice_len = int(self.sr * self.duration)

        if duration <= 0:
            raise ValueError("Duration must be greater than 0.")

        self.generator = nn.Sequential(
            nn.Linear(self.in_dim + self.cond_dim, 100),
            self._make_conv_block(100, 64, kernel_size=kernel_len),
            self._make_conv_block(64, 256, kernel_size=kernel_len),
            self._make_conv_block(256, 1024, kernel_size=kernel_len),
            self._make_conv_block(1024, 4096, kernel_size=kernel_len),
            self._make_conv_block(4096, self.slice_len, kernel_size=kernel_len),
            nn.Tanh()
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        output = self.generator(x)
        output = self.adjust_output_length(output)
        return output

    def _make_conv_block(self, input_dim, output_dim, kernel_size):
        return nn.Sequential(
            nn.ConvTranspose1d(input_dim, output_dim, kernel_size, stride=4, padding=0, bias=True, padding_mode='same'),
            nn.ReLU()
        )

    def adjust_output_length(self, output):
        if output.size(-1) != self.slice_len:
            output = output[:, :, :self.slice_len]
        return output


class WaveGANDiscriminator(nn.Module):
    def __init__(self, in_dim, cond_dim, kernel_len=25, phaseshuffle_rad=0):
        super(WaveGANDiscriminator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2)
        self.phaseshuffle_rad = phaseshuffle_rad

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_dim + cond_dim, 64, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(64, 128, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(128, 256, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(256, 512, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(512, 1024, kernel_len, stride=4, padding=kernel_len//2),
            nn.Conv1d(1024, 2048, kernel_len, stride=2 if phaseshuffle_rad > 0 else 4, padding=kernel_len//2)
        )

        self.linear = nn.Linear(2048, 1)

    def forward(self, x, cond):
        output = torch.cat([x, cond], dim=1)
        for layer in self.conv_layers:
            output = layer(output)
            output = self.lrelu(output)
            output = self._apply_phase_shuffle(output)

        output = output.view(output.size(0), -1)
        output = self.linear(output)
        output = output.squeeze()

        return output

    def _apply_phase_shuffle(self, x, rad, pad_type='reflect'):
        if self.phaseshuffle_rad > 0:
            b, x_len, channels = x.size()

            phase = torch.randint(-rad, rad + 1, (1,))
            pad_l = max(phase, 0)
            pad_r = max(-phase, 0)
            phase_start = pad_r.item()
            x = nn.functional.pad(x, (pad_l.item(), pad_r.item()), mode=pad_type)

            x = x[:, phase_start:phase_start + x_len]
            # x = x.contiguous()
            x.view(b, x_len, channels)

        return x
