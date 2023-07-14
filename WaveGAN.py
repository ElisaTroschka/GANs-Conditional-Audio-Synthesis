import torch
import torch.nn as nn


class WaveGANGenerator(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim, sr, duration, in_ch=1, kernel_size=25):
        super(WaveGANGenerator, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.sr = sr
        self.duration = duration
        self.kernel_size = kernel_size
        self.slice_len = int(self.sr * self.duration)

        if duration <= 0:
            raise ValueError("Duration must be greater than 0.")
        self.fc = nn.Linear(self.in_dim + self.cond_dim, 1024*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size, stride=4, padding=11, output_padding=1),#//2, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size, stride=4, padding=11, output_padding=1),#//2, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size, stride=4, padding=11, output_padding=1),#//2, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size, stride=4, padding=11, output_padding=1),#//2, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(64, in_ch, kernel_size, stride=4, padding=11, output_padding=1),
            nn.Tanh()
        )


    def forward(self, x, cond):
        output = torch.cat([x, cond], dim=1)
        output = self.fc(output).view(-1, 1024, 16)
        output = self.deconv(output)
        #output = self.adjust_output_length(output)
        output = output.squeeze()
        return output

    def adjust_output_length(self, output):
        if output.size(-1) != self.slice_len:
            output = output[:, :, :self.slice_len]
        return


class WaveGANDiscriminator(nn.Module):
    def __init__(self, in_dim, cond_dim, kernel_len=25, in_ch=1, phaseshuffle_rad=0):
        super(WaveGANDiscriminator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2)
        self.phaseshuffle_rad = phaseshuffle_rad
        
        layers = [nn.Conv1d(64*(2**i), 64*(2**(i+1)), kernel_len, stride=4, padding=11) for i in range(4)]
        layers.insert(0, nn.Conv1d(in_ch, 64, kernel_len, stride=4, padding=11))
        if in_dim > 8000:
            layers.append(nn.Conv1d(1024, 2048, kernel_len, stride=4, padding=11))
        if in_dim > 16000:
            layers.append(nn.Conv1d(2048, 4096, kernel_len, stride=4, padding=11))

        self.conv_layers = nn.Sequential(*layers
            #nn.Conv1d(in_ch, 64, kernel_len, stride=4, padding=11), # (50, 1, in__dim + cond_dim)
            #nn.Conv1d(64, 128, kernel_len, stride=4, padding=11), # (50, 64, ) ((L−1)/4)+1
            #nn.Conv1d(128, 256, kernel_len, stride=4, padding=11),
            #nn.Conv1d(256, 512, kernel_len, stride=4, padding=11),
            #nn.Conv1d(512, 1024, kernel_len, stride=4, padding=11)
        )
        self.linear = nn.Linear(8192, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        output = torch.cat([x, cond], dim=1).unsqueeze(1)
        for layer in self.conv_layers:
            output = layer(output)
            output = self.lrelu(output)
            output = self._apply_phase_shuffle(output, self.phaseshuffle_rad)

        output = output.view(-1, 8192)
        output = self.linear(output).squeeze()
        output = self.sigmoid(output)

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
