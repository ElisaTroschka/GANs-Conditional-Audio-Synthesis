import torch
import torch.nn as nn
import torch.nn.init as init



class SpecGANGenerator(nn.Module):
    
    def __init__(self, in_dim, cond_dim, out_dim, in_ch=1, kernel_size=5):
        super(SpecGANGenerator, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        
        self.avgpool = nn.AdaptiveMaxPool2d((128, 128))
        self.fc = nn.Linear(self.in_dim + self.cond_dim, 256 * 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, self.kernel_size, 2, padding=2, output_padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, self.kernel_size, 2, padding=2, output_padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, self.kernel_size, 2, padding=2, output_padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, self.kernel_size, 2, padding=2, output_padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.in_ch, self.kernel_size, 2, padding=2, output_padding=1, bias=True),
            nn.Tanh()
        )
        self.apply(self.init_weights)
        
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
        
    def forward(self, x, cond):
        output = torch.cat([x, cond], dim=1)
        output = self.fc(output).view(-1, 16 * 64, 4, 4)
        output = self.deconv(output)
        output = self.avgpool(output)
        output = output.squeeze()
        return output

        
class SpecGANDiscriminator(nn.Module):
    
    def __init__(self, in_dim, cond_dim, in_ch=1, kernel_size=5, phaseshuffle_rad=0):
        super(SpecGANDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.phaseshuffle_rad = phaseshuffle_rad
        
        self.label_fc = nn.Linear(self.cond_dim, self.in_dim)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, 2, bias=True),
            nn.Conv2d(64, 128, 5, 2, bias=True),
            nn.Conv2d(128, 256, 5, 2, bias=True),
            nn.Conv2d(256, 512, 5, 2, bias=True),
            nn.Conv2d(512, 1024, 5, 2, bias=True)
        )
        self.fc = nn.Linear(1024, 1)
        self.apply(self.init_weights)
        

    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
    
    def forward(self, x, cond):
        cond = self.label_fc(cond.type(torch.float32))
        output = torch.cat((x, cond.unsqueeze(-1)), -1).unsqueeze(1)
        for layer in self.conv:
            output = layer(output)
            output = self.lrelu(output)
            output = self._apply_phase_shuffle(output, self.phaseshuffle_rad)
        output.squeeze_()
        output = self.fc(output)
        return output
            
            
    def _apply_phase_shuffle(self, x, rad, pad_type='reflect'):
        if self.phaseshuffle_rad > 0:
            b, channels, x_len = x.shape
            phase = torch.randint(-rad, rad + 1, (1,))
            pad_l = max(phase, 0)
            pad_r = max(-phase, 0)
            x = nn.functional.pad(x, (pad_l, pad_r), mode=pad_type)
            x = x[..., pad_l:pad_l + x_len]
            x.view(b, channels, x_len)
        return x
