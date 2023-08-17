import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.feature.inverse import mel_to_audio
from IPython.display import Audio
from IPython.core.display import display


from src.WaveGAN import WaveGANGenerator, WaveGANDiscriminator
from src.SpecGAN import SpecGANGenerator, SpecGANDiscriminator
from src.utils import flip_random_elements, display_audio_sample, display_mel_sample


def compute_gradient_penalty(D, real_samples, fake_samples, cond, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates, cond)
    fake = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(train_set, 
          batch_size, 
          D_lr, G_lr,  
          epochs, 
          z_size,
          d_updates=5, 
          g_updates=1, 
          flip_prob=0, 
          verbose=1, 
          val_set=None, 
          ph=0, 
          loss='minimax', 
          save_epochs=5,
          save_dir='users/adcy353/GANs-Conditional-Audio-Synthesis/models/',
          pretr_epochs=0,
          lambda_gp=10
         ):
    
    # Inferring model from data type
    model = 'SpecGAN' if train_set.mel else 'WaveGAN'
    
    # Saving training params
    if pretr_epochs == 0:
        print('Saving train params...')
        with open(f'{save_dir}train_params.txt', 'w') as f:
            for arg in ('batch_size', 'D_lr', 'G_lr', 'epochs', 'z_size', 'd_updates', 'g_updates', 'flip_prob', 'ph', 'loss', 'lambda_gp'):
                f.write(f'{arg}: {locals()[arg]}\n')
        f.close()
                
                

    # Setting the device
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f"Working on {device}")

    # Loading the data
    print('Loading data...')
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    if val_set is not None:
        valloader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        
    # Creating a fixed noise vectors for validation
    fixed_z = torch.rand(batch_size, z_size, device=device)
    
    # Creating the generator and discriminator
    print(f'Creating {model}...')
    G, D = None, None
    if model == 'WaveGAN':
        G = WaveGANGenerator(z_size, train_set.label_size, train_set.y_size, train_set.sampling_rate, train_set.duration).to(device)
        D = WaveGANDiscriminator(train_set.y_size, train_set.label_size, phaseshuffle_rad=ph).to(device)
    elif model == 'SpecGAN':
        G = SpecGANGenerator(z_size, train_set.label_size, train_set.y_size).to(device)
        D = SpecGANDiscriminator(train_set.y_size, train_set.label_size, phaseshuffle_rad=ph).to(device)
    else:
        raise NotImplementedError('Model must be one of "WaveGAN" or "SpecGAN"')
    print(G)
    print(D)
    
    if pretr_epochs != 0:
        print('Loading state dict...')
        G.load_state_dict(torch.load(f'{save_dir}G_{G_lr}-{g_updates}-{pretr_epochs - 1}.pt'))
        D.load_state_dict(torch.load(f'{save_dir}D_{D_lr}-{d_updates}-{pretr_epochs - 1}.pt'))

    # Creating optimizers
    G_optim = optim.Adam(G.parameters(), G_lr, betas=(0.5, 0.9))
    D_optim = optim.Adam(D.parameters(), D_lr, betas=(0.5, 0.9))
    
    # creating lr scheduler
    #G_scheduler = optim.lr_scheduler.StepLR(G_optim, step_size=50, gamma=0.1)
    #D_scheduler = optim.lr_scheduler.StepLR(D_optim, step_size=50, gamma=0.1)

    # Creating loss function
    if loss == 'minimax':
        criterion = nn.BCEWithLogitsLoss()
    elif loss == 'wasserstein':
        pass
    else:
        raise NotImplementedError("loss must be one of 'minimax', 'wasserstein'")
    
    G_losses = []
    D_losses = []
    #torch.cuda.empty_cache()
    
    # Train loop
    for epoch in range(epochs):
        if epoch % verbose == 0:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print(f'\033[94mtotal_memory: {t}, available_memory: {f}\033[0m')
        
                    
        loss_G_batch = []
        loss_D_batch = []
        TP = 0
        TN = 0
        G.train()
        D.train()
        for i, data in enumerate(tqdm(trainloader, leave=True)):

            x, cond, z_pitched = data
            x, cond, z_pitched = x.to(device), cond.to(device), z_pitched.to(device)
            ####################################################################
            # Training the discriminator: maximising log(D(x)) + log(1 - D(G(x))
            ####################################################################
            for i in range(d_updates):
                D_optim.zero_grad()
                
                # Real batch forward pass
                target_r = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                target_r = flip_random_elements(target_r, flip_prob, device)
                real = D(x, cond)

                # Fake batch forward pass
                z_noise = 2 * torch.rand(batch_size, z_size, device=device) - 1
                z = z_pitched + z_noise
                target_f = torch.full((batch_size,), 0, dtype=torch.float, device=device)
                G_out = G(z, cond)
                fake = D(G_out, cond)
                
                # Backward pass
                if loss == 'minimax':
                    pred = torch.cat((real, fake))
                    target = torch.cat((target_r, target_f))
                    loss_D = criterion(pred.squeeze(), target)
                    
                elif loss == 'wasserstein':
                    gradient_penalty = compute_gradient_penalty(D, x, G_out.detach(), cond, device)
                    loss_D = torch.mean(fake) - torch.mean(real) + lambda_gp * gradient_penalty
                    
                loss_D.backward()
                D_optim.step()
                
                #for p in D.parameters():
                #    p.data.clamp_(-0.01, 0.01)
                
                # TP and TN of the last iteration to compute train set accuracy
                if i == d_updates - 1:
                    TP += torch.round(F.sigmoid(real)).sum()
                    TN += torch.round(F.sigmoid(fake)).logical_not().sum()
                    
            ####################################################################
            # Training the generator: maximising log(D(G(x))
            ####################################################################
            D.eval()
            for _ in range(g_updates):
                G_optim.zero_grad()
                z = z_pitched + (2 * torch.rand(batch_size, z_size, device=device) - 1)
                y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                output = D(G(z, cond), cond)
                
                # backward
                if loss == 'minimax':
                    loss_G = criterion(output.squeeze(), y)   
                elif loss == 'wasserstein':
                    loss_G = -torch.mean(output)
                loss_G.backward()
                G_optim.step()

            loss_G_batch.append(loss_G.item())
            loss_D_batch.append(loss_D.item())
        
        
        train_acc = (TP + TN) / (2 * trainloader.__len__() * batch_size)
        
        #G_scheduler.step()
        #D_scheduler.step()
        
        # Output
        loss_G_epoch = torch.mean(torch.tensor(loss_G_batch)).item()
        loss_D_epoch = torch.mean(torch.tensor(loss_D_batch)).item()

        if epoch % verbose == 0:
            print(get_output_str(epoch, epochs, loss_G_epoch, loss_D_epoch, train_acc))
        
        if model == 'SpecGAN':
            display_mel_sample(42, train_set, G)
        display(display_audio_sample(42, train_set, G))

        G_losses.append(loss_G_epoch)
        D_losses.append(loss_D_epoch)
        
        if epoch % save_epochs == 0 or epoch == epochs-1:
            torch.save(G.state_dict(), f'{save_dir}G_{G_lr}-{g_updates}-{epoch + pretr_epochs}.pt')
            torch.save(D.state_dict(), f'{save_dir}D_{D_lr}-{d_updates}-{epoch + pretr_epochs}.pt')
            
            loss_history_path = f'{save_dir}loss_history-{pretr_epochs}.txt'
            with open(loss_history_path, 'w') as f:
                for e, (g_loss, d_loss) in enumerate(zip(G_losses, D_losses)):
                    f.write(f'Epoch {e + 1}: Generator loss = {g_loss:.5f}, Discriminator loss = {d_loss:.5f}\n')
    
    return G_losses, D_losses, G, D


def get_output_str(epoch, epochs, g_loss, d_loss, train_acc, val_acc=None):
    if val_acc:
        return f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {g_loss:.1f}, Discriminator loss: {d_loss:.1f}, Train accuracy: {train_acc:.5f}, Val accuracy: {val_acc:.5f}'
    else:
        return f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {g_loss:.1f}, Discriminator loss: {d_loss:.1f}, Train accuracy: {train_acc:.5f}'

    
