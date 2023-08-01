import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from tqdm import tqdm

from src.WaveGAN import WaveGANGenerator, WaveGANDiscriminator
from src.utils import flip_random_elements


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
          save_epochs=5):

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
    print('Creating WaveGAN...')
    G = WaveGANGenerator(z_size, train_set.label_size, train_set.y_size, train_set.sampling_rate, train_set.duration).to(device)
    print(G)
    D = WaveGANDiscriminator(train_set.y_size, train_set.label_size, phaseshuffle_rad=ph).to(device)
    print(D)

    # Creating optimizers
    G_optim = optim.Adam(G.parameters(), G_lr, betas=(0.5, 0.9))
    D_optim = optim.Adam(D.parameters(), D_lr, betas=(0.5, 0.9))
    
    # creating lr scheduler
    #G_scheduler = optim.lr_scheduler.StepLR(G_optim, step_size=20, gamma=0.1)
    #D_scheduler = optim.lr_scheduler.StepLR(D_optim, step_size=20, gamma=0.1)

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
        
        for i, data in enumerate(tqdm(trainloader, leave=True)):

            x, cond = data
            x, cond = x.to(device), cond.to(device)
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
                z = torch.rand(batch_size, z_size, device=device)
                target_f = torch.full((batch_size,), 0, dtype=torch.float, device=device)
                G_out = G(z, cond)
                fake = D(G_out, cond)
                
                # Backward pass
                if loss == 'minimax':
                    pred = torch.cat((real, fake))
                    target = torch.cat((target_r, target_f))
                    loss_D = criterion(pred, target)
                    
                elif loss == 'wasserstein':
                    loss_D = torch.mean(fake) - torch.mean(real) 
                    
                loss_D.backward()
                D_optim.step()
                
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)
                
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
                z = torch.rand(batch_size, z_size, device=device)
                y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                output = D(G(z, cond), cond)
                
                # backward
                if loss == 'minimax':
                    loss_G = criterion(output, y)   
                elif loss == 'wasserstein':
                    loss_G = -torch.mean(output)
                loss_G.backward()
                G_optim.step()

            loss_G_batch.append(loss_G)
            loss_D_batch.append(loss_D)
        
        
        train_acc = (TP + TN) / (2 * trainloader.__len__() * batch_size)
        
        #G_scheduler.step()
        #D_scheduler.step()
        
        # Output
        loss_G_epoch = torch.mean(torch.tensor(loss_G_batch))
        loss_D_epoch = torch.mean(torch.tensor(loss_D_batch))

        if epoch % verbose == 0:
            print(get_output_str(epoch, epochs, loss_G_epoch.item(), loss_D_epoch.item(), train_acc))

        G_losses.append(loss_G_epoch)
        D_losses.append(loss_D_epoch)
        
        if epoch % save_epochs == 0 or epoch == epochs - 1:
            torch.save(G.state_dict(), f'users/adcy353/GANs-Conditional-Audio-Synthesis/models/G_{G_lr}-{g_updates}-{epoch}.pt')
            torch.save(D.state_dict(), f'users/adcy353/GANs-Conditional-Audio-Synthesis/models/D_{D_lr}-{d_updates}-{epoch}.pt')
    
    return G_losses, D_losses, G, D


def get_output_str(epoch, epochs, g_loss, d_loss, train_acc, val_acc=None):
    if val_acc:
        return f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {g_loss:.5f}, Discriminator loss: {d_loss:.5f}, Train accuracy: {train_acc:.5f}, Val accuracy: {val_acc:.5f}'
    else:
        return f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {g_loss:.5f}, Discriminator loss: {d_loss:.5f}, Train accuracy: {train_acc:.5f}'

