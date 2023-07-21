import torch
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from tqdm import tqdm

from WaveGAN import WaveGANGenerator, WaveGANDiscriminator


def train(train_set, batch_size, lr,  epochs, z_size, d_updates=5, g_updates=1, verbose=1, val_set=None, ph=0):

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
    G_optim = optim.Adam(G.parameters(), lr, betas=(0.5, 0.9))
    D_optim = optim.Adam(D.parameters(), lr, betas=(0.5, 0.9))

    # Creating loss function
    criterion = nn.BCELoss()
    
    G_losses = []
    D_losses = []
    
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
                real = D(x, cond)

                # Fake batch forward pass
                z = torch.rand(batch_size, z_size, device=device)
                target_f = torch.full((batch_size,), 0, dtype=torch.float, device=device)
                G_out = G(z, cond)
                fake = D(G_out, cond)
                
                # Backward pass
                pred = torch.cat((real, fake))
                target = torch.cat((target_r, target_f))
                loss_D = criterion(pred, target)
                loss_D.backward()
                
                D_optim.step()
                
                # TP and TN of the last iteration to compute train set accuracy
                if i == d_updates - 1:
                    TP += torch.round(real).sum()
                    TN += torch.round(fake).logical_not().sum()
                    
            ####################################################################
            # Training the generator: maximising log(D(G(x))
            ####################################################################
            for _ in range(g_updates):
                G_optim.zero_grad()
                z = torch.rand(batch_size, z_size, device=device)
                y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                
                # Repeat forward pass?
                output = D(G(z, cond), cond)
                loss_G = criterion(output, y)
                loss_G.backward()
                G_optim.step()

            loss_G_batch.append(loss_G)
            loss_D_batch.append(loss_D)
        
        
        train_acc = (TP + TN) / (2 * trainloader.__len__() * batch_size)
        
#        if val_set is not None:
#            D.eval()
#            G.eval()
#            TP = 0
#            TN = 0
#            for (x, cond) in tqdm(valloader):
#                x, cond = x.to(device), cond.to(device)
#                val_real = D(x, cond)
#                TP += torch.round(val_real).sum()
#                
#                val_fake = G(fixed_z, cond)
#                val_fake = D(val_fake, cond)
#                TN += torch.round(val_fake).logical_not().sum()
#            
#            val_acc = (TP + TN) / (2 * val_set.__len__())
#        else:
#            val_acc = None
            

        # Output
        loss_G_epoch = torch.mean(torch.tensor(loss_G_batch))
        loss_D_epoch = torch.mean(torch.tensor(loss_D_batch))

        if epoch % verbose == 0:
            if val_set:
                            print(f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {loss_G_epoch.item()}, Discriminator loss: {loss_D_epoch.item()}, Train accuracy: {train_acc}, Validation accuracy: {val_acc}')
            else:
                print(f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {loss_G_epoch.item()}, Discriminator loss: {loss_D_epoch.item()}, Train accuracy: {train_acc}')

        G_losses.append(loss_G_epoch)
        D_losses.append(loss_D_epoch)
    
    return G_losses, D_losses, G, D

