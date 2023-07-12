import torch
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from tqdm import tqdm

from WaveGAN import WaveGANGenerator, WaveGANDiscriminator


def train(train_set, batch_size, lr,  epochs, z_size, dg_ratio=5, verbose=50):

    G_losses = []
    D_losses = []

    # Setting the device
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f"Working on {device}")

    # Loading the data
    print('Loading data...')
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Creating a fixed noise vectors
    fixed_z = torch.rand(batch_size, z_size, device=device)


    # Creating the generator and discriminator
    print('Creating WaveGAN...')
    G = WaveGANGenerator(z_size, train_set.label_size, train_set.y_size, train_set.sampling_rate, train_set.duration).to(device)
    print(G)
    D = WaveGANDiscriminator(train_set.y_size, train_set.label_size).to(device)
    print(D)

    # Creating optimizers
    G_optim = optim.Adam(G.parameters(), lr, betas=(0.5, 0.9))
    D_optim = optim.Adam(D.parameters(), lr, betas=(0.5, 0.9))

    # Creating loss function
    criterion = nn.BCEWithLogitsLoss()

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
        for i, data in enumerate(tqdm(trainloader, leave=True)):

            x, cond = data
            x, cond = x.to(device), cond.to(device)
            ####################################################################
            # Training the discriminator: maximising log(D(x)) + log(1 - D(G(x))
            ####################################################################
            for _ in range(dg_ratio):
                D.zero_grad()
                # Training with real batch
                y_r = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                real = D(x, cond)
                loss_D_real = criterion(real, y_r)
                loss_D_real.backward()

                # Training with fake batch
                #D.zero_grad()
                z = torch.rand(batch_size, z_size, device=device)
                y_f = torch.full((batch_size,), 0, dtype=torch.float, device=device)
                G_out = G(z, cond)
                fake = D(G_out, cond)
                loss_D_fake = criterion(fake.detach(), y_f)
                loss_D_fake.requires_grad = True
                loss_D_fake.backward()
                loss_D = loss_D_real + loss_D_fake
                D_optim.step()

            ####################################################################
            # Training the generator: maximising log(D(G(x))
            ####################################################################
            G.zero_grad()
            y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = D(G_out, cond)
            loss_G = criterion(output, y)
            loss_G.backward()
            G_optim.step()

            loss_G_batch.append(loss_G)
            loss_D_batch.append(loss_D)

        # Output
        loss_G_epoch = torch.mean(torch.tensor(loss_G_batch))
        loss_D_epoch = torch.mean(torch.tensor(loss_D_batch))

        if epoch % verbose == 0:
            print(f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Generator loss: {loss_G_epoch.item()}, Discriminator loss: {loss_D_epoch.item()}')

        G_losses.append(loss_G_epoch)
        D_losses.append(loss_D_epoch)
    
    return G_losses, D_losses, G, D

