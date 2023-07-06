import torch
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from tqdm import tqdm

from WaveGAN import WaveGANGenerator, WaveGANDiscriminator


def train(train_set, batch_size, lr,  epochs, z_size, verbose=50):

    G_losses = []
    D_losses = []

    # Setting the device
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f"Working on {device}")

    # Loading the data
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # Creating a fixed noise vectors
    fixed_z = torch.rand(batch_size, z_size, device=device)


    # Creating the generator and discriminator
    G = WaveGANGenerator(z_size, train_set.label_size, train_set.y_size, train_set.sample_rate, train_set.duration).to(device)
    D = WaveGANDiscriminator(train_set.y_size, train_set.label_size).to(device)

    # Creating optimizers
    G_optim = optim.Adam(G.parameters(), lr, betas=(0.5, 0.9))
    D_optim = optim.Adam(D.parameters(), lr, betas=(0.5, 0.9))

    # Creating loss function
    criterion = nn.BCEWithLogitsLoss()

    # Train loop
    for epoch in range(epochs):
        loss_G_batch = []
        loss_D_batch = []
        for i, data in enumerate(tqdm(trainloader, leave=True)):

            x, cond = data
            ####################################################################
            # Training the discriminator: maximising log(D(x)) + log(1 - D(G(x))
            ####################################################################
            D.zero_grad()
            # Training with real batch
            y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            real = D(x, cond)
            loss_D_real = criterion(real, y)
            loss_D_real.backward()

            # Training with fake batch
            z = torch.rand(batch_size, z_size, device=device)
            y = torch.full((batch_size,), 0, dtype=torch.float, device=device)
            fake = G(z, cond)
            loss_D_fake = criterion(fake.detach(), y)
            loss_D_fake.backward()
            loss_D = loss_D_real + loss_D_fake
            D_optim.step()

            ####################################################################
            # Training the generator: maximising log(D(G(x))
            ####################################################################
            G.zero_grad()
            y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = D(fake, cond)
            loss_G = criterion(output, y)
            loss_G.backward()
            G_optim.step()

            loss_G_batch.append(loss_G)
            loss_D_batch.append(loss_D)

        # Output
        loss_G_epoch = loss_G_batch.mean()
        loss_D_epoch = loss_D_batch.mean()

        if epoch % verbose == 0:
            print(f'\n\033[94m;1mEPOCH {epoch}\033[0m Generator loss: {loss_G_epoch.item()}, Discriminator loss: {loss_D_epoch.item()}')

        G_losses.append(loss_G_epoch)
        D_losses.append(loss_D_epoch)

