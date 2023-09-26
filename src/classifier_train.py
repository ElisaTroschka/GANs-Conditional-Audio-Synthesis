import torch
from torch.utils.data import DataLoader
from torch import nn, cuda
from tqdm import tqdm

from src.classifier import AudioClassifier


def train_classifier(train_set, epochs=1000, batch_size=50, lr=1e-4, save_epochs=25, pretr_epochs=0, save_dir='',
                     valid_set=None):
    """
    Training loop for AudioClassifier, used to compute IS
    :param train_set: train set
    :param epochs: epochs of training
    :param batch_size: batch size
    :param lr: learning rate
    :param save_epochs: number of epochs between checkpoints
    :param pretr_epochs: number of epochs of pretraining
    :param save_dir: checkpoints saving directory
    :param valid_set: validation set. Default: None
    """
    # Set up device
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f"Working on {device}")

    # Constructing AudioClassifier
    n_classes = train_set.label_size
    classifier = AudioClassifier(out_dim=n_classes).to(device)

    # Loading pretrained model
    if pretr_epochs != 0:
        print('Loading state dict...')
        classifier.load_state_dict(torch.load(f'{save_dir}class_{lr}-{pretr_epochs}.pt'))
    print(classifier)

    # Loading validation set
    if valid_set:
        validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Constructing optimizer and loss function
    optim = torch.optim.Adam(classifier.parameters(), lr, betas=(0.5, 0.9))
    criterion = nn.BCELoss()

    loss_hist = []
    # Training loop
    for epoch in range(epochs):
        loss_epoch = 0
        correct_preds = 0
        for i, (x, y_true, _) in enumerate(tqdm(trainloader, leave=True)):
            # forward pass
            x, y_true = x.to(device), y_true.to(device)
            optim.zero_grad()
            y_pred = classifier(x)
            loss = criterion(y_pred, y_true.float())
            # backward pass
            loss.backward()
            optim.step()

            # computing training metrics
            loss_epoch += loss
            max_idx = torch.argmax(y_pred, dim=1)
            preds = torch.zeros_like(y_pred)
            preds[torch.arange(y_pred.shape[0]), max_idx] = 1
            correct_preds += (preds * y_true).sum()

        # updating loss history
        loss_epoch /= len(trainloader) * batch_size
        loss_hist.append(loss_epoch)
        # computing epoch train accuracy
        train_acc = correct_preds / (len(trainloader) * batch_size)

        # outputs
        if valid_set:
            # computing validation accuracy
            val_acc = accuracy_score(validloader, classifier, device)
            print(
                f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Avg loss: {loss_epoch:.3f}, Train accuracy: {train_acc:.3f}, Validation accuracy: {val_acc:.3f}')
        else:
            print(
                f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Avg loss: {loss_epoch:.3f}, Train accuracy: {train_acc:.3f}')

        # saving checkpoint
        if epoch % save_epochs == 0 or epoch == epochs - 1:
            torch.save(classifier.state_dict(), f'{save_dir}class_{lr}-{epoch + pretr_epochs}.pt')
            loss_history_path = f'{save_dir}loss_history-{pretr_epochs}.txt'
            with open(loss_history_path, 'w') as f:
                for e, loss in enumerate(loss_hist):
                    f.write(f'Epoch {e + 1}: {loss:.5f}\n')

    return classifier, loss_hist


def accuracy_score(loader, C, device):
    """
    Computes accuracy of classifier C
    :param loader: data loader
    :param C: classifier
    :param device: cpu or cuda
    :return: accuracy score
    """
    correct_preds = 0
    for (x, y_true, _) in loader:
        x, y_true = x.to(device), y_true.to(device)
        y_pred = C(x)
        max_idx = torch.argmax(y_pred, dim=1)
        preds = torch.zeros_like(y_pred).to(device)
        preds[torch.arange(y_pred.shape[0]).to(device), max_idx] = 1
        correct_preds += (preds * y_true).sum()
    accuracy = correct_preds / (len(loader) * loader.batch_size)
    return accuracy
