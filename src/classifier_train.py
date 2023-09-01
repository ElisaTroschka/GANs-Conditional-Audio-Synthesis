import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from tqdm import tqdm

from src.classifier import MelClassifier, AudioClassifier


def train_classifier(train_set, epochs=1000, batch_size=50, lr=1e-4, save_epochs=25, pretr_epochs=0, save_dir=''):
    
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f"Working on {device}")
    
    n_classes = train_set.label_size
    
    if train_set.mel:
        classifier = MelClassifier(out_dim=n_classes).to(device)
    else:
        classifier = AudioClassifier(out_dim=n_classes).to(device)
    
    if pretr_epochs != 0:
        print('Loading state dict...')
        classifier.load_state_dict(torch.load(f'{save_dir}class_{lr}-{pretr_epochs}.pt'))
        
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(classifier.parameters(), lr, betas=(0.5, 0.9))
    criterion = nn.BCEWithLogitsLoss()
    
    loss_hist = []
    for epoch in range(epochs):
        loss_epoch = 0
        correct_preds = 0
        for i, (x, y_true, _) in enumerate(tqdm(trainloader, leave=True)):
            x, y_true = x.to(device), y_true.to(device)
            optim.zero_grad()
            y_pred = classifier(x)
            loss = criterion(y_pred, y_true.float())
            loss.backward()
            optim.step()
            
            loss_epoch += loss / batch_size
            preds = (y_pred == y_pred.max(dim=0, keepdim=True).values).float()
            correct_preds += torch.sum((preds * y_true))
        
        loss_epoch /= len(trainloader)
        loss_hist.append(loss_epoch)
        train_acc = correct_preds / (len(trainloader) * batch_size)
        
        print(f'\n\033[1mEPOCH {epoch + 1}/{epochs}:\033[0m Avg loss: {loss_epoch:.5f}, Train accuracy: {train_acc:.5f}')
        
        if epoch % save_epochs == 0 or epoch == epochs-1:
            torch.save(classifier.state_dict(), f'{save_dir}class_{lr}-{epoch + pretr_epochs}.pt')
            
            loss_history_path = f'{save_dir}loss_history-{pretr_epochs}.txt'
            with open(loss_history_path, 'w') as f:
                for e, loss in enumerate(loss_hist):
                    f.write(f'Epoch {e + 1}: {loss:.5f}\n')
                    
    return classifier, loss_hist
    