import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import Spectrogram
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on RF spectrogram dataset')
    parser.add_argument('--data_dir',   default='./data',    help='Path to .pt files and class_stats.csv')
    parser.add_argument('--output_dir', default='./new-results', help='Directory to save outputs')
    parser.add_argument('--epochs',     type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr',         type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--workers',    type=int, default=0, help='DataLoader workers')
    parser.add_argument('--device',     default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--k_folds',    type=int, default=5, help='Number of folds for k-fold cross-validation')
    return parser.parse_args()


class DroneDataset(Dataset):
    def __init__(self, data_dir, transform, device):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.transform = transform
        self.device = device
        self.targets = [int(f.split('_')[2][6:]) for f in self.files]
        self.snrs    = [int(f.split('_')[3].split('.')[0][3:]) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        sample = torch.load(os.path.join(self.data_dir, fname))
        iq = sample['x_iq'].to(self.device)
        spec = self.transform(iq)
        label = sample['y']
        snr = sample['snr']
        return spec, label, snr


class SpectrogramTransform(nn.Module):
    def __init__(self, device, n_fft=1024, win_length=1024, hop_length=1024):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            window_fn=torch.hann_window, power=None,
            normalized=False, center=False, onesided=False
        ).to(device)
        self.win_length = win_length

    def forward(self, iq):
        z = iq[0] + 1j * iq[1]
        S = self.spec(z)
        S = torch.view_as_real(S).permute(2, 0, 1)
        return S / self.win_length


def get_model(num_classes, device):
    model = models.resnet18(weights=None)
    c = model.conv1
    model.conv1 = nn.Conv2d(2, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False).to(device)
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
    return model


def train_model(model, loaders, sizes, criterion, optimizer, scheduler, device, epochs, writer):
    best_wts, best_bal, best_epoch = model.state_dict(), 0.0, 0
    history = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc','train_bal','val_bal']}
    acc_m = torchmetrics.Accuracy(task='multiclass', num_classes=model.fc.out_features).to(device)
    bal_m = torchmetrics.Accuracy(task='multiclass', num_classes=model.fc.out_features, average='macro').to(device)

    for ep in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            running_loss = 0.0
            acc_m.reset(); bal_m.reset()
            # batch-level progress bar
            batch_iter = tqdm(
                loaders[phase],
                desc=f'{phase.upper()} batches',
                leave=False,
                unit='batch'
            )
            for X, Y, _ in batch_iter:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out = model(X)
                    loss = criterion(out, Y)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                preds = out.argmax(1)
                running_loss += loss.item() * X.size(0)
                acc_m.update(preds, Y); bal_m.update(preds, Y)
                # update batch bar
                batch_iter.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{acc_m.compute().item():.3f}',
                    'bal': f'{bal_m.compute().item():.3f}'
                })
            # epoch metrics
            loss_ep = running_loss / sizes[phase]
            acc_ep = acc_m.compute().item()
            bal_ep = bal_m.compute().item()
            history[f'{phase}_loss'].append(loss_ep)
            history[f'{phase}_acc'].append(acc_ep)
            history[f'{phase}_bal'].append(bal_ep)
            writer.add_scalar(f'Loss/{phase}', loss_ep, ep)
            writer.add_scalar(f'Acc/{phase}', acc_ep, ep)
            writer.add_scalar(f'BalAcc/{phase}', bal_ep, ep)
            tqdm.write(f'{phase:5} | Loss {loss_ep:.4f} Acc {acc_ep:.4f} Bal {bal_ep:.4f}')
            if phase=='val' and bal_ep > best_bal:
                best_bal, best_wts, best_epoch = bal_ep, model.state_dict(), ep
        scheduler.step(history['val_loss'][-1])

    print(f'Best val balanced acc {best_bal:.4f} at epoch {best_epoch}')
    model.load_state_dict(best_wts)
    return history, best_epoch


def evaluate_model(model, loader, device):
    acc_m = torchmetrics.Accuracy(task='multiclass', num_classes=model.fc.out_features).to(device)
    bal_m = torchmetrics.Accuracy(task='multiclass', num_classes=model.fc.out_features, average='macro').to(device)
    preds, labs, snrs = [], [], []
    model.eval()
    with torch.no_grad():
        for X, Y, S in loader:
            out = model(X.to(device))
            p = out.argmax(1)
            preds.append(p.cpu()); labs.append(Y); snrs.append(S)
            acc_m.update(p, Y.to(device)); bal_m.update(p, Y.to(device))
    preds = torch.cat(preds).numpy()
    labs  = torch.cat(labs).numpy()
    snrs  = torch.cat(snrs).numpy()
    return acc_m.compute().item(), bal_m.compute().item(), preds, labs, snrs


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare data and stats
    transform = SpectrogramTransform(device)
    dataset = DroneDataset(args.data_dir, transform, device)
    stats   = pd.read_csv(os.path.join(args.data_dir, 'class_stats.csv'))
    num_cls = len(stats)

    # cross-validation splits
    indices = list(range(len(dataset)))
    targets = dataset.targets
    for fold in tqdm(range(args.k_folds), desc="Folds", unit="fold"):
        tqdm.write(f"Fold {fold+1}/{args.k_folds}")
        # set up fold-specific output
        fold_dir = os.path.join(args.output_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(fold_dir, 'runs'))

        # seed and split fractions
        seed = fold if args.k_folds > 1 else 42
        if args.k_folds == 1:
            # single split: 80% train_val, 10% test, then 10% val of train_val
            test_size = 0.1
            val_rel = test_size / (1 - test_size)  # 0.1111...
        else:
            # k-fold: split 1/k for test, then 1/k of train_val for val
            test_size = 1.0 / args.k_folds
            val_rel = 1.0 / args.k_folds
        # split off test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=targets, random_state=seed
        )
        # split train vs val from remaining
        y_train_val = [targets[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_rel, stratify=y_train_val, random_state=seed
        )
        sizes = {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}

        # weighted sampler for training
        y_train = [targets[i] for i in train_idx]
        class_counts = pd.Series(y_train).value_counts().sort_index()
        sample_weights = [1.0 / class_counts[targets[i]] for i in train_idx]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # data loaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        # model, loss, optimizer, scheduler
        model = get_model(num_cls, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        # train & validate
        history, best_epoch = train_model(
            model, loaders, sizes, criterion, optimizer, scheduler,
            device, args.epochs, writer
        )

        # plot curves
        plt.figure(); plt.plot(history['train_loss'], history['val_loss']); plt.legend(['train','val']); plt.title('Loss'); plt.savefig(os.path.join(fold_dir,'loss.png'))
        plt.figure(); plt.plot(history['train_acc'], history['val_acc']); plt.legend(['train','val']); plt.title('Accuracy'); plt.savefig(os.path.join(fold_dir,'accuracy.png'))
        plt.figure(); plt.plot(history['train_bal'], history['val_bal']); plt.legend(['train','val']); plt.title('Balanced Accuracy'); plt.savefig(os.path.join(fold_dir,'balanced_accuracy.png'))

        # save model
        torch.save(model.state_dict(), os.path.join(fold_dir,'model.pth'))

        # test evaluation
        test_acc, test_bal, preds, labs, snrs = evaluate_model(model, test_loader, device)
        print(f'Fold {fold+1} test Acc: {test_acc:.4f}, test BalAcc: {test_bal:.4f}')

        # save results
        with open(os.path.join(fold_dir,'results.pkl'),'wb') as fp:
            pickle.dump({
                'history': history,
                'best_epoch': best_epoch,
                'test_acc': test_acc,
                'test_bal': test_bal,
                'preds': preds,
                'labels': labs,
                'snrs': snrs
            }, fp)

        # --- SNR vs Balanced Accuracy plot ---
        unique_snrs = sorted(np.unique(snrs))
        snr_bal_acc = []
        for s in unique_snrs:
            idxs = np.where(snrs == s)[0]
            if len(idxs) > 0:
                snr_bal_acc.append(balanced_accuracy_score(labs[idxs], preds[idxs]))
            else:
                snr_bal_acc.append(np.nan)
        plt.figure()
        plt.bar(unique_snrs, snr_bal_acc)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Balanced Accuracy')
        plt.title(f'Fold {fold+1} SNR vs Balanced Accuracy')
        plt.savefig(os.path.join(fold_dir,'snr_vs_balanced_accuracy.png'))
        plt.close()

if __name__ == '__main__':
    main() 