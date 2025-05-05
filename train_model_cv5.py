import numpy as np
import pandas as pd
import copy
import os
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm  # progressbar
import torchmetrics
import pickle as pkl
import torchvision.models as models


class drone_data_dataset(Dataset):
    # Dataset for drone IQ signals with spectrogram transform
    def __init__(self, path, transform=None, device=None):
        self.path = path
        # load only relevant .pt files
        self.files = [f for f in os.listdir(path) if f.endswith('pt') and f.startswith('IQdata_sample')]
        self.transform = transform
        self.device = device
        # parse targets and SNRs from filenames
        self.targets = [int(f.split('_')[2][6:]) for f in self.files]
        self.snrs = [int(f.split('_')[3].split('.')[0][3:]) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        sample_id = int(file.split('_')[1][6:]) # get sample id from file name
        data_dict = torch.load(self.path + file) # load data       
        iq_data = data_dict['x_iq']
        act_target = data_dict['y']
        act_snr = data_dict['snr']

        if self.transform:
            if self.device:
                iq_data = iq_data.to(device=device)
            transformed_data = self.transform(iq_data)
        else:
            transformed_data = None

        return iq_data, act_target, act_snr, sample_id, transformed_data
    
    def get_targets(self): # return list of targets
        return self.targets

    def get_snrs(self): # return list of snrs
        return self.snrs


class transform_spectrogram(torch.nn.Module):
    # Spectrogram transform for IQ signals
    def __init__(self, device, n_fft=1024, win_length=1024, hop_length=1024):
        super().__init__()
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                 window_fn=torch.hann_window, power=None,
                                 normalized=False, center=False, onesided=False).to(device)
        self.win_length = win_length

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        # Convert IQ to real-imag spectrogram
        iq_signal = iq_signal[0,:] + (1j * iq_signal[1,:])
        spec = self.spec(iq_signal)
        spec = torch.view_as_real(spec)
        spec = torch.moveaxis(spec,2,0)
        spec = spec / self.win_length
        return spec


def get_model_spec(model_name, num_classes):
    # Only ResNet-18 supported
    model = models.resnet18(weights=None)
    conv = model.conv1
    model.conv1 = nn.Conv2d(2, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.bias)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model_observe_snr_performance_spec(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_classes, num_epochs, snr_list_for_observation):
    
    since = time.time()
    train_loss = []
    train_acc = []
    train_weighted_acc = []
    lr = []

    val_loss = []
    val_acc = []
    val_weighted_acc = []

    # create variables to store acc for different SNR samples
    num_snrs_to_observe = len(snr_list_for_observation)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    print('start training')
    print('-' * 10)

    for epoch in range(num_epochs):
        # initialize metric
        # accuracy
        train_metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
        val_metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

        # weigthed accuracy
        # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
        train_metric_weighted_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
        val_metric_weighted_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)

        # snr dependent accuracies metrics
        snr_val_metric_acc_list =[torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device) for i in range(num_snrs_to_observe)]
        snr_val_metric_weighted_acc_list =[torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device) for i in range(num_snrs_to_observe)]
        
        # snr dependent accuracies storage for epoch
        snr_epoch_acc = torch.zeros([num_snrs_to_observe], dtype=torch.float)
        snr_epoch_weighted_acc = torch.zeros([num_snrs_to_observe], dtype=torch.float)

        # Training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                running_loss = 0.0
                epoch_train_loop = tqdm(dataloaders[phase])  # training loop

                for batch_id, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(epoch_train_loop):
                    inputs = transformed_data.to(device)
                    labels = target.to(device)
                    
                    # add model graph to tensorboard
                    if (batch_id==0) & (epoch==0):
                        writer.add_graph(model, inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                    # compute scores for the epoch
                    running_loss += loss.item() * inputs.size(0)

                    # compute scores for batch
                    train_metric_acc.update(preds, labels.data)
                    train_metric_weighted_acc.update(preds, labels.data)

                    # update progress bar
                    epoch_train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                    epoch_train_loop.set_postfix()

                # compute and show metrics for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = train_metric_acc.compute().item()
                epoch_weighted_acc = train_metric_weighted_acc.compute().item()

                print('{} Loss: {:.4f} Acc: {:.4f}  Balanced Acc: {:.4f} |'.format(phase, epoch_loss, epoch_acc, epoch_weighted_acc), end=' ')

                # store metric for epoch
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_weighted_acc.append(epoch_weighted_acc)
                lr.append(optimizer.param_groups[0]['lr'])

                # add to tensor board
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                writer.add_scalar('BalancedAccuracy/train', epoch_weighted_acc, epoch)
                writer.add_scalar('Learnigrate', optimizer.param_groups[0]['lr'], epoch)
            else:
                # phase = 'val'
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0

                # iterate over data of the epoch (evaluation)
                for batch_id, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(dataloaders[phase]):
                    inputs = transformed_data.to(device)
                    labels = target.to(device)
                    snrs = act_snr.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    # compute scores for batch
                    val_metric_acc.update(preds, labels.data)
                    val_metric_weighted_acc.update(preds, labels.data)

                    # compute accuracies for diffrent SNRs
                    for i, snr in enumerate(snr_list_for_observation):
                        act_snr_sample_indices = torch.where(snrs == snr)[0]
                        if act_snr_sample_indices.size(0) > 0: # if there are some samples with current SNR
                            snr_val_metric_acc_list[i].update(preds[act_snr_sample_indices], labels.data[act_snr_sample_indices])
                            snr_val_metric_weighted_acc_list[i].update(preds[act_snr_sample_indices], labels.data[act_snr_sample_indices])
                            
                # compute and show metrics for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = val_metric_acc.compute().item()
                epoch_weighted_acc = val_metric_weighted_acc.compute().item()

                for i in range(num_snrs_to_observe):
                    snr_epoch_acc[i] = snr_val_metric_acc_list[i].compute().item()
                    snr_epoch_weighted_acc[i] = snr_val_metric_weighted_acc_list[i].compute().item()

                # apply LR scheduler ... looking for plateau in val loss
                if scheduler:
                    # Removed unused scheduler step
                    pass

                print('{} Loss: {:.4f} Acc: {:.4f}  Balanced Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_weighted_acc))
                # store validation loss
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                val_weighted_acc.append(epoch_weighted_acc)

                # add to tensor board
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                writer.add_scalar('BalancedAccuracy/val', epoch_weighted_acc, epoch)

                # SNR measures to tensorboard
                for i, snr in enumerate(snr_list_for_observation):
                    writer.add_scalar('SNR/val Accuracy SNR' + str(snr), snr_epoch_acc[i], epoch)
                    writer.add_scalar('SNR/val BalancedAccuracy SNR' + str(snr), snr_epoch_weighted_acc[i], epoch)

                #     best_epoch = epoch
                if epoch_weighted_acc > best_acc:
                    best_acc = epoch_weighted_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best epoch: {}'.format(best_epoch))

    model.load_state_dict(best_model_wts)

    return model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr


def eval_model_spec(model, num_classes, data_loader):
    # init tensor to model outputs and targets
    eval_targets = torch.empty(0, device=device)
    eval_predictions = torch.empty(0, device=device)
    
    eval_snrs = torch.empty(0, device=device)
    eval_duty_cycle = torch.empty(0, device=device)

    eval_metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes,).to(device) # accuracy

    eval_metric_weighted_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device) # weigthed accuracy

    model.eval()  # Set model to evaluate mode

    for batch_id, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(data_loader):
        inputs = transformed_data.to(device)
        labels = target.to(device)
        snrs = act_snr.to(device)

        # forward through model
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # store batch model outputs and targets
        eval_predictions = torch.cat((eval_predictions, preds.data))
        eval_targets = torch.cat((eval_targets, labels.data))
        eval_snrs = torch.cat((eval_snrs, snrs.data))
        
        # compute batch evaluation metric
        eval_metric_acc.update(preds, labels.data)
        eval_metric_weighted_acc.update(preds, labels.data)

    # compute metrics for complete data
    eval_acc = eval_metric_acc.compute().item()
    eval_weighted_acc = eval_metric_weighted_acc.compute().item()

    return eval_acc, eval_weighted_acc, eval_predictions, eval_targets, eval_snrs, eval_duty_cycle


project_path = './'
result_path = project_path + 'results/experiments/'
data_path = './data/'

# global params
num_workers = 0 # number of workers for data loader
num_folds = 1 # number of folds for cross validation - Changed from 2
num_epochs = 10 # number of epochs to train - Changed from 20 to 10 for remaining run
batch_size = 16 # batch size - Changed from 24
learning_rate = 5e-4 # start learning rate
train_verbose = True  # show epoch
model_name = 'resnet18'

# set device
device = torch.device('cpu')  # Force CPU
print(f"Using device: {device}")

experiment_name = model_name + \
                '_CV' + str(num_folds) +\
                '_epochs' + str(num_epochs) + \
                '_lr' + str(learning_rate) + \
                '_batchsize' + str(batch_size)


print('Starting experiment:', experiment_name)

# create path to store results
act_result_path = result_path + experiment_name + '/'
os.makedirs(act_result_path, exist_ok=True)
os.makedirs(act_result_path + 'plots/', exist_ok=True)

# read statistics/class count of the dataset
dataset_stats = pd.read_csv(data_path + 'class_stats.csv', index_col=0)
class_names = dataset_stats['class'].values
num_classes = len(class_names) # Get num_classes from the definitive dataset stats
print(f"Total number of classes based on class_stats.csv: {num_classes}")

# read SNR count of the dataset
snr_stats = pd.read_csv(data_path + 'SNR_stats.csv', index_col=0)
snr_list = snr_stats['SNR'].values

# setup transform: IQ -> SPEC
data_transform = transform_spectrogram(device=device) # create transform object
# create dataset object
full_drone_dataset = drone_data_dataset(path=data_path, device=device, transform=data_transform)

# Setup full dataset indices for splitting
dataset_size = len(full_drone_dataset)
dataset_indices = list(range(dataset_size))
dataset_targets = full_drone_dataset.get_targets()


# fold=0
for fold in range(num_folds):
    print('Fold:', fold)
    # Tensorboard writer will output to ./runs/ directory by default
    writer = SummaryWriter(act_result_path + 'runs/fold' + str(fold))

    if num_folds == 1:
        print("Using single 80/10/10 train/val/test split on the full dataset.")
        # Split full dataset: 80% train_val, 10% test, then 80/10 train/val within train_val
        test_split_fraction = 0.1
        val_split_fraction_relative_to_train_val = test_split_fraction / (1 - test_split_fraction)
        train_val_idx, test_idx = train_test_split(
            dataset_indices,
            test_size=test_split_fraction,
            stratify=dataset_targets,
            random_state=42
        )
        y_test = [dataset_targets[i] for i in test_idx]
        y_train_val = [dataset_targets[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_split_fraction_relative_to_train_val,
            stratify=y_train_val,
            random_state=42
        )
        y_train = [dataset_targets[i] for i in train_idx]
        y_val = [dataset_targets[i] for i in val_idx]
    else: # Original k-fold logic (won't be used if num_folds=1)
        train_idx, test_idx = train_test_split(dataset_indices, test_size=1/num_folds, stratify=full_drone_dataset.get_targets())
        y_test = [full_drone_dataset.get_targets()[x] for x in test_idx]
        y_train = [full_drone_dataset.get_targets()[x] for x in train_idx]

        # split val data from train data in stratified k-fold manner
        train_idx, val_idx = train_test_split(train_idx, test_size=1/num_folds, stratify=y_train)
        y_val = [full_drone_dataset.get_targets()[x] for x in val_idx]
        y_train = [full_drone_dataset.get_targets()[x] for x in train_idx]

    # --- MODIFIED WEIGHT CALCULATION ---
    # Recalculate class weights based ONLY on the 'y_train' targets from the subset split
    print("Calculating class weights for weighted sampler based on training subset...")
    train_class_counts = pd.Series(y_train).value_counts().sort_index()
    # Ensure all classes are present, handle potential missing classes if necessary
    all_classes = range(num_classes)
    train_class_counts = train_class_counts.reindex(all_classes, fill_value=0)
    # Avoid division by zero for classes not present in the training split (shouldn't happen with stratification, but be safe)
    class_weights = {cls: 1.0 / count if count > 0 else 0 for cls, count in train_class_counts.items()}
    print(f"Subset train class counts:\n{train_class_counts}")
    # Assign weights to each sample in the training set 'train_idx' using the new weights
    train_samples_weight = np.array([class_weights[target] for target in y_train])
    # --- END MODIFIED WEIGHT CALCULATION ---

    train_samples_weight = torch.from_numpy(train_samples_weight)

    # Create datasets using the FINAL indices and the ORIGINAL full dataset
    # This avoids issues with nested Subset objects
    train_dataset = torch.utils.data.Subset(full_drone_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_drone_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_drone_dataset, test_idx)

    # define weighted random sampler with the weighted train samples
    train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) # No sampler for validation
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) # No sampler for test

    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    if os.path.exists(act_result_path + 'model_fold' + str(fold) + '.pth'):
        print(f"Loading existing model from fold {fold} to continue training...")
        # Set weights_only=False to load model saved with older PyTorch versions
        model = torch.load(act_result_path + 'model_fold' + str(fold) + '.pth', weights_only=False)
        
        # Quick validation to verify loaded model performance
        model.eval()
        with torch.no_grad():
            sample_data = next(iter(dataloaders['val']))
            _, _, _, _, inputs = sample_data
            inputs = inputs.to(device)
            outputs = model(inputs)
            print(f"Loaded model is producing valid outputs: {outputs.shape}")
        print("Continuing training with pre-trained model...")
    else:
        # Original model creation code
        model = get_model_spec(model_name, num_classes)
        model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    criterion = nn.CrossEntropyLoss()  # don't use class weights in the loss

    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # Use ReduceLROnPlateau - reduces learning rate when validation loss plateaus
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
                                                                                                    
    # train model
    model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr = train_model_observe_snr_performance_spec(model=model,
                                                                                                                        criterion=criterion,
                                                                                                                        optimizer=optimizer_ft,
                                                                                                                        scheduler=plateau_scheduler,
                                                                                                                        dataloaders=dataloaders,
                                                                                                                        dataset_sizes=dataset_sizes,
                                                                                                                        num_classes=num_classes,
                                                                                                                        num_epochs=num_epochs,
                                                                                                                        snr_list_for_observation=[0, -10, -20])

    # show/store learning curves
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train', 'val'])
    plt.title('Loss')
    plt.savefig(act_result_path + 'plots/loss_fold' + str(fold) + '.png')
    plt.close()

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train', 'val'])
    plt.title('Acc')
    plt.savefig(act_result_path + 'plots/acc_fold' + str(fold) + '.png')
    plt.close()
    # plt.show()

    plt.plot(train_weighted_acc)
    plt.plot(val_weighted_acc)
    plt.legend(['train', 'val'])
    plt.title('Weighted Acc')
    plt.savefig(act_result_path + 'plots/weigthed_acc_fold' + str(fold) + '.png')
    plt.close()
    # plt.show()

    # store model
    torch.save(model, act_result_path + 'model_fold' + str(fold) + '.pth', _use_new_zipfile_serialization=False)

    # eval model on test data
    # # load best model
    eval_acc, eval_weighted_acc, eval_predictions, eval_targets, eval_snrs, eval_duty_cycle = eval_model_spec(model=model, num_classes=num_classes, data_loader=dataloaders['test'])

    eval_targets = eval_targets.cpu()
    eval_predictions = eval_predictions.cpu()
    eval_snrs = eval_snrs.cpu()
    target_classes = np.unique(eval_targets)
    pred_classes = np.unique(eval_predictions)
    eval_classes = np.union1d(target_classes, pred_classes)
    eval_class_names = [class_names[int(x)] for x in eval_classes]

    print('Got ' + str(len(target_classes)) + ' target classes')
    print('Got ' + str(len(pred_classes)) + ' prediction classes')
    print('Resulting in ' + str(len(eval_classes)) + ' total classes')
    print(eval_class_names)

    print('Test accuracy:', eval_acc, 'Test weighted accuracy:', eval_weighted_acc, 'Best epoch:', best_epoch)

    save_dict = {'train_weighted_acc': train_weighted_acc,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'val_weighted_acc': val_weighted_acc,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'best_epoch': best_epoch,
                    'test_acc': eval_acc,
                    'test_weighted_acc': eval_weighted_acc,
                    'test_predictions': eval_predictions,
                    'test_targets': eval_targets,
                    'test_snrs': eval_snrs,
                    'class_names': class_names,
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'test_idx': test_idx
                    }
    save_filename = 'results_fold' + str(fold) + '.pkl'

    outfile = open(act_result_path + save_filename, 'wb')
    pkl.dump(save_dict, outfile)
    outfile.close()