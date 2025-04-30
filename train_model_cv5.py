import numpy as np
import pandas as pd
import copy
import os
import time
import torch
import matplotlib.pyplot as plt
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram
import lib.model_VGG2D
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm  # progressbar
import torchmetrics
import pickle as pkl


class drone_data_dataset(Dataset):
    """
    Dataset class for drone IQ Signals + transform to spectrogram
    """
    def __init__(self, path, transform=None, device=None):
        self.path = path
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')] # filter for files with .pt extension  
        self.files = [f for f in self.files if f.startswith('IQdata_sample')] # filter for files which start with IQdata_sample in name
        self.transform = transform
        self.device = device

        # create list of tragets and snrs for all samples
        self.targets = []
        self.snrs = []
        
        for file in self.files:
            self.targets.append(int(file.split('_')[2][6:])) # get target from file name
            self.snrs.append(int(file.split('_')[3].split('.')[0][3:])) # get snr from file name

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
    
    def get_files(self):
        return self.files


class transform_spectrogram(torch.nn.Module):
    def __init__(
        self,
        device,
        n_fft=1024,
        win_length=1024,
        hop_length=1024,
        window_fn=torch.hann_window,
        power=None, # Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc. If None, then the complex spectrum is returned instead. (Default: 2)
        normalized=False,
        center=False,
        #pad_mode='reflect',
        onesided=False
    ):
        super().__init__()
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn, power=power, normalized=normalized, center=center, onesided=onesided).to(device=device)   
        self.win_lengt = win_length

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        # Convert to spectrogram
        iq_signal = iq_signal[0,:] + (1j * iq_signal[1,:]) # convert to complex signal
        spec = self.spec(iq_signal)
        spec = torch.view_as_real(spec) # Returns a view of a complex input as a real tensor. last dimension of size 2 represents the real and imaginary components of complex numbers
        spec = torch.moveaxis(spec,2,0) # move channel dimension to first dimension (1024, 1024, 2) -> (2, 1024, 1024)
        spec = spec/self.win_lengt # normalise by fft window size
        return spec


def plot_two_channel_spectrogram(spectrogram_2d, title='', figsize=(10,6)):
    figure, axis = plt.subplots(1, 2, figsize=figsize)
    re = axis[0].imshow(spectrogram_2d[0,:,:]) #, aspect='auto', origin='lower')
    axis[0].set_title("Re")
    figure.colorbar(re, ax=axis[0], location='right', shrink=0.5)

    im = axis[1].imshow(spectrogram_2d[1,:,:]) #, aspect='auto', origin='lower')
    axis[1].set_title("Im")
    figure.colorbar(im, ax=axis[1], location='right', shrink=0.5)

    figure.suptitle(title)
    plt.show()


def plot_two_channel_iq(iq_2d, title='', figsize=(10,6)):
    figure, axis = plt.subplots(2, 1, figsize=figsize)
    axis[0].plot(iq_2d[0,:]) 
    axis[0].set_title("Re")
    axis[1].plot(iq_2d[1,:])
    axis[1].set_title("Im")
    figure.suptitle(title)
    plt.show()


def get_model_spec(model_name, num_classes):
    if(model_name == 'vgg11'):
        return lib.model_VGG2D.vgg11(num_classes=num_classes)
    elif(model_name == 'vgg11_bn'):
        return lib.model_VGG2D.vgg11_bn(num_classes=num_classes)
    elif(model_name == 'vgg13'):
        return lib.model_VGG2D.vgg13(num_classes=num_classes)
    elif(model_name == 'vgg13_bn'):
        return lib.model_VGG2D.vgg13_bn(num_classes=num_classes)
    elif(model_name == 'vgg16'):
        return lib.model_VGG2D.vgg16(num_classes=num_classes)
    elif(model_name == 'vgg16_bn'):
        return lib.model_VGG2D.vgg16_bn(num_classes=num_classes)
    elif(model_name == 'vgg19'):
        return lib.model_VGG2D.vgg19(num_classes=num_classes)
    elif(model_name == 'vgg19_bn'):
        return lib.model_VGG2D.vgg19_bn(num_classes=num_classes)
    else:
        print('Error: no valid model name:', model_name)
        exit()


def train_model_observe_snr_performance_spec(model, criterion, optimizer, scheduler, num_classes, num_epochs, snr_list_for_observation):
    
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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # phase = 'train'
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                epoch_train_loop = tqdm(dataloaders[phase])  # setup progress bar

                # iterate over data of the epoch (training)
                # inputs, labels, snrs = next(iter(epoch_train_loop))
                # for batch_id, (inputs_iq, inputs_spec, labels, snrs, duty_cycles) in enumerate(epoch_train_loop):
                # iq_data, target, act_snr, sample_id, transformed_data = next(iter(epoch_train_loop))
                for batch_id, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(epoch_train_loop):
                    inputs = transformed_data.to(device)
                    labels = target.to(device)
                    
                    # add model graph to tensorboard
                    if (batch_id==0) & (epoch==0):
                        writer.add_graph(model, inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
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

                    # print(f"Accuracy on batch: {batch_train_acc}")
                    if train_verbose:
                        # show progress bar for the epoch
                        epoch_train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                        # epoch_train_loop.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/batch_size)
                        epoch_train_loop.set_postfix()

                # apply learning rate scheduler after training epoch (for exp_lr_scheduler)
                # scheduler.step()

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
                    scheduler.step(epoch_loss)

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

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr


def eval_model_spec(model, num_classes, data_loader):
    # init tensor to model outputs and targets
    eval_targets = torch.empty(0, device=device)
    eval_predictions = torch.empty(0, device=device)
    
    eval_snrs = torch.empty(0, device=device)
    eval_duty_cycle = torch.empty(0, device=device)

    # initialize metric
    eval_metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes,).to(device) # accuracy

    # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    eval_metric_weighted_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device) # weigthed accuracy

    # evaluate the model
    model.eval()  # Set model to evaluate mode

    # iterate over data of the epoch (evaluation)
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
num_folds = 1 # number of folds for cross validation
num_epochs = 5 # number of epochs to train
batch_size = 8 # batch size
learning_rate = 0.005 # start learning rate
train_verbose = True  # show epoch
model_name = 'vgg11_bn'

# set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
# elif torch.cuda.is_available(): # Optional: Keep if you might run on CUDA machines too
#     device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Using device: {device}") # Good practice to confirm which device is used

experiment_name = model_name + \
                '_CV' + str(num_folds) +\
                '_epochs' + str(num_epochs) + \
                '_lr' + str(learning_rate) + \
                '_batchsize' + str(batch_size)


print('Starting experiment:', experiment_name)

# create path to store results
act_result_path = result_path + experiment_name + '/'
try:
    os.mkdir(act_result_path)
except OSError as error:
    print(error)
try:
    os.mkdir(act_result_path + 'plots/')
except OSError as error:
    print(error)

# read statistics/class count of the dataset
dataset_stats = pd.read_csv(data_path + 'class_stats.csv', index_col=0)
class_names = dataset_stats['class'].values

# read SNR count of the dataset
snr_stats = pd.read_csv(data_path + 'SNR_stats.csv', index_col=0)
snr_list = snr_stats['SNR'].values

# setup transform: IQ -> SPEC
data_transform = transform_spectrogram(device=device) # create transform object
# create dataset object
drone_dataset = drone_data_dataset(path=data_path, device=device, transform=data_transform)

# split data with stratified kfold
dataset_indices = list(range(len(drone_dataset)))

# targets = drone_dataset.get_targets()
# snr_list = drone_dataset.get_snrs()
# files = drone_dataset.get_files()

# fold=0
for fold in range(num_folds):
    print('Fold:', fold)
    # Tensorboard writer will output to ./runs/ directory by default
    writer = SummaryWriter(act_result_path + 'runs/fold' + str(fold))

    if num_folds == 1:
        print("Using single 80/10/10 train/val/test split.")
        # Split into 90% train+val and 10% test
        train_val_idx, test_idx = train_test_split(dataset_indices, test_size=0.1, stratify=drone_dataset.get_targets(), random_state=42) # Added random_state for reproducibility
        y_test = [drone_dataset.get_targets()[x] for x in test_idx]
        y_train_val = [drone_dataset.get_targets()[x] for x in train_val_idx]

        # Split 90% train+val into 80% train and 10% val (relative to original dataset size)
        val_split_size = 0.1 / 0.9 # Calculate split fraction for validation set relative to train_val set
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_split_size, stratify=y_train_val, random_state=42) # Added random_state
        y_val = [drone_dataset.get_targets()[x] for x in val_idx]
        y_train = [drone_dataset.get_targets()[x] for x in train_idx]

    else: # Original k-fold logic
        # split data with stratified kfold with respect to target class
        train_idx, test_idx = train_test_split(dataset_indices, test_size=1/num_folds, stratify=drone_dataset.get_targets())
        y_test = [drone_dataset.get_targets()[x] for x in test_idx]
        y_train = [drone_dataset.get_targets()[x] for x in train_idx]

        # split val data from train data in stratified k-fold manner
        train_idx, val_idx = train_test_split(train_idx, test_size=1/num_folds, stratify=y_train)
        y_val = [drone_dataset.get_targets()[x] for x in val_idx]
        y_train = [drone_dataset.get_targets()[x] for x in train_idx]

    # get train samples weight by class weight for each train target
    class_weights = 1. / dataset_stats['count']

    train_samples_weight = np.array([class_weights[int(i)] for i in y_train])
    train_samples_weight = torch.from_numpy(train_samples_weight)

    train_dataset = torch.utils.data.Subset(drone_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(drone_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(drone_dataset, test_idx)

    # define weighted random sampler with the weighted train samples
    train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=False)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    num_classes = len(np.unique(y_val))

    model = get_model_spec(model_name, num_classes)
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    criterion = nn.CrossEntropyLoss()  # don't use class weights in the loss

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, threshold=0.0001,
                                                        # threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
                                                                                                    
    # train model
    model, train_loss, train_acc, val_loss, val_acc, train_weighted_acc, val_weighted_acc, best_epoch, lr = train_model_observe_snr_performance_spec(model=model,
                                                                                                                        criterion=criterion,
                                                                                                                        optimizer=optimizer_ft,
                                                                                                                        scheduler=None,
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
    torch.save(model, act_result_path + 'model_fold' + str(fold) + '.pth')

    # eval model on test data
    # # load best model
    # model = torch.load(act_result_path + 'model.pth')
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