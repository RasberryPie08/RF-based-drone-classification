import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import os


def plot_cm_plus_class_dist(df_cm, target_class_stats_df, plt_title, plt_filename=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_gridspec(4, 4)

    # plot heatmap
    ax1_heatmap = fig.add_subplot(ax[1:, 0:3])
    ax1_heatmap.imshow(df_cm,
                    # cmap='inferno'
                    cmap='hot',
                    vmin=0.0,
                    vmax=1.0
                    )

    # put the major ticks at the middle of each cell
    ax1_heatmap.set_xticks(np.arange(df_cm.shape[1]), minor=False)
    ax1_heatmap.set_yticks(np.arange(df_cm.shape[0]), minor=False)
    # write labels to ticks
    ax1_heatmap.set_xticklabels(df_cm.columns, minor=False, rotation=45)
    ax1_heatmap.set_yticklabels(df_cm.index, minor=False)
    # add axis labels
    ax1_heatmap.set_xlabel('prediction')
    ax1_heatmap.set_ylabel('target')

    # plot colorbar
    ax1_cbar = fig.add_subplot(ax[0, 0:3])
    ax1_cbar.set_box_aspect(0.05)
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('hot')
    cbar = matplotlib.colorbar.ColorbarBase(ax1_cbar,
                                            cmap=cmap,
                                            orientation='horizontal')
    # move yaxis labels/ticks to the top
    ax1_cbar.xaxis.set_label_position('top')
    ax1_cbar.xaxis.set_ticks_position('top')

    #resize and repostion the cbar
    ll, bb, ww, hh = ax1_cbar.get_position().bounds
    ax1_cbar.set_position([ll+0.075, bb-0.075, ww * 0.74, hh*0.8])

    # add titel to the cbar
    ax1_cbar.title.set_text(plt_title)

    # plot bar chart
    ax2 = fig.add_subplot(ax[1:, 3])
    ax2.barh(y=target_class_stats_df[::-1].index,
            width=target_class_stats_df[::-1]['target_class_counts'],
            height=0.8,
            align='center'
            )

    ax2.set_xscale('log')
    ax2.set_xlabel('class count')
    plt.margins(y=0)
    if plt_filename:
        plt.savefig(plt_filename)
        plt.close()
    else:
        plt.show()


project_path = './'
result_path = project_path + 'results/experiments/'
data_path = './data/'

# global params (defines the experiment to evaluate)
num_workers = 0 
num_epochs = 50
batch_size = 8
learning_rate = 0.005
train_verbose = True  # show epoch
model_name = 'vgg11_bn'
num_folds = 5

experiment_name = model_name + \
                '_CV' + str(num_folds) + \
                '_epochs' + str(num_epochs) + \
                '_lr' + str(learning_rate) + \
                '_batchsize' + str(batch_size)

# create dataframe to store test evaluation results
accuracy_df = pd.DataFrame(columns=['fold', 'accuracy train', 'accuracy test', 'weighted accuracy train', 'weighted accuracy test', 'best_epoch'])

# load dataset stats
# read statistics/class count of the dataset
dataset_stats = pd.read_csv(data_path + 'class_stats.csv', index_col=0)
class_names = dataset_stats['class'].values

# read SNR count of the dataset
snr_stats = pd.read_csv(data_path + 'SNR_stats.csv', index_col=0)
snr_list = snr_stats['SNR'].values

# evaluate model on test set for each fold
for fold in range(num_folds):

    # create path to store plot for diffrent folds
    act_result_path = result_path + experiment_name + '/'
    try:
        os.mkdir(act_result_path + 'plots/fold' + str(fold) + '/')
    except OSError as error:
        print(error)

    # load experiment results
    save_filename = 'results_fold' + str(fold) + '.pkl'

    infile = open(act_result_path + save_filename, 'rb')
    exp_result_dict = pkl.load(infile)
    infile.close()

    exp_result_dict.keys()
    targets = exp_result_dict['test_targets']
    predictions = exp_result_dict['test_predictions']
    snrs = exp_result_dict['test_snrs']
    eval_acc = exp_result_dict['test_acc']
    eval_weighted_acc = exp_result_dict['test_weighted_acc']
    best_epoch = exp_result_dict['best_epoch']
    
    # get best epoch
    train_acc = exp_result_dict['train_acc'][best_epoch]
    train_weighted_acc = exp_result_dict['train_weighted_acc'][best_epoch]

    # store accuray of current fold in dataframe
    accuracy_df = pd.concat([accuracy_df, pd.DataFrame.from_records([{'fold': fold, 'accuracy train': train_acc, 'accuracy test': eval_acc, 'weighted accuracy train': train_weighted_acc, 'weighted accuracy test': eval_weighted_acc, 'best_epoch': best_epoch}])])

    target_classes = np.unique(targets)
    pred_classes = np.unique(predictions)
    eval_classes = np.union1d(target_classes, pred_classes)
    eval_class_names = [class_names[int(x)] for x in eval_classes]

    # print('Got ' + str(len(target_classes)) + ' target classes')
    # print('Got ' + str(len(pred_classes)) + ' prediction classes')
    # print('Resulting in ' + str(len(eval_classes)) + ' total classes')
    # print(eval_class_names)

    snrs_unique, snr_counts = np.unique(snrs, return_counts=True)
    # print('Got', snrs_unique, 'SNRs with counts', snr_counts)

    # create dataframe to store test evaluation results per snr
    if fold == 0:
        snr_df_cols = ['acc fold' + str(x) for x in range(num_folds)] + ['balanced_acc fold' + str(x) for x in range(num_folds)]

        snr_accuracy_df = pd.DataFrame(columns=['SNR'] + snr_df_cols)
        snr_accuracy_df['SNR'] = snrs_unique

    # get counts for classes in targets and put them in a df with all classes
    target_class_names = [class_names[int(x)] for x in target_classes]
    target_class_counts = np.unique(targets, return_counts=True)[1]
    target_class_dict = dict(zip(target_class_names, target_class_counts))

    target_class_stats_df = pd.DataFrame({'target_class_counts': 0}, index=class_names)

    for key, value in target_class_dict.items():
        target_class_stats_df.loc[key, 'target_class_counts'] = value

    cf_matrix = confusion_matrix(targets, predictions, labels=eval_classes, normalize='true')
    # cf_matrix = confusion_matrix(targets, predictions, labels=eval_classes)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in eval_class_names],
                        columns=[i for i in eval_class_names])

    plt_title = experiment_name + '\n' + \
                'Acc: ' + str(round(eval_acc, 2)) + ' wAcc: ' + str(round(eval_weighted_acc, 2)) + '\n' + \
                '\n\n' + \
                'classwise accuracy (normalised row wise)'
    
    plt_filename = act_result_path + 'plots/fold' + str(fold) + '/cm.png'

    plot_cm_plus_class_dist(df_cm, target_class_stats_df, plt_title, plt_filename)

    # eval
    for i, act_snr in enumerate(snrs_unique):
        # print(i, act_snr)
        act_snr_sample_indices = np.where(snrs == act_snr)[0]

        # compute accuracies    
        act_targets = targets[act_snr_sample_indices]
        act_predictions = predictions[act_snr_sample_indices]
        act_acc = accuracy_score(y_true=act_targets, y_pred=act_predictions)
        act_balanced_acc = balanced_accuracy_score(y_true=act_targets, y_pred=act_predictions)

        snr_accuracy_df.loc[snr_accuracy_df['SNR'] == act_snr, 'acc fold' + str(fold)] = act_acc
        snr_accuracy_df.loc[snr_accuracy_df['SNR'] == act_snr, 'balanced_acc fold' + str(fold)] = act_balanced_acc

        # compute confusion matrix
        
        # get counts for classes in targets and put them in a df with all classes
        target_class_names = [class_names[int(x)] for x in target_classes]
        target_class_counts = np.unique(act_targets, return_counts=True)[1]
        target_class_dict = dict(zip(target_class_names, target_class_counts))
        target_class_stats_df = pd.DataFrame({'target_class_counts': 0}, index=class_names)
        for key, value in target_class_dict.items():
            target_class_stats_df.loc[key, 'target_class_counts'] = value

        cf_matrix = confusion_matrix(act_targets, act_predictions, labels=eval_classes, normalize='true')
        # cf_matrix = confusion_matrix(targets, predictions, labels=eval_classes)
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in eval_class_names],
                            columns=[i for i in eval_class_names])


        plt_title = experiment_name + '\n' + \
                'SNR' + str(act_snr) + ' Acc: ' + str(round(act_acc, 2)) + ' wAcc: ' + str(round(act_balanced_acc, 2)) + '\n' + \
                '\n\n' + \
                'classwise accuracy (normalised row wise)'

        plt_filename = act_result_path + 'plots/fold' + str(fold) + '/cm_SNR' + str(act_snr) + '.png'
        # plt_filename=None
        plot_cm_plus_class_dist(df_cm, target_class_stats_df, plt_title, plt_filename)

# compute mean and std of snr accuracies
snr_df_cols = ['acc fold' + str(x) for x in range(num_folds)]
snr_accuracy_df['mean acc'] = snr_accuracy_df[snr_df_cols].mean(axis=1)
snr_accuracy_df['std acc'] = snr_accuracy_df[snr_df_cols].std(axis=1)

snr_df_cols = ['balanced_acc fold' + str(x) for x in range(num_folds)]
snr_accuracy_df['mean balanced acc'] = snr_accuracy_df[snr_df_cols].mean(axis=1)
snr_accuracy_df['std balanced acc'] = snr_accuracy_df[snr_df_cols].std(axis=1)

# save snr accuracy df
snr_accuracy_df.to_csv(act_result_path + 'acc_per_SNR.csv')
print(snr_accuracy_df)

# compute mean and std of overall accuracies
accuracy_df = pd.concat([accuracy_df, accuracy_df.mean(axis=0).to_frame().transpose()])
accuracy_df = pd.concat([accuracy_df, accuracy_df.std(axis=0).to_frame().transpose()])
accuracy_df.reset_index(inplace=True, drop=True)
accuracy_df.iloc[num_folds, 0] = 'mean'
accuracy_df.iloc[num_folds + 1, 0] = 'std'
accuracy_df.to_csv(act_result_path + 'acc.csv')