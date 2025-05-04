import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import os
import argparse

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description='Plot Test Balanced Accuracy vs SNR for a given experiment/fold')
parser.add_argument('--exp', '-e', required=True, help='Experiment directory name under results/experiments')
parser.add_argument('--fold', '-f', type=int, default=0, help='Fold number to use')
parser.add_argument('--base_path', '-b', default='./results/experiments', help='Base path to experiment results')
args = parser.parse_args()

experiment_name = args.exp
fold_number = args.fold
base_result_path = args.base_path.rstrip('/') + '/'
# ------------------

# Construct the path to the results file
result_file_path = os.path.join(base_result_path, experiment_name, f'results_fold{fold_number}.pkl')
plot_output_dir = os.path.join(base_result_path, experiment_name, 'plots')

# Ensure the plot directory exists
os.makedirs(plot_output_dir, exist_ok=True)

# Check if results file exists
if not os.path.exists(result_file_path):
    print(f"ERROR: Results file not found at: {result_file_path}")
    print("Please ensure the training script has run successfully and the experiment_name is correct.")
    exit()

# Load the results dictionary
print(f"Loading results from: {result_file_path}")
try:
    with open(result_file_path, 'rb') as f:
        results = pkl.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

# Extract necessary data
# Convert to numpy arrays if they are torch tensors for easier handling
try:
    predictions = results['test_predictions'].cpu().numpy()
    targets = results['test_targets'].cpu().numpy()
    snrs = results['test_snrs'].cpu().numpy()
    class_names = results.get('class_names', None) # Optional: get class names if saved
except KeyError as e:
    print(f"ERROR: Missing key in results dictionary: {e}")
    print("The results file might be incomplete or from an older version.")
    exit()
except AttributeError:
    # If already numpy arrays (or other types without .cpu().numpy())
    predictions = np.array(results['test_predictions'])
    targets = np.array(results['test_targets'])
    snrs = np.array(results['test_snrs'])
    class_names = results.get('class_names', None)


print(f"Loaded {len(targets)} test results.")

# Get unique SNRs present in the test set and sort them
unique_snrs = np.unique(snrs)
unique_snrs.sort()
print(f"Unique SNRs found in test set: {unique_snrs}")

# Calculate balanced accuracy for each unique SNR
snr_balanced_accuracies = []
sample_counts_per_snr = []

print("Calculating Balanced Accuracy per SNR...")
for snr_val in unique_snrs:
    # Find indices of samples matching the current SNR
    indices = np.where(snrs == snr_val)[0]
    sample_counts_per_snr.append(len(indices))

    if len(indices) > 0:
        # Filter predictions and targets for this SNR
        preds_snr = predictions[indices]
        targets_snr = targets[indices]

        # Calculate balanced accuracy for this subset
        # Note: balanced_accuracy_score default 'adjusted=False' is equivalent to macro average accuracy
        bal_acc = balanced_accuracy_score(targets_snr, preds_snr)
        snr_balanced_accuracies.append(bal_acc)
        print(f"  SNR: {snr_val:>4} dB | Samples: {len(indices):>5} | Balanced Accuracy: {bal_acc:.4f}")
    else:
        # Should not happen if unique_snrs is derived from snrs, but handle defensively
        snr_balanced_accuracies.append(np.nan)
        print(f"  SNR: {snr_val:>4} dB | Samples: {len(indices):>5} | Balanced Accuracy: N/A")


# Create the plot
plt.figure(figsize=(12, 7))

# Use a bar chart - often better for discrete categories like SNR
bar_width = (unique_snrs[1] - unique_snrs[0]) * 0.6 if len(unique_snrs) > 1 else 1.0 # Auto-adjust width or default
plt.bar(unique_snrs, snr_balanced_accuracies, width=bar_width, label='Balanced Accuracy', color='skyblue', edgecolor='black')

# Add sample count text above bars
for i, count in enumerate(sample_counts_per_snr):
    plt.text(unique_snrs[i], snr_balanced_accuracies[i] + 0.02, f'n={count}', ha='center', va='bottom', fontsize=9)

# Alternatively, use a line plot:
# plt.plot(unique_snrs, snr_balanced_accuracies, marker='o', linestyle='-', label='Balanced Accuracy')

plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]")
plt.ylabel("Balanced Accuracy")
plt.title("Test Set Balanced Accuracy vs. SNR (18 Epochs)")
plt.xticks(unique_snrs) # Ensure ticks land exactly on the SNR values
plt.ylim(0, 1.05) # Set y-axis from 0% to 105%
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
output_plot_path = os.path.join(plot_output_dir, f'snr_vs_balanced_acc_fold{fold_number}.png')
plt.savefig(output_plot_path)
print(f"\nSaved SNR vs Balanced Accuracy plot to: {output_plot_path}")

# Display the plot
plt.show()
