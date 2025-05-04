import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from combined_metrics.csv')
    parser.add_argument('csv_file', help='Path to combined_metrics.csv')
    parser.add_argument('--output_dir', default=None, help='Directory to save plots (defaults to CSV file directory)')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv_file)
    output_dir = args.output_dir or os.path.dirname(args.csv_file)
    os.makedirs(output_dir, exist_ok=True)

    epochs = df['epoch']

    # Generate separate plots for each metric
    metrics = [
        ('train_loss', 'val_loss', 'Loss', 'loss'),
        ('train_acc', 'val_acc', 'Accuracy', 'accuracy'),
        ('train_bal_acc', 'val_bal_acc', 'Balanced Accuracy', 'balanced_accuracy'),
    ]
    for train_col, val_col, title, fname in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, df[train_col], 'b-', label=f'Train {title}')
        plt.plot(epochs, df[val_col],   'r-', label=f'Val {title}')
        plt.title(f'Training & Validation {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(output_dir, f'{fname}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {title} plot to {out_path}")

if __name__ == '__main__':
    main() 