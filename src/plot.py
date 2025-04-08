import matplotlib.pyplot as plt
import numpy as np

def plot_training_summary(train_means, train_stds, experiment_name):

    (train_loss_mean, train_acc_mean, val_acc_mean) = train_means
    (train_loss_std, train_acc_std, val_acc_std) = train_stds

    epochs = np.arange(1, len(train_loss_mean) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_mean, label="Train Loss (mean)")
    plt.fill_between(
        epochs,
        train_loss_mean - train_loss_std,
        train_loss_mean + train_loss_std,
        alpha=0.3,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Mean ± Std)")
    plt.legend()

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_mean, label="Train Acc (mean)")
    plt.fill_between(
        epochs,
        train_acc_mean - train_acc_std,
        train_acc_mean + train_acc_std,
        alpha=0.3,
    )
    plt.plot(epochs, val_acc_mean, label="Val Acc (mean)", color="orange")
    plt.fill_between(
        epochs,
        val_acc_mean - val_acc_std,
        val_acc_mean + val_acc_std,
        alpha=0.3,
        color="orange",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy (Mean ± Std)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}.png")
    plt.close()
