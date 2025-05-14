from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt

class MetricsHistory(Callback):
    def __init__(self):
        super().__init__()
        self.val_loss = []
        self.val_mae = []
        self.val_mape = []
        self.val_mae_at_15 = []
        self.val_mae_at_30 = []
        self.val_mae_at_60 = []

        self.train_loss = []
        self.train_mae = []
        self.train_mape = []
        self.train_mae_at_15 = []
        self.train_mae_at_30 = []
        self.train_mae_at_60 = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        # Note: pl automatically logs 'train_loss_epoch' if you log loss in training_step
        self.train_loss.append(m['train_loss'].cpu().item())
        self.train_mae.append(m['train_mae'].cpu().item())
        self.train_mape.append(m['train_mape'].cpu().item())
        self.train_mae_at_15.append(m['train_mae_at_15'].cpu().item())
        self.train_mae_at_30.append(m['train_mae_at_30'].cpu().item())
        self.train_mae_at_60.append(m['train_mae_at_60'].cpu().item())

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        # Note: pl automatically logs 'train_loss_epoch' if you log loss in training_step
        self.val_loss.append(m['val_loss'].cpu().item())
        self.val_mae.append(m['val_mae'].cpu().item())
        self.val_mape.append(m['val_mape'].cpu().item())
        self.val_mae_at_15.append(m['val_mae_at_15'].cpu().item())
        self.val_mae_at_30.append(m['val_mae_at_30'].cpu().item())
        self.val_mae_at_60.append(m['val_mae_at_60'].cpu().item())


def plot_history(history: MetricsHistory):
    epochs = list(range(1, len(history.val_loss) + 1))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Loss subplot
    # axs[0].plot(epochs, history.train_loss, label='Train Loss')
    axs[0].plot(epochs, history.val_loss,   label='Val Loss')
    axs[0].plot(epochs[1:], history.train_loss,  label='Train Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Metrics subplot
    axs[1].plot(epochs, history.val_mae,       label='Val MAE')
    axs[1].plot(epochs[1:], history.train_mae,     label='Train MAE')
    axs[1].plot(epochs, history.val_mae_at_15, label='Val MAE@15min')
    axs[1].plot(epochs[1:], history.train_mae_at_15, label='Train MAE@15min')
    axs[1].plot(epochs, history.val_mae_at_30, label='Val MAE@30min')
    axs[1].plot(epochs[1:], history.train_mae_at_30, label='Train MAE@30min')
    axs[1].plot(epochs, history.val_mae_at_60, label='Val MAE@60min')
    axs[1].plot(epochs[1:], history.train_mae_at_60, label='Train MAE@60min')

    # add a second axis to the right for mape
    ax2 = axs[1].twinx()
    ax2.plot(epochs, history.val_mape, label='Val MAPE', color='orange')
    ax2.plot(epochs[1:], history.train_mape, label='Train MAPE', color='red')
    ax2.set_ylabel('MAPE')
    ax2.legend(loc='upper right')
    ax2.grid(False)
    
    axs[1].set_title('Validation Metrics over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Metric Value')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig()


