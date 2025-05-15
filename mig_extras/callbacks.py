from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt
import os

class MetricsHistory(Callback):
    def __init__(self, log_dir=None):
        super().__init__()

        self.log_dir = log_dir

        self.val_loss = []
        self.val_mae = []
        self.val_mape = []
        self.val_mae_at_15 = []
        self.val_mae_at_30 = []
        self.val_mae_at_60 = []

        self.train_loss = [0]
        self.train_mae = [0]
        self.train_mape = [0]
        self.train_mae_at_15 = [0]
        self.train_mae_at_30 = [0]
        self.train_mae_at_60 = [0]

    # def on_train_start(self, trainer, pl_module):
    #     # 1) Si tienes un logger explícito (ej. TensorBoardLogger)
    #     if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
    #         self.log_dir = trainer.logger.log_dir
    #     else:
    #         # 2) Caída a default_root_dir
    #         self.log_dir = trainer.default_root_dir

    #     # Asegúrate de que exista la carpeta
    #     os.makedirs(self.log_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        # Note: pl automatically logs 'train_loss_epoch' if you log loss in training_step
        self.train_loss.append(m['train_loss'].cpu().item())
        self.train_mae.append(m['train_mae'].cpu().item())
        # self.train_mape.append(m['train_mape'].cpu().item())
        self.train_mae_at_15.append(m['train_mae_at_15'].cpu().item())
        self.train_mae_at_30.append(m['train_mae_at_30'].cpu().item())
        self.train_mae_at_60.append(m['train_mae_at_60'].cpu().item())

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics


        # Note: pl automatically logs 'train_loss_epoch' if you log loss in training_step
        self.val_loss.append(m['val_loss'].cpu().item())
        self.val_mae.append(m['val_mae'].cpu().item())
        # self.val_mape.append(m['val_mape'].cpu().item())
        self.val_mae_at_15.append(m['val_mae_at_15'].cpu().item())
        self.val_mae_at_30.append(m['val_mae_at_30'].cpu().item())
        self.val_mae_at_60.append(m['val_mae_at_60'].cpu().item())


    def on_train_end(self, trainer, pl_module):
        # Save the metrics to a file
        epochs = list(range(1, len(self.val_loss) + 1))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Loss subplot
        # axs[0].plot(epochs, self.train_loss, label='Train Loss')
        axs[0].plot(epochs, self.val_loss,   label='Val Loss')
        axs[0].plot(epochs[1:], self.train_loss[1:],  label='Train Loss')
        axs[0].set_title('Loss over Epochs')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)

        # Metrics subplot
        axs[1].plot(epochs, self.val_mae,       label='Val MAE')
        axs[1].plot(epochs[1:], self.train_mae[1:],     label='Train MAE')
        axs[1].plot(epochs, self.val_mae_at_15, label='Val MAE@15min')
        axs[1].plot(epochs[1:], self.train_mae_at_15[1:], label='Train MAE@15min')
        axs[1].plot(epochs, self.val_mae_at_30, label='Val MAE@30min')
        axs[1].plot(epochs[1:], self.train_mae_at_30[1:], label='Train MAE@30min')
        axs[1].plot(epochs, self.val_mae_at_60, label='Val MAE@60min')
        axs[1].plot(epochs[1:], self.train_mae_at_60[1:], label='Train MAE@60min')

        # add a second axis to the right for mape
        # ax2 = axs[1].twinx()
        # ax2.plot(epochs, self.val_mape, label='Val MAPE', color='orange')
        # ax2.plot(epochs[1:], self.train_mape, label='Train MAPE', color='red')
        # ax2.set_ylabel('MAPE')
        # ax2.legend(loc='upper right')
        # ax2.grid(False)
        
        axs[1].set_title('Validation Metrics over Epochs')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Metric Value')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir + '/metrics_history.png')
        plt.close(fig)

        # Optionally, you can also save the metrics to a CSV file

        with open(os.path.join(self.log_dir, 'metrics_history.txt'), 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Train MAE,Val MAE,Train MAE@15min,Val MAE@15min,Train MAE@30min,Val MAE@30min,Train MAE@60min,Val MAE@60min\n')
            for i in range(len(self.val_loss)):
                f.write(f"{i+1},{self.train_loss[i]},{self.val_loss[i]},{self.train_mae[i]},{self.val_mae[i]},{self.train_mae_at_15[i]},{self.val_mae_at_15[i]},{self.train_mae_at_30[i]},{self.val_mae_at_30[i]},{self.train_mae_at_60[i]},{self.val_mae_at_60[i]}\n")
        # Save the metrics to a CSV file
        with open(os.path.join(self.log_dir, 'metrics_history.csv'), 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Train MAE,Val MAE,Train MAE@15min,Val MAE@15min,Train MAE@30min,Val MAE@30min,Train MAE@60min,Val MAE@60min\n')
            for i in range(len(self.val_loss)):
                f.write(f"{i+1},{self.train_loss[i]},{self.val_loss[i]},{self.train_mae[i]},{self.val_mae[i]},{self.train_mae_at_15[i]},{self.val_mae_at_15[i]},{self.train_mae_at_30[i]},{self.val_mae_at_30[i]},{self.train_mae_at_60[i]},{self.val_mae_at_60[i]}\n")
    # def on_test_end(self, trainer, pl_module):
    #     m = trainer.callback_metrics



