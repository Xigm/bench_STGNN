import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics

from lib.datasets import LocalGlobalGPVARDataset, AirQuality
from lib.nn import models, EmbeddingPredictor
from lib.utils import find_devices, cfg_to_python

from tsl.nn.models.stgn.dcrnn_model import DCRNNModel as DCRNNModelTSL 
from mig_models.traffic_transformer import TrafficTransformer
from mig_models.DCGRU import DCRNNModel
from mig_extras.callbacks import MetricsHistory


def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == 'ttg_iso':
        model = models.TimeThenGraphIsoModel
    elif model_str == 'ttg_aniso':
        model = models.TimeThenGraphAnisoModel
    elif model_str == 'tag_iso':
        model = models.TimeAndGraphIsoModel
    elif model_str == 'tag_aniso':
        model = models.TimeAndGraphAnisoModel
    # Baseline models  ##################################################
    elif model_str == 'rnn':
        model = models.RNNModel
    elif model_str == 'fcrnn':
        model = models.FCRNNModel
    elif model_str == 'local_rnn':
        model = models.LocalRNNModel
    # SOTA baseline models  #############################################
    elif model_str == 'dcrnn':
        model = models.DCRNNModel
    elif model_str == 'gwnet':
        model = models.GraphWaveNetModel
    elif model_str == 'agcrn':
        model = models.AGCRNModel
    elif model_str == 'dcrnn_TSL':
        model = DCRNNModelTSL
    # Mig Models #####################################################
    elif model_str == 'traffic_transformer':
        model = TrafficTransformer
    elif model_str == 'dcrnn_mig':
        model = DCRNNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif name == 'bay':
        dataset = PemsBay()
    elif name == 'pems3':
        dataset = PeMS03()
    elif name == 'pems4':
        dataset = PeMS04()
    elif name == 'pems7':
        dataset = PeMS07()
    elif name == 'pems8':
        dataset = PeMS08()
    elif name == 'air':
        dataset = AirQuality(impute_nans=True)
    elif name == 'gpvar':
        dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams, p_max=0)
    elif name == 'lgpvar':
        dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run_traffic(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    # create a config dict to pass to the dataset
    # dataset_cfg = DictConfig({"name" : "la"})
    dataset = get_dataset(cfg.dataset)

    # add to dataset as inputs the shifted past values as a second channel
    past_values = dataset.dataframe().shift( 7 * 24 * 12 + 1, fill_value=0.0).values   
    
    # add a fourth dimension to past_values
    # past_values = np.expand_dims(past_values, axis=-1)
    # dataset.add_covariate('past_values', past_values, pattern='t n')

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert not isinstance(dataset, LocalGlobalGPVARDataset)
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)
    
    input_size = torch_dataset.n_channels
    
    # past_values = np.expand_dims(past_values, axis=-1)
    # torch_dataset.add_exogenous('past_values', past_values)
    # input_size = input_size + 1

    if cfg.get('mask_as_exog', False) and 'u' in torch_dataset:
        torch_dataset.update_input_map(u=['u', 'mask'])

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis),
        # 'past_values': StandardScaler(axis=scale_axis),
    }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice)
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)
    # model_cls = get_model_class("dcrnn")

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=input_size,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        embedding_cfg=cfg.get('embedding'), #### changed from None to embedding_cfg
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   "mae_at_15": torch_metrics.MaskedMAE(at=2),
                   "mae_at_30": torch_metrics.MaskedMAE(at=5),
                   "mae_at_60": torch_metrics.MaskedMAE(at=11),
                   'mre': torch_metrics.MaskedMRE(),
                   'mse': torch_metrics.MaskedMSE()}

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = EmbeddingPredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        beta=cfg_to_python(cfg.regularization_weight),
        embedding_var=cfg.embedding.get('initial_var', 0.2),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    ########################################
    # logging options                      #
    ########################################

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    metrics_callback = MetricsHistory(
        log_dir=cfg.run.dir
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=find_devices(1),
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback, metrics_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    trainer.test(predictor, dataloaders=dm.test_dataloader())

    exp_logger.finalize('success')


if __name__ == '__main__':    

    # run_traffic["dataset"] = "la"
    # run_traffic["model"] = "dcrnn"
    # run_traffic["embedding"] = "none"

    exp = Experiment(run_fn=run_traffic, config_path='../config/static',
                     config_name='default')
    res = exp.run()
    logger.info(res)
