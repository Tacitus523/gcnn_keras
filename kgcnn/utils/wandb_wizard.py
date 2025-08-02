from typing import Dict

import wandb
from wandb.integration.keras import WandbMetricsLogger

def init_wandb(train_config: Dict) -> None:       
    wandb.init(
        project=train_config["wandb_project"],
        entity=train_config["wandb_entity"],
        name=train_config["wandb_name"],
        config=train_config,
        reinit="finish_previous"
    )
            
    wandb.run.define_metric("charge_rmse", summary="min")
    wandb.run.define_metric("energy_rmse", summary="min")
    wandb.run.define_metric("force_rmse", summary="min")
    wandb.run.define_metric("charge_mae", summary="min")
    wandb.run.define_metric("energy_mae", summary="min")
    wandb.run.define_metric("force_mae", summary="min")
    wandb.run.define_metric("charge_r2", summary="max")
    wandb.run.define_metric("energy_r2", summary="max")
    wandb.run.define_metric("force_r2", summary="max")


class CustomWandbCallback(WandbMetricsLogger):
    def __init__(self, *args, key_prefix: str = "", **kwargs):
        super(CustomWandbCallback, self).__init__(*args, **kwargs)
        self.key_prefix = key_prefix

    def on_epoch_end(self, epoch, logs=None):
        modified_logs = {f"{self.key_prefix}/epoch": epoch}
        if logs is not None:
            for key, value in logs.items():
                modified_key = "/".join([self.key_prefix, key]).strip()
                modified_logs[modified_key] = value
        
        wandb.log(modified_logs, commit=True)
        #super().on_epoch_end(epoch, modified_logs)

def construct_wandb_callback(key_prefix: str = "") -> CustomWandbCallback:
    # If training 2 models they will have the same keys in the history and therefore we might need to change some keys with a prefix
    return CustomWandbCallback(key_prefix=key_prefix)

def log_wandb_metrics(metrics: Dict) -> None:
    wandb.log(metrics, commit=True)

def finish_wandb():
    # [optional] finish the wandb run, necessary in notebooks
    wandb.run.finish()
    