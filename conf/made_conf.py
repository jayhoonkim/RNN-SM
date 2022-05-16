import os
import hydra
import datetime
from omegaconf import OmegaConf
from utils.config_utils_tf import register_config

def made_conf(data_list):
    max_tar_len, max_src_len, src_vocab_size, tar_vocab_size = data_list
    drive_project_root = './result'
    
    # data configuration
    data_iot_word_cfg = {
        "name": "iot_word_cfg",
        "src": {
            "vocab_size": src_vocab_size,
            "max_len": max_src_len
        },
        "tar": {
            "vocab_size": tar_vocab_size,
            "max_len": max_tar_len
        },
        "train_val_test_split_ratio": [0.8, 0.1, 0.1],
        "train_val_shuffle": True,
    }

    # model configuration
    model_translate_Stacked_RNN_cfg = {
        "name": "Stacked_RNN",
        "enc": {
            "embed_size": 256,
            "rnn": {
                "units": 1024,
            },
        },
        "dec": {
            "embed_size": 256,
            "rnn": {
                "units": 1024,
            },
        }
    }


    # optimizer configs
    adam_warmup_lr_sch_opt_cfg = {
        "optimizer": {
            "name": "Adam",
            "other_kwargs": {},
        },
        "lr_scheduler": {
            "name": "LinearWarmupLRScheduler",
            "kwargs": {
                "lr_peak": 1e-3,
                "warmup_end_steps": 1500,
            },
        },
    }

    radam_no_lr_sch_opt_cfg = {
        "optimizer": {
            "name": "RectifiedAdam",
            "learning_rate": 1e-3,
            "other_kwargs": {},
        },
        "lr_scheduler": None,
    }

    # train_cfg
    train_cfg: dict = {
        "train_batch_size": 128,
        "val_batch_size": 32,
        "test_batch_size": 32,
        "max_epochs": 300,
        "distribute_strategy": "MirroredStrategy",
        "teacher_forcing_ratio": 0.5,
    }

    _merged_cfg_presets: dict = {
        "Stacked_RNN_radam": {
            "data": data_iot_word_cfg,
            "model": model_translate_Stacked_RNN_cfg,
            "opt": radam_no_lr_sch_opt_cfg,
            "train": train_cfg,
        },
    }    
    
    ### hydra composition ###
    # clear hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # register preset configs
    register_config(_merged_cfg_presets)

    # initialization
    hydra.initialize(config_path=None)

    using_config_key = "Stacked_RNN_radam"
    cfg = hydra.compose(using_config_key)
    
    # override log _cfg
    model_name = cfg.model.name
    run_name = f"{datetime.datetime.now().isoformat(timespec='seconds')}-{using_config_key}-{model_name}"
    log_dir = os.path.join(drive_project_root, run_name)

    log_cfg = {
        "run_name": run_name,
        "checkpoint_filepath": os.path.join(log_dir, "model"),
        "tensorboard_log_dir": log_dir,
        "callbacks": {
            "TensorBoard": {
                "log_dir": log_dir,
                "update_freq": 50,
            },
            "EarlyStopping": {
                "patience": 30,
                "verbose": True,
            },
        },
    }

    # unlock struct of config & set log config
    OmegaConf.set_struct(cfg, False)
    cfg.log = log_cfg

    # relock config
    OmegaConf.set_struct(cfg, True)

    # save yaml
    with open('./conf/config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)
        
    return cfg