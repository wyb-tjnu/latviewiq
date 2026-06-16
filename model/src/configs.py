# config file
from config import Config

def config():
    config = Config({

        'wandb' : False,
        'wandb_name': '1111',

        # device
        'gpu_id': "0",
        'num_workers': 0,

        # ViT structure
        'n_enc_seq': [16,16],
        'n_layer': 14,
        'd_hidn': 384,
        'i_pad': 0,
        'd_ff': 384,
        'd_MLP_head': 1152,
        'n_head': 6,
        'd_head': 384,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'layer_norm_epsilon': 1e-12,
        'n_output': 1,
        'batch_size': 4,
        'patch_size': 32,

        'n_epoch': 100,
        'learning_rate': 1e-4,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'T_max': 3e4,
        'eta_min': 0,
        'save_freq': 1,
        'val_freq': 5,
        'train_size': 0.6,
        'val_size' : 0.2,

        # tangent images
        'global_data_path': '',
        'tan_data_path': '',
        'txt_file_tans': '',


        'split_avail' :True,
        'train_list': '',
        'test_list': '',

        'snap_path': './weights',
        'checkpoint': None,
        'if_test':True
    })
    return config
