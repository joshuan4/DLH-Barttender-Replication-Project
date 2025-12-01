# This file is for replicating the primary barttender experiments
# Training the DenseNet model on our 3 sets of images created in barttender_processing_replication.py and generating metrics

from mimic_constants import *
import importlib
import os
import wandb
os.environ['WANDB_MODE'] = 'offline'
import fire

from barttender.mimic_k_fold import main

if __name__ == '__main__':
    main(
        image_type='xray',
        batch_size=32,
        epochs=100,
        n_splits=10,
        idp=True,
        num_workers=0
    )

    main(
        image_type='blank',
        batch_size=32,
        epochs=100,
        n_splits=10,
        idp=True,
        num_workers=0
    )

    main(
        image_type='xray',
        batch_size=32,
        epochs=100,
        n_splits=10,
        idp=True,
        num_workers=0,
        no_bars=True
    )
