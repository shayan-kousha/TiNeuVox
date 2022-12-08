import os
_base_ = './default.py'

expname = 'small/dnerf_water_bottle_new_data'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir=os.path.join(os.environ['N3DR_DATA_PATH'], 'data_dnerf/water_bottle_2'),
    dataset_type='dnerf',
    white_bkgd=True,
)

train_config = dict(
    N_iters=20000,
)