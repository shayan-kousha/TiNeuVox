_base_ = './default.py'

expname = 'small/dnerf_water_bottle'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data_dnerf/water_bottle',
    dataset_type='dnerf',
    white_bkgd=True,
)

train_config = dict(
    N_iters=20000,
)