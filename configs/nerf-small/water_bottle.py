import os
_base_ = './default.py'

expname = 'small/dnerf_water_bottle_new_data_new_xyz_sdf'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir=os.path.join(os.environ['N3DR_DATA_PATH'], 'data_dnerf/water_bottle_2'),
    dataset_type='dnerf',
    white_bkgd=True,
)

train_config = dict(
    N_iters=10000,
    lrate_variance=0.001,
    lrate_feature=8e-1,
)

model_and_render = dict(
    num_voxels=100**3,
    num_voxels_base=100**3,
    world_bound_scale=2.05,
    representation_type='sdf',
)