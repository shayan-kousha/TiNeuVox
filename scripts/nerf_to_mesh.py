import argparse
import os
import numpy as np
import torch
import mmcv

from tqdm import tqdm
import sys
sys.path.insert(1, './')
from lib import tineuvox, utils

from importlib import import_module
from pytorch3d.structures import Meshes
from skimage import measure

def batchify(*data, batch_size=1024, device="cpu", progress=True):
    assert all(sample is None or sample.shape[0] == data[0].shape[0] for sample in data), \
        "Sizes of tensors must match for dimension 0."

    # Custom batchifier
    def batchifier():
        # Data size and current batch offset
        size, batch_offset = data[0].shape[0], 0

        while batch_offset < size:
            # Subsample slice
            batch_slice = slice(batch_offset, batch_offset + batch_size)

            # Yield each subsample, and move to available device
            yield [sample[batch_slice].to(device) if sample is not None else sample for sample in data]

            batch_offset += batch_size

    iterator = batchifier()
    if not progress:
        return iterator

    # Total batches
    total = (data[0].shape[0] - 1) // batch_size + 1

    return tqdm(iterator, total=total)

def export_obj(vertices, triangles, diffuse, normals, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    print('Writing to obj...')

    with open(filename, "w") as fh:

        for index, v in enumerate(vertices):
            fh.write("v {} {} {}".format(*v))
            if len(diffuse) > index:
                fh.write(" {} {} {}".format(*diffuse[index]))

            fh.write("\n")

        for n in normals:
            fh.write("vn {} {} {}\n".format(*n))

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")

    print(f"Finished writing to {filename} with {len(vertices)} vertices")


def create_mesh(vertices, faces_idx):
    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    vertices = vertices - vertices.mean(0)
    scale = max(vertices.abs().max(0)[0])
    vertices = vertices / scale

    # We construct a Meshes structure for the target mesh
    target_mesh = Meshes(verts=[vertices], faces=[faces_idx])

    return target_mesh


def extract_radiance(model, args, device, nums, **kwargs):
    assert (isinstance(nums, tuple) or isinstance(nums, list) or isinstance(nums, int)), \
        "Nums arg should be either iterable or int."

    if isinstance(nums, int):
        nums = (nums,) * 3
    else:
        assert (len(nums) == 3), "Nums arg should be of length 3, number of axes for 3D"

    # Create sample tiles
    tiles = [torch.linspace(-args.limit, args.limit, num) for num in nums]

    # Generate 3D samples
    samples = torch.stack(torch.meshgrid(*tiles), -1).view(-1, 3).float()

    radiance_samples = []
    for (samples,) in batchify(samples, batch_size=args.batch_size, device=device):
        # Query radiance batch
        radiance_batch = model(
            rays_o=None,
            rays_d=samples,
            viewdirs=samples/samples.norm(dim=1).unsqueeze(1),
            times_sel=torch.zeros([samples.shape[0], 1]).to(device),
            ray_pts=samples,
            **kwargs['render_kwargs'])

        # Accumulate radiance
        radiance_samples.append(torch.cat((radiance_batch['raw_rgb'], radiance_batch['density_result']), dim=1).cpu())

    # Radiance 3D grid (rgb + density)
    radiance = torch.cat(radiance_samples, 0).view(*nums, 4).contiguous().numpy()

    return radiance


def extract_iso_level(density, args):
    # Density boundaries
    min_a, max_a, std_a = density.min(), density.max(), density.std()

    # Adaptive iso level
    iso_value = min(max(args.iso_level, min_a + std_a), max_a - std_a)
    print(f"Min density {min_a}, Max density: {max_a}, Mean density {density.mean()}")
    print(f"Querying based on iso level: {iso_value}")

    return iso_value


def extract_geometry(model, device, args, **kwargs):
    # Sample points based on the grid
    radiance = extract_radiance(model, args, device, args.res, **kwargs)

    # Density grid
    density = radiance[..., 3]

    # Adaptive iso level
    iso_value = extract_iso_level(density, args)

    # Extracting iso-surface triangulated
    results = measure.marching_cubes(density, iso_value)

    # Use contiguous tensors
    vertices, triangles, normals, _ = [torch.from_numpy(np.ascontiguousarray(result)) for result in results]

    # Use contiguous tensors
    normals = torch.from_numpy(np.ascontiguousarray(normals))
    vertices = torch.from_numpy(np.ascontiguousarray(vertices))
    triangles = torch.from_numpy(np.ascontiguousarray(triangles))

    # Normalize vertices, to the (-limit, limit)
    vertices = args.limit * (vertices / (args.res / 2.) - 1.)

    return vertices, triangles, normals, density


def extract_geometry_with_super_sampling(model, device, args):
    raise NotImplementedError
    try:
        mcubes = import_module("marching_cubes")
    except ModuleNotFoundError:
        print("""
            Run the following instructions within your environment:
            https://github.com/JustusThies/PyMarchingCubes#installation
        """)

        # Close process
        exit(-1)

    # Sampling resolution per axis
    nums = np.array([args.res + (args.res - 1) * args.super_sampling, args.res, args.res])

    # Radiance per axis, super sampling across each axis
    radiances = []
    for i in range(0, 3):
        # Roll such that each axis is rich
        radiance_axis = extract_radiance(model, args, device, np.roll(nums, i))[..., 3]

        radiances.append(radiance_axis)

    # accumulate
    density = np.stack(radiances, axis=0)

    # Adaptive iso level
    iso_value = extract_iso_level(density, args)

    vertices, triangles = mcubes.marching_cubes_super_sampling(*radiances, iso_value)

    vertices = np.ascontiguousarray(vertices)
    mcubes.export_obj(vertices, triangles, os.path.join(args.save_dir, "mesh.obj"))

class RaySampleInterval(torch.nn.Module):
    def __init__(self, count):
        super(RaySampleInterval, self).__init__()
        self.count = count

        # Ray sample count
        point_intervals = torch.linspace(0.0, 1.0, self.count, requires_grad = False)[None, :]
        self.register_buffer("point_intervals", point_intervals, persistent = False)

    def forward(self, cfg, ray_count, near, far):
        if len(near.shape) > 0 and near.shape[0] == ray_count:
            near, far = near[:, None], far[:, None]

        # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
        # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
        if not cfg['lindisp']:
            point_intervals = near * (1.0 - self.point_intervals) + far * self.point_intervals
        else:
            point_intervals = 1.0 / (1.0 / near * (1.0 - self.point_intervals) + 1.0 / far * self.point_intervals)

        if len(near.shape) == 0 or near.shape[0] != ray_count:
            point_intervals = point_intervals.expand([ ray_count, self.count ])

        if cfg['perturb']:
            # Get intervals between samples.
            mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
            upper = torch.cat((mids, point_intervals[..., -1:]), dim = -1)
            lower = torch.cat((point_intervals[..., :1], mids), dim = -1)

            # Stratified samples in those intervals.
            t_rand = torch.rand(
                point_intervals.shape,
                dtype = point_intervals.dtype,
                device = point_intervals.device,
            )

            point_intervals = lower + (upper - lower) * t_rand

        return point_intervals

def intervals_to_ray_points(point_intervals, ray_directions, ray_origin):
    ray_points = ray_origin[..., None, :] + ray_directions[..., None, :] * point_intervals[..., :, None]

    return ray_points

def export_marching_cubes(model, args, cfg, device, **kwargs):
    # Mesh Extraction

    if args.super_sampling >= 1:
        print("Generating mesh geometry...")

        # Extract model geometry with super sampling across each axis
        extract_geometry_with_super_sampling(model, device, args)
        return

    # Cached mesh path containing data
    mesh_cache_path = os.path.join(args.save_dir, args.cache_name)

    cached_mesh_exists = os.path.exists(mesh_cache_path)
    cache_new_mesh = args.use_cached_mesh and not cached_mesh_exists
    if cache_new_mesh:
        print(f"Cached mesh does not exist - {mesh_cache_path}")

    if args.use_cached_mesh and cached_mesh_exists:
        print("Loading cached mesh geometry...")
        vertices, triangles, normals, density = torch.load(mesh_cache_path)
    else:
        print("Generating mesh geometry...")
        # Extract model geometry
        vertices, triangles, normals, density = extract_geometry(model, device, args, **kwargs)

        if cache_new_mesh or args.override_cache_mesh:
            torch.save((vertices, triangles, normals, density), mesh_cache_path)
            print(f"Cached mesh geometry saved to {mesh_cache_path}")

    # Extracting the mesh appearance

    # Ray targets and directions
    targets, directions = vertices, -normals

    diffuse = []
    if args.no_view_dependence:
        print("Diffuse map query directly  without specific-views...")
        # Query directly without specific-views
        batch_generator = batchify(targets, directions, batch_size=args.batch_size, device=device)
        for (pos_batch, dir_batch) in batch_generator:
            # Diffuse batch queried
            diffuse_batch = model.sample_ray(pos_batch, dir_batch)[..., :3]

            # Accumulate diffuse
            diffuse.append(diffuse_batch.cpu())
    else:
        print("Diffuse map query with view dependence...")
        ray_bounds = torch.tensor([0., args.view_disparity_max_bound], dtype=directions.dtype)

        # Query with view dependence
        # Move ray origins slightly towards negative sdf
        ray_origins = targets - args.view_disparity * directions

        print("Started ray-casting")
        batch_generator = batchify(ray_origins, directions, batch_size=args.batch_size, device=device)
        for (ray_origins, ray_directions) in batch_generator:
            # # View dependent diffuse batch queried
            # x = (ray_origins, ray_directions, ray_bounds)
            # ray_origins, ray_directions, (near, far) = x

            # # Get current configuration
            # nerf_cfg = {
            #     'chunksize': 2048,
            #     'lindisp': False,
            #     'num_coarse': 64,
            #     'num_fine': 128,
            #     'num_samples': 1,
            #     'perturb': False,
            #     'radiance_field_noise_std': 0.0
            # }

            # # Generating depth samples
            # ray_count = ray_directions.shape[0]

            # import ipdb;ipdb.set_trace()
            # sampler = RaySampleInterval(192)
            # fine_ray_intervals = sampler(nerf_cfg, ray_count, near, far).to(ray_directions.device)

            # ray_points = intervals_to_ray_points(
            #     fine_ray_intervals, ray_directions, ray_origins
            # )
            # import ipdb;ipdb.set_trace()
            render_kwargs = {
                'bg': 1.0,
                'near': 0.0,
                'far': 1.0,
                'stepsize': 0.5

            }
            output_bundle = model(
                ray_origins,
                ray_directions,
                ray_directions/ray_directions.norm(dim=1).unsqueeze(1),
                torch.zeros([ray_origins.shape[0], 1]).to(device),
                **render_kwargs
            )

            # Accumulate diffuse
            diffuse.append(output_bundle['rgb_marched'].cpu())

    # Query the whole diffuse map
    diffuse = torch.cat(diffuse, dim=0).numpy()

    # Target mesh path
    mesh_path = os.path.join(args.save_dir, args.mesh_name)

    # Export model
    export_obj(vertices, triangles, diffuse, normals, mesh_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument(
        "--log-checkpoint", type=str, default=None,
        help="Training log path with the config and checkpoints to load existent configuration.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="model_last.ckpt",
        help="Load existent configuration from the latest checkpoint by default.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=".",
        help="Save mesh to this directory, if specified.",
    )
    parser.add_argument(
        "--mesh-name", type=str, default="mesh_2.obj",
        help="Mesh name to be generated.",
    )
    parser.add_argument(
        "--iso-level", type=float, default=32,
        help="Iso-level value for triangulation",
    )
    parser.add_argument(
        "--limit", type=float, default=1.,
        help="Limits in -xyz to xyz for marching cubes 3D grid.",
    )
    parser.add_argument(
        "--res", type=int, default=128,
        help="Sampling resolution for marching cubes, increase it for higher level of detail.",
    )
    parser.add_argument(
        "--super-sampling", type=int, default=0,
        help="Add super sampling along the edges.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Higher batch size results in faster processing but needs more device memory.",
    )
    parser.add_argument(
        "--no-view-dependence", action="store_true", default=False,
        help="Disable view dependent appearance, use sampled diffuse color based on the grid"
    )
    parser.add_argument(
        "--view-disparity", type=float, default=1e-2,
        help="Ray origins offset from target based on the inverse normal for the view dependent appearance.",
    )
    parser.add_argument(
        "--view-disparity-max-bound", type=float, default=1e0,
        help="Far max possible bound, usually set to (cfg.far - cfg.near), lower it for better "
             "appearance estimation when using higher resolution e.g. at least view_disparity * 2.0.",
    )
    parser.add_argument(
        "--use-cached-mesh", action="store_true", default=False,
        help="Use the cached mesh.",
    )
    parser.add_argument(
        "--override-cache-mesh", action="store_true", default=False,
        help="Caches the mesh, useful for rapid configuration appearance tweaking.",
    )
    parser.add_argument(
        "--cache-name", type=str, default="mesh_cache.pt",
        help="Mesh cache name, allows for multiple unique meshes of different resolutions.",
    )
    config_args = parser.parse_args()

    # # Existent log path
    # path_parser = PathParser()
    # cfg, _ = path_parser.parse(None, config_args.log_checkpoint, None, config_args.checkpoint)

    # Available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model checkpoint

    cfg = mmcv.Config.fromfile(config_args.config)
    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    ckpt_name = ckpt_path.split('/')[-1][:-4]
    model_class = tineuvox.TiNeuVox
    model = utils.load_model(model_class, ckpt_path).to(device)
    # near=data_dict['near']
    # far=data_dict['far']
    stepsize = cfg.model_and_render.stepsize
    render_viewpoints_kwargs = {
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            # 'near': near,
            # 'far': far,
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'render_depth': True,
        },
    }

    with torch.no_grad():
        # Perform marching cubes and export the mesh
        export_marching_cubes(model, config_args, cfg, device, **render_viewpoints_kwargs)