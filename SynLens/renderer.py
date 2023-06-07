import os
import torch
import matplotlib.pyplot as plt

import numpy as np
import json
import imageio
import math
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import torch.nn.functional as F

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    TexturesUV,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    HardPhongShader
)


# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("cuda is available.")
else:
    device = torch.device("cpu")

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def camera_arrays(target_cameras, image_size):
    n_sample = 24
    i, j = np.meshgrid(
        np.linspace(-1, 1, n_sample),
        np.linspace(-1, 1, n_sample),
        indexing="ij",
    )
    i = i.T
    j = j.T
    P_sensor = torch.from_numpy(np.stack((i, j), axis=-1)).float()
    P_sensor_flat = P_sensor.reshape((-1, 2))
    P_world = torch.cat((P_sensor_flat, torch.ones((P_sensor_flat.shape[0], 1))), dim=1)
    world_coords = torch.cat((P_sensor_flat, torch.zeros((P_sensor_flat.shape[0], 1))), dim=1)
    world_coords = world_coords.reshape((-1, 3)).numpy()

    x_coords = []
    y_coords = []
    for cameras in target_cameras:
        ndc_points = cameras.transform_points_ndc(P_world)
        camera_points = cameras.transform_points(P_world)
        screen_points = cameras.transform_points_screen(P_world, image_size=(image_size, image_size))
        x_coords.append(screen_points[:,0])
        y_coords.append(screen_points[:,1])

    x_coords = torch.stack(x_coords).cpu().numpy()
    y_coords = torch.stack(y_coords).cpu().numpy()
    return (x_coords, y_coords, world_coords)

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

def image_grid_vis(
    images,
    x_coords,
    y_coords,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im, coord_x, coord_y in zip(axarr.ravel(), images, x_coords, y_coords):
        ax.imshow(im[..., :3])
        ax.scatter(coord_x, coord_y, marker='o', s=8, color='red')
        if not show_axes:
            ax.set_axis_off()

def render_plane():

    # Define the texture and create a TexturesVertex object
    texture_image = plt.imread('target_mesh/target.png')
    texture_image = texture_image[:, :, :3]  # remove alpha channel
    texture_image = torch.from_numpy(texture_image)
    texture_image = texture_image.permute(2, 0, 1).float() / 255.0
    texture_image = torch.unsqueeze(texture_image, dim=0)

    verts_uvs = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    faces_uvs = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.int64)

    # Create a TexturesUV object
    textures = TexturesUV(
        maps=texture_image,
        faces_uvs=faces_uvs.unsqueeze(0),
        verts_uvs=verts_uvs.unsqueeze(0),
    )

    # Create a square mesh with UV coordinates
    vertices = torch.tensor([    [-1.0, -1.0, 0.0],  # bottom left
        [ 1.0, -1.0, 0.0],  # bottom right
        [ 1.0,  1.0, 0.0],  # top right
        [-1.0,  1.0, 0.0],  # top left
    ], dtype=torch.float32)

    faces = torch.tensor([    [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.int64)

    mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=textures,
    )

    # Move the mesh to the center of the scene
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    num_views = 10
    # Define the camera and rasterization settings
    # Get a batch of viewing angles.
    elev = torch.linspace(0, 0, num_views)
    azim = torch.linspace(-20, 20, num_views)
    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    at = np.array([[np.cos(t) * 0.5, np.sin(t) * 0.5, 0] for t in np.linspace(-np.pi, np.pi, num_views)])

    R, T = look_at_view_transform(dist=3.6, elev=elev, azim=azim) #, at=at)

    fov = lens_model["focal"]
    cameras = FoVPerspectiveCameras(device=device, fov=fov, R=R, T=T)

    raster_settings = RasterizationSettings(image_size=595, blur_radius=0.0, faces_per_pixel=1)

    # Create a MeshRasterizer and MeshRenderer object
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = HardPhongShader(device=torch.device("cpu"), lights=None)

    # Render the mesh from different camera perspectives


    # Display the rendered images
    fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
    for i, image in enumerate(images):
        axs[i].imshow(image.squeeze().cpu().numpy())
    plt.show()

def render_mesh(obj_filename, num_views, image_size, render_rgb, lens_model):
    # the number of different viewpoints from which we want to render the mesh.
    # num_views = 200
    # Set paths
    # obj_filename = "schops_mesh/cube.obj"

    # Load obj file
    # mesh = load_objs_as_meshes([obj_filename], device=device)

    # Define the texture and create a TexturesVertex object
    texture_image = plt.imread('target_mesh/target.png')
    texture_image = texture_image[:, :, :3]  # remove alpha channel
    texture_image = torch.from_numpy(texture_image)
    texture_image = texture_image.permute(2, 0, 1).float() / 255.0
    texture_image = torch.unsqueeze(texture_image, dim=0)

    verts_uvs = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    faces_uvs = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.int64)

    # Create a TexturesUV object
    textures = TexturesUV(
        maps=texture_image,
        faces_uvs=faces_uvs.unsqueeze(0),
        verts_uvs=verts_uvs.unsqueeze(0),
    )

    # Create a square mesh with UV coordinates
    vertices = torch.tensor([    [-1.0, -1.0, 0.0],  # bottom left
        [ 1.0, -1.0, 0.0],  # bottom right
        [ 1.0,  1.0, 0.0],  # top right
        [-1.0,  1.0, 0.0],  # top left
    ], dtype=torch.float32)

    faces = torch.tensor([    [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.int64)

    mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=textures,
    )


    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 0, num_views)
    azim = torch.linspace(-20, 20, num_views)
    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    at = np.array([[np.cos(t) * 0.5, np.sin(t) * 0.5, 0] for t in np.linspace(-np.pi, np.pi, num_views)])

    R, T = look_at_view_transform(dist=3.6, elev=elev, azim=azim) #, at=at)

    fov = lens_model["focal"]
    cameras = FoVPerspectiveCameras(device=device, fov=fov, R=R, T=T)
    target_cameras = [FoVPerspectiveCameras(device=device, fov=fov, R=R[None, i, ...],
                                            T=T[None, i, ...]) for i in range(num_views)]

    if render_rgb:
        # Define the settings for rasterization and shading. Here we set the output
        # image to be of size 512X512. As we are rendering images for visualization
        # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
        # rasterize_meshes.py for explanations of these parameters.  We also leave
        # bin_size and max_faces_per_bin to their default values of None, which sets
        # their values using heuristics and ensures that the faster coarse-to-fine
        # rasterization method is used.  Refer to docs/notes/renderer.md for an
        # explanation of the difference between naive and coarse-to-fine rasterization.
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # Create a Phong renderer by composing a rasterizer and a shader. The textured
        # Phong shader will interpolate the texture uv coordinates for each vertex,
        # sample from a texture image and apply the Phong lighting model

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
        # Create a batch of meshes by repeating the mesh and associated textures.
        # Meshes has a useful `extend` method which allows us do this very easily.
        # This also extends the textures.
        meshes = mesh.extend(num_views)

        # Render the mesh from each viewing angle
        target_images = renderer(meshes, cameras=cameras, lights=lights)
        # Our multi-view cow dataset will be represented by these 2 lists of tensors,
        # each of length num_views.
        target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    else:
        target_rgb = None

    return (target_cameras, target_rgb)

def process_mesh(lens_model, lens_name, num_views, image_size, render_rgb, out_fd):
    # print(lens_model)
    obj_filename = "target_mesh/cube.obj"
    if not os.path.exists(out_fd):
        os.mkdir(out_fd)

    # render_plane()
    # sys.exit()
    target_cameras, target_rgb = render_mesh(obj_filename, num_views, image_size, render_rgb, lens_model)
    x_coords, y_coords, world_coords = camera_arrays(target_cameras, image_size)
    screen_coords = np.array([x_coords, y_coords]).T

    from lens_distortion import BrownLensDistortion, ABCDistortion

    projection = {}
    projection['image_width_px'] = image_size
    projection['image_height_px'] = image_size
    projection['center_x_px'] = image_size / 2
    projection['center_y_px'] = image_size / 2

    coeffs = lens_model["coeffs"]
    lens_type = lens_model["lens_type"]
    if lens_type == "abc":
        lens = ABCDistortion(a=coeffs[0], b=coeffs[1]*4, c=coeffs[2], projection=projection)
        cam_coords = lens.distortedFromImage(screen_coords)
    elif lens_type == "brown":
        lens = BrownLensDistortion(k1=coeffs[0], k2=coeffs[1], k3=coeffs[2], projection=projection)
        cam_coords = lens.distortedFromImage(screen_coords)

    board_coords = []
    frame_coords = []

    for i in range(len(target_cameras)):
        frame_coords.append(cam_coords[:, i, :])
        board_coords.append(world_coords)

    with open("{}/{}_{:02d}_points.json".format(out_fd, lens_name, int(lens_model["model_name"])), "w") as outf:
            json.dump(
                {
                    "frame_coordinates_xy": frame_coords,
                    "board_coordinates_xyz": board_coords,
                    "resolution_wh": (image_size, image_size),
                },
                outf,
                cls=NumpyEncoder,
                indent=4,
                sort_keys=False,
            )

    if target_rgb is not None:
        from lens_distortion import Vignetting

        image_grid_vis(target_rgb[:10], x_coords[:10], y_coords[:10], rows=2, cols=5, rgb=True)
        # image_grid(target_rgb[:10], rows=2, cols=5, rgb=True)
        # plt.show()

        vig = Vignetting(k1=-0.19, k2=0.09, k3=-0.39, projection=projection)
        vig_rgb = [vig.vignet(lens.distortImage(target_rgb[i])) for i in range(num_views)]
        vig_rgb = np.stack(vig_rgb, 0)
        image_grid(vig_rgb, rows=2, cols=5, rgb=True)

        undist_rgb = [lens.distortImage(target_rgb[i]) for i in range(num_views)]

        brown_undist_rgb = np.stack(undist_rgb, 0)
        image_grid_vis(brown_undist_rgb, cam_coords.T[0, :], cam_coords.T[1, :], rows=2, cols=5, rgb=True)
        # plt.show()

        # image_grid(brown_undist_rgb, rows=2, cols=5, rgb=True)
        # plt.show()

        output_dir = "{}/{}_{:02d}_frames".format(out_fd, lens_name, int(lens_model["model_name"]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vig_output_dir = "{}/{}_{:02d}_vig_frames".format(out_fd, lens_name, int(lens_model["model_name"]))
        if not os.path.exists(vig_output_dir):
            os.makedirs(vig_output_dir)

        for i in range(len(target_cameras)):
            imageio.imwrite(
                '{}/frame_{:05d}.png'.format(vig_output_dir, i), np.uint8(vig_rgb[i] * 255))
            imageio.imwrite(
                '{}/frame_{:05d}.png'.format(output_dir, i), np.uint8(brown_undist_rgb[i] * 255))
