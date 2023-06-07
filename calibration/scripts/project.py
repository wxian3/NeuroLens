import torch
from ..camera import Camera
from ..networks import LensNet


def cli():
    RT = torch.eye(3, 4)
    RT[0, 3] = 5.0
    cam = Camera(torch.eye(3), lensnet=LensNet(), RT=RT)
    # import pdb
    # pdb.set_trace()
    pt = torch.tensor([[1.0, 4.0, 5.0]], dtype=torch.float32)
    proj = cam.project_points(pt)
    print(f"proj: {proj}")
    unproj = cam.unproject_points(torch.tensor([[proj[0, 0], proj[0, 1], 5.0]]))
    print(f"unproj: {unproj}")
    ray = cam.get_rays_view(torch.tensor([[0.2, 0.8]]))
    print(f"ray: {ray}")
