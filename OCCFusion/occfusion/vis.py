import os
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(2560, 1440))
        display.start()
        offscreen = True
    except:
        print("Failed to start virtual display.")

# try:
#     from mayavi import mlab
#     import mayavi
#     mlab.options.offscreen = offscreen
#     print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
# except:
#     print("No Mayavi installation found.")

import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid
import os



def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_occ(
    save_dir, 
    gaussian, 
    name,
    sem=False,
    cap=2,
    dataset='rellis-3d'
):
    # 설정
    if dataset == 'rellis-3d':
        voxel_size = [0.1] * 3
        vox_origin = [-25.6, -12.8, -1.6]
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    elif dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Occupancy 처리
    voxels = gaussian[0].cpu().to(torch.int)
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1

    np_voxels = voxels.numpy()
    # nonzero_indices = np.argwhere(np_voxels > 0)
    valid_mask = (np_voxels > 0) & (np_voxels != 255)
    nonzero_indices = np.argwhere(valid_mask)

    if nonzero_indices.shape[0] == 0:
        print("No occupied voxels.")
        return

    # 좌표 변환
    coords = nonzero_indices * voxel_size + np.array(vox_origin)
    labels = np_voxels[valid_mask]
    # labels = np_voxels[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

    # 색상 매핑
    if sem:
        cmap = np.array([
            [0, 0, 0],      # unknown
            [  0, 150, 245],       # car                  blue
            [255,   0,   0],       # pedestrian           red
            # [0, 255, 0],    # vegetation
            # [255, 255, 0],  # road
            [255, 0, 0],    # obstacle
        ]) / 255.0
        colors = cmap[np.clip(labels, 0, len(cmap) - 1)]
    else:
        norm = (labels - labels.min()) / (labels.max() - labels.min() + 1e-6)
        cmap = plt.get_cmap('jet')
        colors = cmap(norm)[:, :3]

    # 시각화 및 저장
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=25, azim=135)
    ax.view_init(elev=90, azim=-90)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=1)
    ax.set_axis_off()

    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved matplotlib 3D image: {save_path}")

def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [  0, 150, 245, 255],       # car                  blue
            [255,   0,   0, 255],       # pedestrian           red
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.float32) / 255.
    return colors

def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False):

    empty_label = 0
    sem_cmap = get_nuscenes_colormap()

    

    means = gaussian.means[0].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy() # g
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.3)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk
        
        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask


    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    # number of ellipsoids 
    ellipNumber = means.shape[0]

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=180)
    ax.set_box_aspect([np.ptp(means[:, 0]), np.ptp(means[:, 1]), np.ptp(means[:, 2])])

    # compute each and plot each ellipsoid iteratively
    border = np.array([
        [-25.0, -25.0, 0.0],
        [-25.0, 25.0, 0.0],
        [25.0, -25.0, 0.0],
        [25.0, 25.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in range(ellipNumber):
        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("auto")
    # plt.gca().set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_axis_off()    

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath)
    plt.cla()
    plt.clf()

def save_gaussian_and_occ(
    save_dir,
    gaussian,
    voxel_tensor,
    name,
    draw_gaussian_params=None,
    occ_params=None,
    dataset='rellis-3d'
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pyquaternion import Quaternion
    import numpy as np
    import torch
    import os

    draw_gaussian_params = draw_gaussian_params or {}
    occ_params = occ_params or {}

    scalar = draw_gaussian_params.get('scalar', 1.5)
    ignore_opa = draw_gaussian_params.get('ignore_opa', False)
    filter_zsize = draw_gaussian_params.get('filter_zsize', False)
    sem = occ_params.get('sem', False)
    cap = occ_params.get('cap', 2)

    sem_cmap = get_nuscenes_colormap()

    means = gaussian.means[0].detach().cpu().numpy()
    scales = gaussian.scales[0].detach().cpu().numpy()
    rotations = gaussian.rotations[0].detach().cpu().numpy()
    opas = gaussian.opacities[0]
    opas = torch.ones_like(gaussian.means[0][..., :1]) if opas.numel() == 0 else opas
    opas = opas.squeeze().detach().cpu().numpy()
    sems = gaussian.semantics[0].detach().cpu().numpy()
    pred = np.argmax(sems, axis=-1)
    empty_label = 0

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.3)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl, binr = zbins[idx], zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask &= zmsk
        z_small_mask = scales[:, 2] > 0.1
        mask &= z_small_mask

    means, scales, rotations, opas, pred = means[mask], scales[mask], rotations[mask], opas[mask], pred[mask]

    if dataset == 'rellis-3d':
        voxel_size = [0.1] * 3
        vox_origin = [-25.6, -12.8, -1.6]
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    elif dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    voxels = voxel_tensor[0].cpu().to(torch.int)
    if not sem:
        voxels[..., -cap:] = 0
        for z in range(voxels.shape[-1] - cap):
            mask_occ = (voxels > 0)[..., z]
            voxels[..., z][mask_occ] = z + 1

    np_voxels = voxels.numpy()
    valid_mask = (np_voxels > 0) & (np_voxels != 255)
    nonzero_indices = np.argwhere(valid_mask)
    coords_occ = nonzero_indices * voxel_size + np.array(vox_origin)
    labels_occ = np_voxels[valid_mask]
    if sem:
        colors_occ = sem_cmap[np.clip(labels_occ, 0, len(sem_cmap) - 1)]
    else:
        norm = (labels_occ - labels_occ.min()) / (labels_occ.max() - labels_occ.min() + 1e-6)
        cmap = plt.get_cmap('jet')
        colors_occ = cmap(norm)[:, :3]

    fig = plt.figure(figsize=(16, 8), dpi=300)

    # Gaussian subplot
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.view_init(elev=90, azim=180)
    ax0.set_title("Gaussian")
    ax0.set_axis_off()
    ax0.set_box_aspect([np.ptp(means[:, 0]), np.ptp(means[:, 1]), np.ptp(means[:, 2])])
    for i in range(means.shape[0]):
        center = means[i]
        radii = scales[i] * scalar
        rot_matrix = Quaternion(rotations[i]).rotation_matrix.T
        u, v = np.linspace(0.0, 2.0 * np.pi, 10), np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        xyz = np.stack([x, y, z], axis=-1)
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1) + center
        ax0.plot_surface(xyz[..., 1], -xyz[..., 0], xyz[..., 2], color=sem_cmap[pred[i]], linewidth=0, alpha=opas[i], shade=True)

    # Occupancy subplot
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.view_init(elev=90, azim=-90)
    ax1.set_title("Occupancy")
    ax1.set_axis_off()
    ax1.scatter(coords_occ[:, 0], coords_occ[:, 1], coords_occ[:, 2], c=colors_occ, s=1)

    os.makedirs(save_dir, exist_ok=True)
    base_name = f"{name}_combined"
    ext = ".png"
    i = 1
    while True:
        save_path = os.path.join(save_dir, f"{base_name}_{i}{ext}")
        if not os.path.exists(save_path):
            break
        i += 1
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved combined image: {save_path}")
