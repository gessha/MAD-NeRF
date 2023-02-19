import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
import random
import json
from pathlib import Path

def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R


def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)

class TanksTempleDataset(Dataset):
    """NSVF Generic Dataset."""
    def __init__(self, datadir, split='train', downsample=1.0, wh=[1920,1080], is_stack=False):
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.img_wh = (int(wh[0]/downsample),int(wh[1]/downsample))
        self.define_transforms()

        self.white_bg = True
        self.near_far = [0.01,6.0]
        self.scene_bbox = torch.from_numpy(np.loadtxt(f'{self.root_dir}/bbox.txt')).float()[:6].view(2,3)*1.2

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
    
    def bbox2corners(self):
        corners = self.scene_bbox.unsqueeze(0).repeat(4,1,1)
        for i in range(3):
            corners[i,[0,1],i] = corners[i,[1,0],i] 
        return corners.view(-1,3)
        
        
    def read_meta(self):

        self.intrinsics = np.loadtxt(os.path.join(self.root_dir, "intrinsics.txt"))
        self.intrinsics[:2] *= (np.array(self.img_wh)/np.array([1920,1080])).reshape(2,1)
        pose_files = sorted(os.listdir(os.path.join(self.root_dir, 'pose')))
        img_files  = sorted(os.listdir(os.path.join(self.root_dir, 'rgb')))

        if self.split == 'train':
            pose_files = [x for x in pose_files if x.startswith('0_')]
            img_files = [x for x in img_files if x.startswith('0_')]
        elif self.split == 'val':
            pose_files = [x for x in pose_files if x.startswith('1_')]
            img_files = [x for x in img_files if x.startswith('1_')]
        elif self.split == 'test':
            test_pose_files = [x for x in pose_files if x.startswith('2_')]
            test_img_files = [x for x in img_files if x.startswith('2_')]
            if len(test_pose_files) == 0:
                test_pose_files = [x for x in pose_files if x.startswith('1_')]
                test_img_files = [x for x in img_files if x.startswith('1_')]
            pose_files = test_pose_files
            img_files = test_img_files

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.intrinsics[0,0],self.intrinsics[1,1]], center=self.intrinsics[:2,2])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)


        
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        assert len(img_files) == len(pose_files)
        for img_fname, pose_fname in tqdm(zip(img_files, pose_files), desc=f'Loading data {self.split} ({len(img_files)})'):
            image_path = os.path.join(self.root_dir, 'rgb', img_fname)
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs.append(img)
            

            c2w = np.loadtxt(os.path.join(self.root_dir, 'pose', pose_fname))# @ cam_trans
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)  # C2W
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 8)

        self.poses = torch.stack(self.poses)

        center = torch.mean(self.scene_bbox, dim=0)
        radius = torch.norm(self.scene_bbox[1]-center)*1.2
        up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
        pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
        self.render_path = gen_path(pos_gen, up=up,frames=200)
        self.render_path[:, :3, 3] += center



        if 'train' == self.split:
            if self.is_stack:
                self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3) 
            else:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

 
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = torch.from_numpy(self.intrinsics[:3,:3]).unsqueeze(0).float() @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self, points):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]

            sample = {'rays': rays,
                      'rgbs': img}
        return sample

class TanksTempleDataset_AUDIO(Dataset):
    """NSVF Generic Dataset."""
    def __init__(self, datadir, split='train', downsample=1.0, wh=[450,450], is_stack=False, nearfar=None, seed=1337, adnerf=True, testskip=1, per_image_loading=True, ray_sample_rate=4096, frame_face_mouth_sampling_ratios=[1.00, 0.00, 0.00], smo_size=8, evaluation_mode=False):
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.img_wh = (int(wh[0]/downsample),int(wh[1]/downsample))
        self.define_transforms()
        self.adnerf = adnerf
        self.testskip = testskip
        self.per_image_loading = per_image_loading
        self.ray_sample_rate = ray_sample_rate
        self.frame_face_mouth_sampling_ratios = frame_face_mouth_sampling_ratios
        self.smo_size = smo_size
        self.evaluation = evaluation_mode
        random.seed(seed)
        
        self.white_bg = True
        if nearfar:
            self.near_far = nearfar
        else:
            self.near_far = [0.01,6.0]
        self.scene_bbox = torch.from_numpy(np.loadtxt(f'{self.root_dir}/bbox.txt')).float()[:6].view(2,3) # *1.2 # COMMENTED OUT, DONE IN REGULAR TANKS TEMPLE DATASET
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        if self.adnerf:
            self.read_meta_ADNERF()
        else:
            self.read_meta()
        self.define_proj_mat()
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
    
    def bbox2corners(self):
        corners = self.scene_bbox.unsqueeze(0).repeat(4,1,1)
        for i in range(3):
            corners[i,[0,1],i] = corners[i,[1,0],i] 
        return corners.view(-1,3)
        
    ###
    # AD-NeRF modification
    ### 
    def read_meta_ADNERF(self):
        
        if self.split == "train":
            if not self.evaluation:
                data = json.loads((Path(self.root_dir) / "transforms_train_extended.json").read_text())
            else:
                data = json.loads((Path(self.root_dir) / "transforms_train_extended_eval.json").read_text())
        if self.split == "val":
            if not self.evaluation:
                data = json.loads((Path(self.root_dir) / "transforms_val_extended.json").read_text())
            else:
                data = json.loads((Path(self.root_dir) / "transforms_val_extended_eval.json").read_text())
        
        aud_features = np.load(Path(self.root_dir) / 'aud.npy')
        
        self.intrinsics = np.zeros((4,4))
        self.intrinsics[0, 0] = data['focal_len']
        self.intrinsics[1, 1] = data['focal_len']
        self.intrinsics[0, 2] = data["cx"] 
        self.intrinsics[1, 2] = data["cy"]
        self.intrinsics[2, 2] = 1
        self.intrinsics[3, 3] = 1
        
         # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.intrinsics[0,0],self.intrinsics[1,1]], center=self.intrinsics[:2,2])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        
        # img_files = sorted((Path(self.root_dir) / "head_imgs").glob("*.jpg"))
        # assert len(img_files) == len(data['frames']), f"{len(img_files)} != {len(data['frames'])}"
        
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_face_rects = []
        self.all_mouth_rects = []
        self.auds = []
        
        if self.split == 'train' or self.testskip == 0:
            skip = 1
        else:
            skip = self.testskip
        
        for frame in tqdm(data['frames'][::skip], desc=f'Loading data {self.split} ({len(data["frames"][::skip])})'):
            # load image
            image_path = Path(self.root_dir) / "head_imgs" / f"{frame['img_id']}.jpg"
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs.append(img)
            # print("Loaded images")
            
            # add transformation
            c2w = np.array(frame['transform_matrix'])# @ cam_trans
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)
            # print("Loaded pose")
            
            # add the rays
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 8)
            # print("Loaded rays")
            
            # add audio
            self.auds.append(self.transform(aud_features[min(frame['aud_id'], aud_features.shape[0]-1)]))
            # print("Loaded audio")

            self.all_face_rects.append(np.array(frame['face_rect'], dtype=np.int32))
            self.all_mouth_rects.append(np.array(frame['mouth_rect'], dtype=np.int32))

        self.rays_per_image = len(rays_o)
        self.poses = torch.stack(self.poses)
        self.all_face_rects = np.stack(self.all_face_rects, 0)
        self.all_mouth_rects = np.stack(self.all_mouth_rects, 0)

        # center = torch.mean(self.scene_bbox, dim=0)
        # radius = torch.norm(self.scene_bbox[1]-center)*1.2
        # up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
        # pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
        # self.render_path = gen_path(pos_gen, up=up,frames=200)
        # self.render_path[:, :3, 3] += center        
        
        ### NEW WAY of stacking data
        self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3)
        
        ###
        # AD-NeRF modification END
        ### 
    def get_random_indeces_within_rect(self, rect, N_rays_in, N_rays_out):
        H = self.img_wh[0]
        W = self.img_wh[1]

        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

        rect_inds = (coords[:, 0] >= rect[0]) & (
            coords[:, 0] <= rect[0] + rect[2]) & (
                coords[:, 1] >= rect[1]) & (
                    coords[:, 1] <= rect[1] + rect[3])
        coords_rect = coords[rect_inds]
        coords_norect = coords[~rect_inds]

        rect_num = N_rays_in
        norect_num = N_rays_out

        select_inds_rect = np.random.choice(
            coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
        # (N_rand, 2)
        select_coords_rect = coords_rect[select_inds_rect].long()
        select_inds_norect = np.random.choice(
            coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
        # (N_rand, 2)
        select_coords_norect = coords_norect[select_inds_norect].long(
        )
        select_coords = torch.cat(
            (select_coords_rect, select_coords_norect), dim=0)
        
        return select_coords
 
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = torch.from_numpy(self.intrinsics[:3,:3]).unsqueeze(0).float() @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self, points):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        if self.split == 'train':
            if self.per_image_loading:
                return len(self.all_rgbs)
            else:
                return len(self.all_rays)
        else:
            if self.per_image_loading:
                return len(self.all_rgbs)
            else:
                return len(self.all_rgbs)        
    
    def __getitem__(self, idx):

        if self.split == 'train':
            if not self.evaluation:
                index = random.randint(0, len(self.all_rgbs)-1)
                rays = self.all_rays[index]
                img = self.all_rgbs[index]
                
                auds = self.auds[index].float()
                
                ### get audio window ###
                smo_half_win = int(self.smo_size / 2)
                left_i = index - smo_half_win
                right_i = index + smo_half_win
                pad_left, pad_right = 0, 0
                dataset_size = len(self.all_rgbs)
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > dataset_size:
                    pad_right = right_i-dataset_size
                    right_i = dataset_size
                auds_win = self.auds[left_i:right_i]
                auds_win = torch.concatenate(auds_win).float()
                if pad_left > 0:
                    auds_win = torch.cat((torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat((auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                # print(f"Left: {left_i}, Right: {right_i}, Auds_win.shape: {len(auds_win)}")
                ### get audio window END ###

                face_rect = self.all_face_rects[index]
                mouth_rect = self.all_mouth_rects[index]
                frame_rate, face_rate, mouth_rate = self.frame_face_mouth_sampling_ratios
                batch_size = self.ray_sample_rate
                mouth_ray_count = int(batch_size * mouth_rate)
                face_ray_count = int(batch_size * face_rate)
                frame_ray_count = batch_size - face_ray_count - mouth_ray_count

                mouth_rays_coords = self.get_random_indeces_within_rect(mouth_rect, mouth_ray_count, N_rays_out=0) # select only rays and rgbs within mouth bbox
                face_rays_coords = self.get_random_indeces_within_rect(face_rect, face_ray_count, N_rays_out=0) # select only rays and rgbs within face bbox
                frame_rays_coords = self.get_random_indeces_within_rect(face_rect, N_rays_in=0, N_rays_out=frame_ray_count) # select rays and rgbs OUTSIDE face box

                # print(f"{mouth_ray_count}, {face_ray_count}, {frame_ray_count}")
                # print(f"Mouth coord shape: {mouth_rays_coords.shape} Face coord shape: {face_rays_coords.shape}, Frame coord shape: {frame_rays_coords.shape}")
                
                if face_rate != 0 or mouth_rate != 0:
                    # print("face_rate != 0 or mouth_rate != 0:")
                    rays = rays.reshape(self.img_wh[0], self.img_wh[1], 6)
                    rgbs = img
                    
                    mouth_rays = rays[mouth_rays_coords[:, 0], mouth_rays_coords[:, 1]]
                    mouth_rgbs = rgbs[mouth_rays_coords[:, 0], mouth_rays_coords[:, 1]]

                    face_rays = rays[face_rays_coords[:, 0], face_rays_coords[:, 1]]
                    face_rgbs = rgbs[face_rays_coords[:, 0], face_rays_coords[:, 1]]

                    frame_rays = rays[frame_rays_coords[:, 0], frame_rays_coords[:, 1]]
                    frame_rgbs = rgbs[frame_rays_coords[:, 0], frame_rays_coords[:, 1]]
                    
                    batch_rays = torch.concatenate([mouth_rays, face_rays, frame_rays])
                    batch_rgbs = torch.concatenate([mouth_rgbs, face_rgbs, frame_rgbs])

                else:
                    # print("NOT face_rate != 0 or mouth_rate != 0:")
                    indeces = random.sample(range(rays.shape[0]), self.ray_sample_rate)
                    indeces = torch.Tensor(indeces).to(torch.int32)

                    batch_rays = torch.index_select(rays, 0, indeces) 
                    batch_rgbs = img.reshape(rays.shape[0], 3)
                    batch_rgbs = torch.index_select(batch_rgbs, 0, indeces)

                sample = {
                    'index': index,
                    'rays': batch_rays,
                    "rgbs": batch_rgbs,
                    "auds": auds,
                    "auds-win": auds_win,
                    "face-rect": face_rect,
                    "mouth-rect": mouth_rect, 
                }
            else:
                # create data for each image separately
                rays = self.all_rays[idx]
                img = self.all_rgbs[idx]
                auds = self.auds[idx].float()
                # rect = self.all_face_rects[idx] not needed during testing

                sample = {
                    'rays': rays,
                    'rgbs': img,
                    'auds': auds,
                    # "face-rect": rect, # not needed during testing
                }
                return sample

        else:  
            # create data for each image separately
            rays = self.all_rays[idx]
            img = self.all_rgbs[idx]
            auds = self.auds[idx].float()
            # rect = self.all_face_rects[idx] not needed during testing

            sample = {
                'rays': rays,
                'rgbs': img,
                'auds': auds,
                # "face-rect": rect, # not needed during testing
            }
        return sample