import torch
torch.set_num_threads(4) # fix to prevent throttling on single GPU processes

import os
from tqdm.auto import tqdm
from opt import config_parser

import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# renderer = OctreeRender_trilinear_fast
renderer = OctreeRender_trilinear_fast_AUDIO

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    assert args.ckpt is not None, "Network checkpoint is None"
    assert args.audio_checkpoint is not None, "Audio network checkpoint is None"
    assert args.audio_attention_checkpoint is not None, "Audio attention network checkpoint is None"

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(audio_dimension=args.audio_dimension, window_size=args.window_size, **kwargs)
    tensorf.load(ckpt)

    ckpt = torch.load(args.audio_checkpoint, map_location=device)
    audio_network = AudioNet(args.audio_dimension, args.window_size).to(device)
    audio_network.load(ckpt)

    ckpt = torch.load(args.audio_attention_checkpoint, map_location=device)
    audio_attention_network = AudioAttNet().to(device)
    audio_attention_network.load(ckpt)

    checkpoint_iteration_count = args.audio_checkpoint.split("_")[-2]

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, evaluation_mode=True)
        white_bg = train_dataset.white_bg
        PSNRs_train = evaluation(train_dataset, tensorf, audio_network, args, renderer, f'{logfolder}/imgs_train_all_{checkpoint_iteration_count}', prtx=f"{checkpoint_iteration_count}_",
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'NOT FULL EVALUATION SET ======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        test_dataset = dataset(args.datadir, split='val', downsample=args.downsample_train, is_stack=True, evaluation_mode=True)
        white_bg = test_dataset.white_bg
        PSNRs_test = evaluation(test_dataset, tensorf, audio_network, args, renderer, f'{logfolder}/imgs_test_all_{checkpoint_iteration_count}', prtx=f"{checkpoint_iteration_count}_",
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'NOT FULL EVALUATION SET ======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def accumulate_gradient_norm(model_module):
    total_norm = 0
    parameters = [p for p in model_module.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, nearfar=args.nearfar, ray_sample_rate=args.batch_size, frame_face_mouth_sampling_ratios=args.frame_face_mouth_sampling_ratios)
    test_dataset = dataset(args.datadir, split='val', downsample=args.downsample_train, is_stack=True, nearfar=args.nearfar)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        assert args.audio_checkpoint is not None, "Audio network checkpoint is None"
        assert args.audio_attention_checkpoint is not None, "Audio attention network checkpoint is None"

        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)

        ckpt = torch.load(args.audio_checkpoint, map_location=device)
        audio_network = AudioNet(args.audio_dimension, args.window_size).to(device)
        audio_network.load(ckpt)

        ckpt = torch.load(args.audio_attention_checkpoint, map_location=device)
        audio_attention_network = AudioAttNet().to(device)
        audio_attention_network.load(ckpt)

        # TODO load audio network checkpoints
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device, audio_dimension=args.audio_dimension, window_size=args.window_size,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)
        audio_network = AudioNet(args.audio_dimension, args.window_size).to(device)
        audio_attention_network = AudioAttNet().to(device)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    optimizer_audio = torch.optim.Adam(params=list(audio_network.parameters()), lr=args.lr_basis, betas=(0.9, 0.999))
    optimizer_audio_attention = torch.optim.Adam(params=list(audio_attention_network.parameters()), lr=args.lr_basis, betas=(0.9, 0.999))

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    # allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    # if not args.ndc_ray:
    #     allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    # trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        # ray_idx = trainingSampler.nextids()
        # dataset_index = iteration % len(train_dataset)
        dataset_item = train_dataset[iteration]

        # rays_train, rgb_train, aud_train = train_dataset[dataset_index]
        rays_train, rgb_train, audio_train, audio_window = dataset_item['rays'].to(device), dataset_item['rgbs'].to(device), dataset_item['auds'].to(device), dataset_item['auds-win'].to(device)
        # rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)


        if iteration < args.no_smoothing_training_period:
            audio_features = audio_network(audio_train)
            # audio_features = audio_features.broadcast_to([rays_train.shape[0], audio_features.shape[0]])
        else:
            audio_features_interim = audio_network(audio_window)
            audio_features = audio_attention_network(audio_features_interim)
            # audio_features = audio_features.broadcast_to([rays_train.shape[0], audio_features.shape[0]])

        # print(audio_features.shape)
        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, audio_features, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)


        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        optimizer_audio.zero_grad()
        optimizer_audio_attention.zero_grad()
        total_loss.backward()
        # log audio network gradient norm
        audio_network_gradient_norm = accumulate_gradient_norm(audio_network)
        summary_writer.add_scalar('train/audio_net_norm', audio_network_gradient_norm, global_step=iteration)
        audio_attention_network_gradient_norm = accumulate_gradient_norm(audio_attention_network)
        summary_writer.add_scalar('train/audio_net_norm', audio_attention_network_gradient_norm, global_step=iteration)
        
        optimizer.step()
        optimizer_audio.step()
        optimizer_audio_attention.step()
        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,tensorf, audio_network, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration % args.save_every == args.save_every - 1:
            tensorf.save(f'{logfolder}/{args.expname}_{iteration}.th')
            audio_network.save(f'{logfolder}/{args.expname}_{iteration}_audio.th')
            audio_attention_network.save(f'{logfolder}/{args.expname}_{iteration}_audio_attention.th')

        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            # if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                # allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                # trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, audio_network, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, audio_network, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    args.frame_face_mouth_sampling_ratios = [float(item) for item in args.frame_face_mouth_sampling_ratios]
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path or args.render_train):
        render_test(args)
    else:
        reconstruction(args)