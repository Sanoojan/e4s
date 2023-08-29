import torchvision.transforms as transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch.distributed as dist
import math
import numpy as np
# torch.autograd.set_detect_anomaly(True)
from collections import OrderedDict

from src.datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, FFHQDataset, FFHQ_MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, FFHQ_MASK_CONVERT_TF_DETAILED
from src.criteria.w_norm import WNormLoss
from src.criteria.id_loss import IDLoss
from src.criteria.face_parsing.face_parsing_loss import FaceParsingLoss
from src.criteria.lpips.lpips import LPIPS
from src.criteria.adv_loss import AdvDLoss,AdvGLoss,DR1Loss,GPathRegularizer
from src.criteria.style_loss import StyleLoss
from src.training.ranger import Ranger
from src.models.networks import Net3
from src.models.stylegan2.model import Generator,Discriminator
from src.utils import torch_utils



#Reenactment
from src.pretrained.face_vid2vid.driven_demo import init_facevid2vid_pretrained_model, drive_source_demo
from src.pretrained.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo

import sys
sys.path.append('/home/sb1/sanoojan/e4s/ddim')
import yaml
from ddim.runners.diffusion import Diffusion
from ddim.models.diffusion import Model
from ddim.models.ema import EMAHelper
from ddim.datasets import get_dataset, data_transform, inverse_data_transform
from ddim.functions.losses import loss_registry
import argparse
from PIL import Image
from torchvision.utils import save_image

from src.training.support import dict2namespace, get_beta_schedule

ACCUM = 0.5 ** (32 / (100 * 1000))  #  0.9977843871238888

config_file='celeba.yml'
with open(os.path.join("ddim/configs", config_file), "r") as f:
    config = yaml.safe_load(f)
new_config = dict2namespace(config)

normalize=transforms.Compose([NORMALIZE])




class Coach:
     
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0

        # distributed training
        if self.opts.dist_train:
            self.num_gpus = torch.cuda.device_count()
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])

            torch.cuda.set_device(self.rank % self.num_gpus)

            dist.init_process_group(
                backend='nccl',
                world_size=self.world_size,
                rank=self.rank,
            )
            self.device = torch.device("cuda", self.local_rank)
            
            
        else:
            self.rank=0 # dummy rank
            self.device = torch.device("cuda", 0)
        
        self.opts.device=self.device

        # ==== Initialize network ====
        #Diffusion
        # self.runner = Diffusion(None, new_config)
        self.model = Model(new_config)
        self.model= self.model.to(self.device)
        self.model.train()
        self.net_ema = EMAHelper(mu=new_config.model.ema_rate)
        self.net_ema.register(self.model)
        betas = get_beta_schedule(
            beta_schedule=new_config.diffusion.beta_schedule,
            beta_start=new_config.diffusion.beta_start,
            beta_end=new_config.diffusion.beta_end,
            num_diffusion_timesteps=new_config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        
        #Diffusion end
        
        #Reenactment
        face_vid2vid_cfg = "./pretrained_ckpts/facevid2vid/vox-256.yaml"
        face_vid2vid_ckpt = "./pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar"
        self.generator, self.kp_detector, self.he_estimator, self.estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)

        
        

        self.generator = self.generator.to(self.device).eval()
        self.kp_detector = self.kp_detector.to(self.device).eval()
        self.he_estimator = self.he_estimator.to(self.device).eval()

        gpen_model_params = {
        "base_dir": "./pretrained_ckpts/gpen/",  # a sub-folder named <weights> should exist
        "in_size": 512,
        "model": "GPEN-BFR-512", 
        "use_sr": True,
        "sr_model": "realesrnet",
        "sr_scale": 4,
        "channel_multiplier": 2,
        "narrow": 1,
    }
        with torch.set_grad_enabled(False):
            self.GPEN_model = init_gpen_pretrained_model(model_params = gpen_model_params)
        
        # print(self.device)
      

        # self.net_ema = Net3(self.opts).to(self.device).eval()
        # torch_utils.accumulate(self.net_ema,self.net, 0)
        
        if self.opts.train_D:
            self.D = Discriminator(self.opts.out_size).to(self.device).eval()
    
        # if self.opts.dist_train:
        #     # Wrap the model
        #     self.net = nn.parallel.DistributedDataParallel(self.net,
        #     device_ids=[self.local_rank], output_device=self.local_rank,
        #     broadcast_buffers=False, 
        #     find_unused_parameters=True
        #     )

        #     if self.opts.train_D: 
        #         self.D = nn.parallel.DistributedDataParallel(self.D,
        #         device_ids=[self.local_rank], output_device=self.local_rank,
        #         broadcast_buffers=False,
        #         find_unused_parameters=True
        #         )
            
        # resume
        if self.opts.checkpoint_path is not None:
            ckpt_dict = torch.load(self.opts.checkpoint_path)
            self.global_step = ckpt_dict["opts"]["max_steps"] + 1
            
            if self.opts.dist_train:    
                # self.net.module.latent_avg = ckpt_dict['latent_avg'].to(self.device)
                # self.net.load_state_dict(ckpt_dict["state_dict"])
                if self.opts.train_D:
                    self.D.module.load_state_dict(ckpt_dict["D_state_dict"]) 
            else:
                # self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
                # self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
                if self.opts.train_D:
                    self.D.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["D_state_dict"],prefix="module."))

            print("Resume training at step %d..."%self.global_step)            
        
        # load StyleGAN weights
        else:
            styleGAN2_ckpt = torch.load(self.opts.stylegan_weights)
            
            if self.opts.dist_train:
                # self.net.module.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
                if self.opts.train_D:
                    if self.opts.out_size == 1024:
                        self.D.module.load_state_dict(styleGAN2_ckpt['d'], strict=False) # 1024 resolution
                    else:
                        self.custom_load_D_state_dict(self.D.module, styleGAN2_ckpt['d'])  # load partial D
                # avg latent code
                # self.net.module.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)    
                # if self.opts.learn_in_w:
                #     self.net.module.latent_avg = self.net.module.latent_avg.repeat(1, 1)
                # else:
                #     self.net.module.latent_avg = self.net.module.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) -2 , 1)
            else:
                # self.net.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
                if self.opts.train_D:
                    if self.opts.out_size == 1024:
                        self.D.load_state_dict(styleGAN2_ckpt['d'], strict=False) # 1024 resolution
                    else:
                        self.custom_load_D_state_dict(self.D, styleGAN2_ckpt['d']) # load partial D
                # avg latent code
                # self.net.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)    
                # if self.opts.learn_in_w:
                #     self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
                # else:
                #     self.net.latent_avg = self.net.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) -2 , 1)
            
            print('Loading pretrained styleGAN2 weights!')

        # Estimate latent_avg via dense sampling if latent_avg is not available
        # if self.opts.dist_train:
        #     if self.net.module.latent_avg is None:
        #         self.net.module.latent_avg = self.net.module.G.mean_latent(int(1e5))[0].detach()
        # else:
        #     if self.net.latent_avg is None:
        #         self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.device).eval()
        if self.opts.face_parsing_lambda > 0:
            self.face_parsing_loss = FaceParsingLoss(self.opts).to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.style_lambda > 0:  # gram matrix loss
            self.style_loss = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3,8,15,22],
                                        normalize = self.opts.style_loss_norm==1,
                                        in_size=self.opts.out_size).to(self.device).eval()

        self.adv_d_loss=AdvDLoss().to(self.device).eval()
        self.adv_g_loss=AdvGLoss().to(self.device).eval()
        self.d_r1_reg_loss=DR1Loss().to(self.device).eval()
        self.g_path_reg_loss=GPathRegularizer().to(self.device).eval()
        
        # Initialize optimizer
        self.optimizer,self.optimizer_D = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        # breakpoint()
        if self.opts.dist_train:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,shuffle=True)
            self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.opts.batch_size,
                                            num_workers=int(self.opts.workers),
                                            drop_last=True,
                                            pin_memory=True,
                                            sampler=self.train_sampler)
        else:
            self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        # test set
        self.test_dataloader = DataLoader(self.test_dataset,
                                        batch_size=self.opts.test_batch_size,
                                        shuffle=False,
                                        num_workers=int(
                                            self.opts.test_workers),
                                        drop_last=False)

        # Initialize tensorborad logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        if self.rank==0:
            self.logger = SummaryWriter(logdir =log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps
    
    
    def custom_load_D_state_dict(self, module, state_dict):
        """Load partial StyleGAN discriminator weights
        Args:
            module (nn.Module): the module to be updated
            state_dict (): styleGAN weights, convs.0 corresponds to 1024 resolution
        """
        local_state = {k: v for k, v in module.named_parameters() if v is not None}

        # 
        del local_state["convs.0.0.weight"]
        del local_state["convs.0.1.bias"]

        idx_gap = int(math.log(1024, 2)) - int(math.log(self.opts.out_size, 2))
        
        new_state_dict = OrderedDict()
        for name, param in local_state.items():
            if name[:5]=="convs":
                layer_idx = int(name[6])
                name_in_pretrained = name[:6] + str(layer_idx + idx_gap) + name[7:]
                new_state_dict[name] = state_dict[name_in_pretrained]
            else:
                new_state_dict[name] = state_dict[name]  # FC
        
        module.load_state_dict(new_state_dict, strict=False)
        
                
    def configure_optimizers(self):
        self.params=list(filter(lambda p: p.requires_grad ,list(self.model.parameters())))
        self.params_D=list(filter(lambda p: p.requires_grad ,list(self.D.parameters()))) if self.opts.train_D else None
        
        d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1) if self.opts.d_reg_every >0 else 1
        
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(self.params, lr=self.opts.learning_rate)
            optimizer_D = torch.optim.Adam(self.params_D, lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
        else:
            optimizer = Ranger(self.params, lr=self.opts.learning_rate)
            optimizer_D = Ranger(self.params_D, lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
        return optimizer,optimizer_D

    def configure_datasets(self):
        if self.opts.dataset_name=="ffhq":    
            train_ds = FFHQDataset(dataset_root=self.opts.ffhq_dataset_root,
                                    img_transform=transforms.Compose(
                                        [TO_TENSOR]),  # Removed Normalize and aded after reenactment @sanoojan
                                    label_transform=transforms.Compose(
                                        [FFHQ_MASK_CONVERT_TF_DETAILED, TO_TENSOR]),  # FFHQ_MASK_CONVERT_TF
                                    fraction=self.opts.ds_frac,
                                    flip_p=self.opts.flip_p)
        else:
            # breakpoint()
            train_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="train",
                                    img_transform=transforms.Compose(
                                        [TO_TENSOR]),     # Removed Normalize and aded after reenactment @sanoojan
                                    label_transform=transforms.Compose(
                                        [MASK_CONVERT_TF_DETAILED, TO_TENSOR]),  # MASK_CONVERT_TF_DETAILED
                                    fraction=self.opts.ds_frac,
                                    flip_p=self.opts.flip_p)
        
        test_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="test",
                                  img_transform=transforms.Compose(
                                      [TO_TENSOR]),  #[TO_TENSOR, NORMALIZE]), # Removed Normalize and aded after reenactment @sanoojan
                                  label_transform=transforms.Compose(
                                      [MASK_CONVERT_TF_DETAILED, TO_TENSOR]),  # MASK_CONVERT_TF
                                  fraction=self.opts.ds_frac)
        print(f"Number of training samples: {len(train_ds)}")
        print(f"Number of test samples: {len(test_ds)}")
        return train_ds, test_ds

    # @torch.no_grad()
    def train(self):
        # self.net.train()
        self.model.train()
        if self.opts.train_D:
            self.D.train()
        # self.generator, self.kp_detector, self.he_estimator, self.estimate_jacobian
        torch_utils.requires_grad(self.generator, False)
        torch_utils.requires_grad(self.kp_detector, False)
        torch_utils.requires_grad(self.he_estimator, False)
        # torch_utils.requires_grad(self.estimate_jacobian, False)
        # torch_utils.requires_grad(self.GPEN_model, False)
        
        while self.global_step <= self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):   
                # Diffusion include source image and target image
                # 1. Order Source image and target images
                # 2. Use renactment on source image
                # 3. Replace features of target image with source image according to segmentation map
                # 4. Use diffusion to generate Swapped Image

                # 




                img, mask, mask_vis = batch
                # breakpoint()
                img = img.to(self.device).float()
                mask = (mask*255).long().to(self.device)
                # [bs,1,H,W] format mask to one-hot，i.e., [bs,#seg_cls,H,W]
                onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
                # breakpoint()

                #2. Reenactment
                B= img.size(0)
                src_img=F.interpolate(img[:B//2], size=(256, 256), mode='bilinear', align_corners=True)
                tar_img=F.interpolate(img[B//2:], size=(256, 256), mode='bilinear', align_corners=True)
        
                #  faceVid2Vid  input & output [0,1] range with RGB
                predictions = drive_source_demo(src_img, tar_img, self.generator, self.kp_detector, self.he_estimator, self.estimate_jacobian,for_train=True)
                
                # pred_img=torch.from_numpy(predictions[0]).permute(2, 0, 1).float() 
                # save_image(pred_img, 'pred_img.png')
                # breakpoint()
    
                predictions = (predictions*255).astype(np.uint8)
                # change predictions as numpy array
                # predictions = predictions.astype(np.uint8)

                # del generator, kp_detector, he_estimator
                # breakpoint()
                # GPEN input & output [0,255] range with BGR
                # drivens = [GPEN_demo(pred.flip(dims=[2]), self.GPEN_model, aligned=False,batched=False) for pred in predictions]
                drivens = [GPEN_demo(pred[:,:,::-1], self.GPEN_model, aligned=False,batched=False) for pred in predictions]
                driven_tensor=torch.from_numpy(np.array(drivens)[:,:,:,::-1]/255.0).permute(0, 3, 1, 2).float().cuda()
                img=torch.cat((driven_tensor,img[B//2:]),dim=0)
                #normalize img with transform
                img=normalize(img)



                # breakpoint()
                # D = Image.fromarray(drivens[0][:,:,::-1]) # to PIL.Image


                # ============ update D ===============
                if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                    torch_utils.requires_grad(self.model, False)
                    torch_utils.requires_grad(self.D, True)
                
                    # recon1, _, latent = self.net(img, onehot, return_latents=True)

                    n = img.size(0)
                    # img resize to 64
                    x = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=True)
                
                    # img = data_transform(self.config, img)
                    B, C, H, W = x.size()
                    e = torch.randn(B//2, C, H, W, device=self.device)
                    b = self.betas
                    n=n//2
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    
                    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                    a=torch.concat((a,a),dim=0)
                    # breakpoint()
                    x = x * a.sqrt() + e * (1.0 - a).sqrt()
                    recon1,latent=self.model.forward_swap(x, t.float())

                    
                    # print("recon1 shape: ", recon1.shape)
                    # if keepdim:
                    #     return (e - output).square().sum(dim=(1, 2, 3))
                    # else:
                    #     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    # loss = loss_registry[config.model.type](self.model, img, t, e, b)
                    # loss.backward()

                    #resize img to 1024
                    recon1 = F.interpolate(recon1, size=(1024, 1024), mode='bilinear', align_corners=True)

                    # print("recon1 shape: ", recon1.shape)
                    # print("latent shape: ", latent.shape)
                    fake_pred_1 = self.D(recon1)
                    real_pred = self.D(img)
                    
                    d_loss = self.adv_d_loss(real_pred,fake_pred_1)
                    
                    d_loss_dict={}
                    d_loss_dict["d_loss"]=float(d_loss)
                    d_loss_dict["real_score"]=float(real_pred.mean())
                    d_loss_dict["fake_score_1"]=float(fake_pred_1.mean())
                    
                    self.D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()
                    
                    r1_loss = torch.tensor(0.0, device=self.device)
                    # R1 regularization
                    if self.opts.d_reg_every!=-1 and batch_idx % self.opts.d_reg_every==0:
                        img.requires_grad=True
                        
                        real_pred = self.D(img)
                        r1_loss = self.d_r1_reg_loss(real_pred, img)
                        
                        self.D.zero_grad()
                        (self.opts.r1_lambda / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]).backward()
                        self.optimizer_D.step()
                        
                    d_loss_dict["r1_loss"] = r1_loss
                    # breakpoint()
                # ============ update G ===============
                # self.opts.train_G and self.opts.train_D should be both true or false
                if self.opts.train_G and self.opts.train_D:  
                    torch_utils.requires_grad(self.model, True)
                    # if self.opts.dist_train:
                    #     torch_utils.requires_grad(self.net.module.G.style, False)  # fix z-to-W mapping of original StyleGAN
                    #     if self.opts.remaining_layer_idx != 17:
                    #         torch_utils.requires_grad(self.net.module.G.convs[-(17-self.opts.remaining_layer_idx):],False)
                    #         torch_utils.requires_grad(self.net.module.G.to_rgbs[-(17-self.opts.remaining_layer_idx)//2 - 1:],False)
                    # else:
                    #     torch_utils.requires_grad(self.net.G.style, False)  # fix z-to-W mapping of original StyleGAN
                    #     if self.opts.remaining_layer_idx != 17:
                    #         torch_utils.requires_grad(self.net.G.convs[-(17-self.opts.remaining_layer_idx):],False)
                    #         torch_utils.requires_grad(self.net.G.to_rgbs[-(17-self.opts.remaining_layer_idx)//2 - 1:],False)
                    
                    
                
                # only training Encoder
                # elif not self.opts.train_G and not self.opts.train_D:  
                #     if self.opts.dist_train:
                #         torch_utils.requires_grad(self.net.module.G, False)
                #     else:
                #         torch_utils.requires_grad(self.net.G, False)
                        
                if self.opts.train_D:
                    torch_utils.requires_grad(self.D, False)  
                
                # recon1, _, latent = self.net(img, onehot, return_latents=True)
                
                #Diffusion start
                n = img.size(0)
                # img resize to 64
                x = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=True)
            
                # img = data_transform(self.config, img)
                e = torch.randn_like(x)
                b = self.betas
                n=n//2
                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                a=torch.concat((a,a),dim=0)
                # breakpoint()
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                recon1,latent=self.model.forward_swap(x, t.float())
                # print("recon1 shape: ", recon1.shape)
                # if keepdim:
                #     return (e - output).square().sum(dim=(1, 2, 3))
                # else:
                #     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
                # loss = loss_registry[config.model.type](self.model, img, t, e, b)
                # loss.backward()

                #resize img to 1024
                recon1 = F.interpolate(recon1, size=(1024, 1024), mode='bilinear', align_corners=True)

                # Diffusion end



                g_loss = torch.tensor(0.0, device=self.device)                
                if self.opts.train_D:
                    fake_pred_1 = self.D(recon1)
                    
                    g_loss = self.adv_g_loss(fake_pred_1)
                
                loss_, loss_dict, id_logs = self.calc_loss(img, recon1, mask, latent)
                loss_dict["g_loss"] = float(g_loss)
                
                overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
                loss_dict["loss"] = float(overall_loss)
                
                
              
                self.model.zero_grad()
                overall_loss.backward()
                self.optimizer.step()
                
                # Logging related
                if self.rank==0 and (self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0)):
                    
                    imgs = self.parse_images(onehot, img, recon1)
                    self.log_images('images/train/faces', imgs1_data=imgs, subscript=None)

                if self.rank==0 and (self.global_step % self.opts.board_interval == 0):
                    self.print_metrics(loss_dict, prefix='train')
                    if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                        self.print_metrics(d_loss_dict, prefix='train')
                        
                    self.log_metrics(loss_dict, prefix='train')
                    if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                        self.log_metrics(d_loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                # save model
                if self.rank==0 and (self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
                    val_loss_dict = self.validate()
                if self.rank==0 and (val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss)):
                    self.best_val_loss = val_loss_dict['loss']
                    self.checkpoint_me(val_loss_dict, is_best=True)

                if self.rank==0 and (self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                self.global_step += 1
                if self.global_step == 100000:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.opts.learning_rate * 0.1
            
                # ema
                # if self.opts.dist_train:
                #     torch_utils.accumulate(self.net_ema, self.net.module, ACCUM)
                # else:
                #     torch_utils.accumulate(self.net_ema, self.net, ACCUM)
                
        if self.rank==0:
            print('OMG, finished training!')

    def calc_loss(self, img, recon1, mask, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        B=img.shape[0]
        no_src_imgs=B//2
        # For swapping
        source_img = img[:no_src_imgs]
        target_img = img[no_src_imgs:]
        source_mask = mask[:no_src_imgs]
        target_mask = mask[no_src_imgs:]
        ###########
        
        if self.opts.face_parsing_lambda > 0:
            loss_face_parsing_1, face_parsing_sim_improvement_1 = self.face_parsing_loss(recon1, target_img)
            
            loss_dict['loss_face_parsing'] = float(loss_face_parsing_1)
            loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement_1)
            loss += loss_face_parsing_1 * self.opts.face_parsing_lambda
            
        if self.opts.id_lambda > 0:
            loss_id_1, sim_improvement_1, id_logs_1 = self.id_loss(recon1, source_img)
                   
            loss_dict['loss_id'] = float(loss_id_1)
            loss_dict['id_improve'] = float(sim_improvement_1)
            loss += loss_id_1 * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            # loss_l2_1 = F.mse_loss(recon1, img)       @check this @sanoojan
            
            # loss_dict['loss_l2'] = float(loss_l2_1)
            # loss += loss_l2_1 * self.opts.l2_lambda
            loss_dict['loss_l2'] = 0.0
            loss += 0.0

        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(recon1,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(target_img,(1024//2**i,1024//2**i))
                )
               
                loss_lpips += loss_lpips_1
            
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.w_norm_lambda > 0:
            # if self.opts.dist_train:
            #     loss_w_norm = self.w_norm_loss(latent, self.net.module.latent_avg)
            # else:
            #     loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
                
            # loss_dict['loss_w_norm'] = float(loss_w_norm)
            # loss += loss_w_norm * self.opts.w_norm_lambda  
            #check here @ sanoojan
            loss+=0



        if self.opts.style_lambda > 0:  # gram matrix loss
            loss_style_1 = self.style_loss(recon1, target_img, mask_x = (target_mask==3).float(), mask_x_hat = (target_mask==3).float())
            
            loss_dict['loss_style'] = float(loss_style_1)
            loss += loss_style_1 * self.opts.style_lambda
            
        loss_dict['loss'] = float(loss)
        return (loss, loss_dict, id_logs_1) if self.opts.id_lambda > 0 else (loss, loss_dict, None)

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(
                f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_images(self, mask, img, recon1, display_count=2):
        B=img.shape[0]
        no_src_imgs=B//2
        source_img = img[:no_src_imgs]
        target_img = img[no_src_imgs:]
        source_mask = mask[:no_src_imgs]
        target_mask = mask[no_src_imgs:]
        im_data = []

        display_count=min(display_count,len(source_img))
        for i in range(display_count):
            cur_im_data = {
                'Source_face': torch_utils.tensor2im(source_img[i]),
                'Target_face': torch_utils.tensor2im(target_img[i]),
                'Source_mask': torch_utils.tensor2map(source_mask[i]),
                'Target_mask': torch_utils.tensor2map(target_mask[i]),
                'recon_styleCode': torch_utils.tensor2im(recon1[i]),
            }
            im_data.append(cur_im_data)

        return im_data

    def log_images(self, name, imgs1_data, subscript=None, log_latest=False):
        fig = torch_utils.vis_faces(imgs1_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.logdir , name,
                                f'{subscript}_{step:06d}.jpg')
        else:
            path = os.path.join(self.logger.logdir , name, f'{step:06d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.model.state_dict(),
            'opts': vars(self.opts),
            'state_dict_ema': self.net_ema.state_dict(),
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        # if self.opts.start_from_latent_avg:
        #     save_dict['latent_avg'] = self.net.module.latent_avg if self.opts.dist_train else self.net.latent_avg
            
        if self.opts.train_D:
            save_dict['D_state_dict'] = self.D.state_dict()
        return save_dict

    def validate(self):
        # show_images=False
        # Logging related 
        if self.global_step % (4*self.opts.val_interval) == 0 or self.global_step == self.opts.max_steps:
            show_images=True
        else:
            show_images=False
            
     
        self.model.eval()
        if self.opts.train_D:
            self.D.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            
            img, mask, mask_vis = batch
            
            with torch.no_grad():
                img = img.to(self.device).float()
                mask = (mask*255).long().to(self.device)
                # [bs,1,H,W] format mask to one-hot, i.e., [bs,#seg_cls,H,W]
                onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
                
                # recon1, _, latent = self.net(img, onehot, return_latents=True)    

                # Diffustion start
                n = img.size(0)
                # img resize to 64
                x = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=True)
            
                # img = data_transform(self.config, img)
                e = torch.randn_like(x)
                b = self.betas
                n=n//2
                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                a=torch.concat((a,a),dim=0)
                # breakpoint()
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                recon1,latent=self.model.forward_swap(x, t.float())
                # print("recon1 shape: ", recon1.shape)
                # if keepdim:
                #     return (e - output).square().sum(dim=(1, 2, 3))
                # else:
                #     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
                # loss = loss_registry[config.model.type](self.model, img, t, e, b)
                # loss.backward()

                #resize img to 1024
                recon1 = F.interpolate(recon1, size=(1024, 1024), mode='bilinear', align_corners=True)

                # Diffusion end


                g_loss = torch.tensor(0.0, device=self.device)                
                if self.opts.train_D:
                    fake_pred_1 = self.D(recon1)
                    g_loss = self.adv_g_loss(fake_pred_1)
                
                loss_, loss_dict, id_logs = self.calc_loss(img, recon1, mask, latent)
                loss_dict["g_loss"] = float(g_loss)
                
                overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
                loss_dict["loss"] = float(overall_loss)
                
            agg_loss_dict.append(loss_dict)
            
            if show_images:
                imgs = self.parse_images(onehot, img, recon1)
                self.log_images('images/test/faces', imgs1_data=imgs, subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.model.train()
                if self.opts.train_D:
                    self.D.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = torch_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.model.train()
        if self.opts.train_D:
            self.D.train()
        return loss_dict
