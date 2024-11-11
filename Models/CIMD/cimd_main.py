import copy
import functools

import torch
import torch.nn as nn

from .guided_diffusion.resample import create_named_schedule_sampler
from .guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from .guided_diffusion.fp16_util import MixedPrecisionTrainer
from .guided_diffusion.nn import update_ema
from .guided_diffusion.resample import LossAwareSampler, UniformSampler
from torch.optim import AdamW

class CIMD(nn.Module):
    def __init__(self, num_experts, input_channels=1, num_classes=1):
        super(CIMD, self).__init__()

        self.num_experts = num_experts
        args = dict(
            schedule_sampler="uniform",
            lr=1e-4,
            weight_decay=0.0,
            lr_anneal_steps=0,
            batch_size=1,
            microbatch=-1,  # -1 disables microbatches
            ema_rate="0.9999",  # comma-separated list of EMA values
            log_interval=100,
            save_interval=5000,
            resume_checkpoint='',#'"./results/pretrainedmodel.pt",
            use_fp16=True,
            fp16_scale_growth=1e-3,
            maxt=1000,
            classifier=None,
            use_ddim=False,
            clip_denoised=True
        )
        self.__dict__.update(args)
        
        modelargs = model_and_diffusion_defaults()
        modelargs.update(dict(
            use_fp16=args['use_fp16'],
            image_size=128,
            in_channels=input_channels+1,
            out_channels=num_classes,
            num_channels=128,
            num_res_blocks=2,
            num_heads=1,
            latent_dim = 6,
            learn_sigma=True,
            class_cond=False,
            use_scale_shift_norm=False,
            attention_resolutions="16",
            diffusion_steps=1000,
            noise_schedule="linear",
            rescale_learned_sigmas=False,
            rescale_timesteps=False
        ))
        self.model, self.diffusion, self.prior, self.posterior = create_model_and_diffusion(**modelargs)
    
        self.schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], self.diffusion,  maxt=args['maxt'])

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=args['use_fp16'],
            fp16_scale_growth=args['fp16_scale_growth'],
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.ema_rate = (
            [args["ema_rate"]]
            if isinstance(args["ema_rate"], float)
            else [float(x) for x in args["ema_rate"].split(",")]
        )

        self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def run_step(self, batch, cond):
        batch=torch.cat((batch, cond), dim=1)

        cond={}
        lossseg, losscls, lossrec, loss, sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        return lossseg, losscls, lossrec, loss, sample

    def run_inference(self, batch, time=1000, n_samples=-1):
        c = torch.randn_like(batch[:, :1, ...], device=batch.device, dtype=batch.dtype)
        img = torch.cat((batch, c), dim=1)     #add a noise channel$

        n_samples = self.num_experts if n_samples == -1 else n_samples

        outputs = []
        for i in range(n_samples):
            model_kwargs = {}
            sample_fn = (
                self.diffusion.p_sample_loop_known if not self.use_ddim else self.diffusion.ddim_sample_loop_known
            )
            
            sample, x_noisy, org = sample_fn(
                self.model,
                batch.shape, img,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
                time=time,
            )            
            
            outputs.append(sample.detach())

        return outputs

    def forward_backward(self, batch, cond):
        self.microbatch = self.microbatch if self.microbatch > 0 else batch.shape[0]
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in cond.items()
            }
            
            t, weights = self.schedule_sampler.sample(micro.shape[0], micro.device)

            losses1 = self.diffusion.training_losses_segmentation(self.model,
                                                                    self.classifier,
                                                                    self.prior,
                                                                    self.posterior,
                                                                    micro,
                                                                    t,
                                                                    model_kwargs=micro_cond, 
                )
            losses = losses1[0]
            sample = losses1[1]

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            lossseg = (losses["mse"] * weights).mean().detach()
            if "vb" in losses:
                losscls = (losses["vb"] * weights).mean().detach()
            else:
                losscls =loss*0
            lossrec =loss*0
            
            self.mp_trainer.backward(loss)
            return lossseg.detach(), losscls.detach(), lossrec.detach(), loss.detach(), sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr