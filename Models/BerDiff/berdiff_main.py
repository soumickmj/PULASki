import copy
import functools

import torch
import torch.nn as nn

from torch.optim import AdamW

from .model.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad
)

from .model.basic_module import update_ema
from .diffusion.resample import LossAwareSampler, UniformSampler

from .diffusion.resample import create_named_schedule_sampler
from .scripts.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

INITIAL_LOG_LOSS_SCALE = 20.0

class BerDiff(nn.Module):
    def __init__(self, num_experts, input_channels=1, num_classes=1):
        super(BerDiff, self).__init__()

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
            image_size=128,
            img_channels=1,
            in_channels=input_channels, 
            out_channels=num_classes, 
            num_channels=128,
            num_res_blocks=2,
            num_heads=1,
            num_heads_upsample=-1,
            attention_resolutions="16",
            dropout=0.0,
            diffusion_steps=1000,
            timestep_respacing="ddimuni10",
            noise_schedule="cosine2",
            ltype="mix",
            mean_type="epsilon",
            rescale_timesteps=False,
            use_checkpoint="",
            use_scale_shift_norm=False,
            # learn_sigma=True,
            # class_cond=False,
            # rescale_learned_sigmas=False,
        ))
        self.model, self.diffusion = create_model_and_diffusion(**modelargs)
    
        self.schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], self.diffusion)
        
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(
            self.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.ema_rate = (
            [args["ema_rate"]]
            if isinstance(args["ema_rate"], float)
            else [float(x) for x in args["ema_rate"].split(",")]
        )

        self.ema_params = [
                copy.deepcopy(self.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def to(self, device):
        super(BerDiff, self).to(device)
        self.master_params = [param.to(device) for param in self.master_params]
        self.ema_params = [[paramset[e].to(device) for e in range(len(self.ema_rate))] for paramset in self.ema_params]
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    def run_step(self, batch, cond):
        loss, sample = self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimise_fp16()
        else:
            self.optimise_normal()        
        return loss, sample

    def run_inference(self, batch, time=1000, n_samples=-1):
        sample_fn = self.diffusion.ddim_sample_loop

        n_samples = self.num_experts if n_samples == -1 else n_samples

        outputs = []
        for _ in range(n_samples):
            sample = sample_fn(self.model, batch.shape, model_kwargs={"img": batch}, progress=True)
            outputs.append(sample.detach())

        return outputs
    
    def forward_backward(self, batch, cond):
        self.microbatch = self.microbatch if self.microbatch > 0 else batch.shape[0]
        zero_grad(self.model_params)

        total_loss = 0
        samples = []
        for i in range(0, batch.shape[0], self.microbatch):
            micro = {"img": batch[i : i + self.microbatch]}
            micro_cond = cond[i : i + self.microbatch]
            t, weights = self.schedule_sampler.sample(micro_cond.shape[0], batch.device)

            losses1 = self.diffusion.training_losses(self.model, micro_cond, t, model_kwargs=micro)
            losses = losses1[0]
            sample = losses1[1]

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            total_loss += loss.item()
            samples.append(sample.detach())

        return total_loss, torch.cat(samples, dim=0)
    
    def optimise_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            print(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
    
    def optimise_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        
    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr