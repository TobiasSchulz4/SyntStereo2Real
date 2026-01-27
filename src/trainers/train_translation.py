"""Training loop for SyntStereo2Real translation GAN.

Implements edge-aware generator with warping loss and PatchGAN discriminators.
"""
from __future__ import annotations

import argparse
import os
import tqdm
import random
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import yaml

from src.dataloader.dataset_synthetic import SyntheticStereoDataset
from src.dataloader.dataset_real import RealImageDataset
from src.dataloader.transforms import random_hflip
from src.models.generator import EdgeAwareGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.losses.adv import gan_discriminator_loss, gan_generator_loss
from src.losses.warping import warp_loss
from src.utils.sobel import sobel_edges


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_checkpoint(state: Dict[str, Any], out_dir: str, step: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint_{step:07d}.pt")
    torch.save(state, path)


def _save_samples(
    images: Dict[str, torch.Tensor], out_dir: str, step: int, max_items: int = 4
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for name, img in images.items():
            img = img[:max_items]
            save_image((img + 1) * 0.5, os.path.join(out_dir, f"{name}_{step:07d}.png"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SyntStereo2Real translation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="experiments", help="Output root")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--use_edge", dest="use_edge", action="store_true")
    parser.add_argument("--no_edge", dest="use_edge", action="store_false")
    parser.set_defaults(use_edge=True)
    parser.add_argument("--use_warp", dest="use_warp", action="store_true")
    parser.add_argument("--no_warp", dest="use_warp", action="store_false")
    parser.set_defaults(use_warp=True)
    return parser.parse_args()


def _build_dataloaders(cfg: Dict[str, Any], num_workers: int) -> Dict[str, DataLoader]:
    syn_cfg = cfg["synthetic"]
    real_cfg = cfg["real"]

    syn_ds = SyntheticStereoDataset(
        root=syn_cfg["root"],
        left_dir=syn_cfg.get("left_dir", "left"),
        right_dir=syn_cfg.get("right_dir", "right"),
        disp_dir=syn_cfg.get("disp_dir", "disp"),
        disp_rename=tuple(syn_cfg.get("disp_rename")) if syn_cfg.get("disp_rename") else None,
        file_list=syn_cfg.get("file_list"),
        resize=tuple(syn_cfg.get("resize")) if syn_cfg.get("resize") else None,
        normalize=True,
        compute_edges=False,
        return_edges=False,
    )

    real_ds = RealImageDataset(
        root=real_cfg["root"],
        image_dir=real_cfg.get("image_dir", "images"),
        file_list=real_cfg.get("file_list"),
        resize=tuple(real_cfg.get("resize")) if real_cfg.get("resize") else None,
        normalize=True,
        compute_edges=False,
        return_edges=False,
    )

    syn_loader = DataLoader(
        syn_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    real_loader = DataLoader(
        real_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return {"synthetic": syn_loader, "real": real_loader}


def train(config_path: str, output_root: str, device: str, num_workers: int, log_every: int, save_every: int, use_edge: bool, use_warp: bool) -> None:
    cfg = _load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    loaders = _build_dataloaders(cfg, num_workers)
    syn_loader = loaders["synthetic"]
    real_loader = loaders["real"]

    gen = EdgeAwareGenerator(
        style_dim=cfg.get("style_dim", 8),
        content_base_channels=cfg.get("content_base_channels", 64),
        edge_base_channels=cfg.get("edge_base_channels", 32),
        content_res_blocks=cfg.get("content_res_blocks", 4),
        edge_res_blocks=cfg.get("edge_res_blocks", 2),
        decoder_res_blocks=cfg.get("decoder_res_blocks", 4),
    ).to(device)

    disc_a = PatchGANDiscriminator().to(device)
    disc_b = PatchGANDiscriminator().to(device)

    lr = cfg.get("lr", 1e-4)
    betas = tuple(cfg.get("betas", [0.5, 0.999]))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_da = torch.optim.Adam(disc_a.parameters(), lr=lr, betas=betas)
    opt_db = torch.optim.Adam(disc_b.parameters(), lr=lr, betas=betas)

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        opt_g,
        lr_lambda=lambda e: 1.0 if e < cfg.get("lr_decay_start", 50) else max(0.0, 1 - (e - cfg.get("lr_decay_start", 50)) / float(cfg.get("epochs", 100) - cfg.get("lr_decay_start", 50))),
    )
    scheduler_da = torch.optim.lr_scheduler.LambdaLR(opt_da, lr_lambda=scheduler_g.lr_lambdas[0])
    scheduler_db = torch.optim.lr_scheduler.LambdaLR(opt_db, lr_lambda=scheduler_g.lr_lambdas[0])

    out_dir = os.path.join(output_root, cfg.get("experiment_name", "translation"))
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    writer = SummaryWriter(log_dir=log_dir)

    lambda_rec = cfg.get("lambda_rec", 0.8)
    lambda_cycle = cfg.get("lambda_cycle", 10.0)
    lambda_adv = cfg.get("lambda_adv", 10.0)
    lambda_l1 = cfg.get("lambda_warp_l1", 1.0)
    lambda_ssim = cfg.get("lambda_warp_ssim", 1.0)

    epochs = cfg.get("epochs", 100)
    global_step = 0

    real_iter = iter(real_loader)
    for epoch in tqdm.trange(epochs):
        for syn_batch in tqdm.tqdm(syn_loader, desc=f"batches in epoch {epoch}"):
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_batch = next(real_iter)

            syn_batch = random_hflip(syn_batch)
            real_batch = random_hflip(real_batch)

            xl = syn_batch["xl"].to(device)
            xr = syn_batch["xr"].to(device)
            xd = syn_batch["xd"].to(device)
            xb = real_batch["xb"].to(device)

            if use_edge:
                xl_edge = sobel_edges(xl)
                xr_edge = sobel_edges(xr)
                xb_edge = sobel_edges(xb)
            else:
                xl_edge = xr_edge = xb_edge = None

            # Generator forward
            style_a = gen.get_style("a", xl.size(0), xl.device)
            style_b = gen.get_style("b", xl.size(0), xl.device)

            c_a = gen.encode(xl)
            e_a = gen.encode_edge(xl_edge) if xl_edge is not None else None
            c_ra = gen.encode(xr)
            e_ra = gen.encode_edge(xr_edge) if xr_edge is not None else None
            c_b = gen.encode(xb)
            e_b = gen.encode_edge(xb_edge) if xb_edge is not None else None

            x_ab = gen.decode(gen.fuse(c_a, e_a), style_b)
            x_rab = gen.decode(gen.fuse(c_ra, e_ra), style_b)
            x_ba = gen.decode(gen.fuse(c_b, e_b), style_a)

            # Reconstruction
            rec_a = gen.decode(gen.fuse(c_a, e_a), style_a)
            rec_b = gen.decode(gen.fuse(c_b, e_b), style_b)

            # Cycle
            c_ab = gen.encode(x_ab)
            e_ab = gen.encode_edge(sobel_edges(x_ab)) if use_edge else None
            cyc_a = gen.decode(gen.fuse(c_ab, e_ab), style_a)

            c_ba = gen.encode(x_ba)
            e_ba = gen.encode_edge(sobel_edges(x_ba)) if use_edge else None
            cyc_b = gen.decode(gen.fuse(c_ba, e_ba), style_b)

            # Losses
            l_rec = torch.mean(torch.abs(rec_a - xl)) + torch.mean(torch.abs(rec_b - xb))
            l_cycle = torch.mean(torch.abs(cyc_a - xl)) + torch.mean(torch.abs(cyc_b - xb))

            pred_fake_a = disc_a(x_ba)
            pred_fake_b = disc_b(x_ab)
            l_adv = gan_generator_loss(pred_fake_a) + gan_generator_loss(pred_fake_b)

            if use_warp:
                l_warp, _, _ = warp_loss(x_ab, x_rab, xd, lambda_l1=lambda_l1, lambda_ssim=lambda_ssim)
            else:
                l_warp = torch.tensor(0.0, device=device)

            l_gen = lambda_rec * l_rec + lambda_cycle * l_cycle + lambda_adv * l_adv + l_warp

            opt_g.zero_grad()
            l_gen.backward()
            opt_g.step()

            # Discriminator updates
            with torch.no_grad():
                x_ab_det = x_ab.detach()
                x_ba_det = x_ba.detach()

            pred_real_a = disc_a(xl)
            pred_fake_a = disc_a(x_ba_det)
            d_a_loss, _, _ = gan_discriminator_loss(pred_real_a, pred_fake_a)

            pred_real_b = disc_b(xb)
            pred_fake_b = disc_b(x_ab_det)
            d_b_loss, _, _ = gan_discriminator_loss(pred_real_b, pred_fake_b)

            opt_da.zero_grad()
            d_a_loss.backward()
            opt_da.step()

            opt_db.zero_grad()
            d_b_loss.backward()
            opt_db.step()

            if global_step % log_every == 0:
                writer.add_scalar("loss/gen_total", l_gen.item(), global_step)
                writer.add_scalar("loss/rec", l_rec.item(), global_step)
                writer.add_scalar("loss/cycle", l_cycle.item(), global_step)
                writer.add_scalar("loss/adv", l_adv.item(), global_step)
                writer.add_scalar("loss/warp", l_warp.item(), global_step)
                writer.add_scalar("loss/d_a", d_a_loss.item(), global_step)
                writer.add_scalar("loss/d_b", d_b_loss.item(), global_step)

            if global_step % save_every == 0:
                _save_samples(
                    {
                        "xl": xl,
                        "xr": xr,
                        "xb": xb,
                        "x_ab": x_ab,
                        "x_ba": x_ba,
                    },
                    sample_dir,
                    global_step,
                )
                _save_checkpoint(
                    {
                        "gen": gen.state_dict(),
                        "disc_a": disc_a.state_dict(),
                        "disc_b": disc_b.state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_da": opt_da.state_dict(),
                        "opt_db": opt_db.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "seed": seed,
                        "config": cfg,
                    },
                    ckpt_dir,
                    global_step,
                )

            global_step += 1

        scheduler_g.step()
        scheduler_da.step()
        scheduler_db.step()

    writer.close()


def _validate_device(device_str: str) -> str:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    return device_str


def main() -> None:
    args = _parse_args()
    device = _validate_device(args.device)

    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    train(
        config_path=args.config,
        output_root=args.output,
        device=device,
        num_workers=args.num_workers,
        log_every=args.log_every,
        save_every=args.save_every,
        use_edge=args.use_edge,
        use_warp=args.use_warp,
    )


if __name__ == "__main__":
    main()
