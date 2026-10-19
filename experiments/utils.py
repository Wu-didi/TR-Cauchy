import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods import METHODS


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument("--n-epochs", type=int, default=300)
common_parser.add_argument("--batch-size", type=int, default=120, help="batch size")
common_parser.add_argument(
    "--method", type=str, choices=list(METHODS.keys()), help="MTL weight method"
)
common_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
common_parser.add_argument(
    "--method-params-lr",
    type=float,
    default=0.025,
    help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")
# NashMTL
common_parser.add_argument(
    "--nashmtl-optim-niter", type=int, default=20, help="number of CCCP iterations"
)
common_parser.add_argument(
    "--update-weights-every",
    type=int,
    default=1,
    help="update task weights every x iterations.",
)
# stl
common_parser.add_argument(
    "--main-task",
    type=int,
    default=0,
    help="main task for stl. Ignored if method != stl",
)
# cagrad
common_parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
common_parser.add_argument(
    "--cagrad-rescale", type=int, default=1, choices=[0, 1, 2],
    help="CAGrad rescale mode; shared unchanged by CAGrad and TR-CAGrad.",
)
# TR-Cauchy (shared by TR-PCGrad and TR-CAGrad)
common_parser.add_argument(
    "--tr-mode", choices=["adaptive_tr", "fixed_tr", "cosine"],
    default="adaptive_tr",
)
common_parser.add_argument("--tr-delta0", type=float, default=1e-2)
common_parser.add_argument("--tr-delta-min", type=float, default=1e-4)
common_parser.add_argument("--tr-delta-max", type=float, default=1.0)
common_parser.add_argument("--tr-eta1", type=float, default=0.1)
common_parser.add_argument("--tr-eta2", type=float, default=0.8)
common_parser.add_argument("--tr-gamma-dec", type=float, default=0.8)
common_parser.add_argument("--tr-gamma-inc", type=float, default=1.5)
common_parser.add_argument(
    "--tr-curvature", choices=["identity"], default="identity",
    help="Explicit curvature model used in the Cauchy quadratic model.",
)
common_parser.add_argument("--tr-damping", type=float, default=1e-8)
# famo
common_parser.add_argument("--gamma", type=float, default=0.01, help="gamma of famo")
common_parser.add_argument("--use_log", action='store_true', help="whether use log for famo")
common_parser.add_argument("--max_norm", type=float, default=1.0, help="beta for RMS_weight alg.")
common_parser.add_argument("--task", type=int, default=0, help="train single task number for (celeba)")
# dwa
common_parser.add_argument(
    "--dwa-temp",
    type=float,
    default=2.0,
    help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.",
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


import logging, os, time

def get_logger(log_dir="logs", name="run", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    stamp   = time.strftime("%Y%m%d_%H%M%S")          # 20250515_215310
    logfile = f"{log_dir}/{name}_{stamp}.log"

    logging.basicConfig(
        filename=logfile,            # 写文件
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()                  # 终端
    console.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(console)

    logging.info(f"logger ready → {logfile}")
    return logging.getLogger()


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def extract_weight_method_parameters_from_args(args):
    tr_parameters = dict(
        tr_mode=args.tr_mode,
        delta0=args.tr_delta0,
        delta_min=args.tr_delta_min,
        delta_max=args.tr_delta_max,
        eta1=args.tr_eta1,
        eta2=args.tr_eta2,
        gamma_dec=args.tr_gamma_dec,
        gamma_inc=args.tr_gamma_inc,
        curvature_mode=args.tr_curvature,
        damping=args.tr_damping,
    )
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
                max_norm=args.max_norm,
            ),
            stl=dict(main_task=args.main_task),
            dwa=dict(temp=args.dwa_temp),
            cagrad=dict(
                c=args.c,
                rescale=args.cagrad_rescale,
                max_norm=args.max_norm,
            ),
            trpcgrad=dict(**tr_parameters),
            trcagrad=dict(
                c=args.c,
                rescale=args.cagrad_rescale,
                **tr_parameters,
            ),
            log_cagrad=dict(c=args.c, max_norm=args.max_norm),
            famo=dict(gamma=args.gamma,
                      w_lr=args.method_params_lr,
                      max_norm=args.max_norm),
        )
    )
    return weight_methods_parameters


import csv, time
from pathlib import Path

def save_metrics_nyuv2(epoch: int,
                 avg_cost,
                 test_delta_m,
                 save_dir: str = "logs"):
    """
    将一个 epoch 的训练 / 测试指标追加到  <save_dir>/metrics.csv
    - avg_cost 形状: [n_epoch, 24]
    - save_dir 若不存在将自动创建
    """
    # 1️⃣ 指标名称 & 当前行数据
    metric_names = [
        "Train Semantic Loss", "Train Mean IoU",        "Train Pixel Accuracy",
        "Train Depth Loss",     "Train Absolute Error", "Train Relative Error",
        "Train Normal Loss",    "Train Loss Mean",      "Train Loss Med",
        "Train Loss <11.25",    "Train Loss <22.5",     "Train Loss <30",
        "Test  Semantic Loss",  "Test  Mean IoU",       "Test  Pixel Accuracy",
        "Test  Depth Loss",     "Test  Absolute Error", "Test  Relative Error",
        "Test  Normal Loss",    "Test  Loss Mean",      "Test  Loss Med",
        "Test  Loss <11.25",    "Test  Loss <22.5",     "Test  Loss <30",
        "Test  Δm"
    ]

    values = list(avg_cost[epoch, :24]) + [test_delta_m]
    row    = {
        "epoch": epoch,
        "time" : time.strftime("%Y-%m-%d %H:%M:%S"),
        **{k: v for k, v in zip(metric_names, values)}
    }

    # 2️⃣ 目标文件路径：<save_dir>/metrics.csv
    csv_path = Path(save_dir) / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 3️⃣ 首次写入需带表头
    first_write = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "time"] + metric_names)
        if first_write:
            writer.writeheader()
        writer.writerow(row)


def save_metrics_cityscapes(epoch: int,
                    avg_cost,
                    test_delta_m,
                    save_dir: str = "logs"):
    """
    把一个 epoch 的 13 个指标写入 <save_dir>/metrics.csv
    avg_cost 只需包含 [n_epoch, 12]，下标 0-11 对应 train/test 各 6 项
    """
    metric_names = [
        "Train Semantic Loss", "Train Mean IoU",      "Train Pixel Accuracy",
        "Train Depth Loss",    "Train Absolute Error","Train Relative Error",
        "Test  Semantic Loss", "Test  Mean IoU",      "Test  Pixel Accuracy",
        "Test  Depth Loss",    "Test  Absolute Error","Test  Relative Error",
        "Test  Δm"
    ]

    values = list(avg_cost[epoch, :12]) + [test_delta_m]
    row = {
        "epoch": epoch,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **{k: v for k, v in zip(metric_names, values)}
    }

    csv_path = Path(save_dir) / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    first_write = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "time"] + metric_names)
        if first_write:
            writer.writeheader()
        writer.writerow(row)
