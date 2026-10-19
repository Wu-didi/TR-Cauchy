import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error
from experiments.utils import extract_weight_method_parameters_from_args, set_seed, str2bool
from methods.weight_methods import WeightMethods
from methods.tr_cauchy import FunctionalCandidateEvaluator


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        return F.nll_loss(x_pred, x_output, ignore_index=-1)
    if task_type == "depth":
        return torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)
    if task_type == "normal":
        return 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    raise ValueError(f"unknown task type: {task_type}")


def _run_csv_command(args):
    try:
        output = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


class ResourceSampler:
    def __init__(self, pid, physical_gpu, interval=1.0):
        self.pid = pid
        self.physical_gpu = physical_gpu
        self.interval = interval
        self.process = psutil.Process(pid)
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.process.cpu_percent(None)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)

    def _sample_gpu(self):
        gpu_rows = _run_csv_command(
            [
                "nvidia-smi",
                "-i",
                str(self.physical_gpu),
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
        gpu_util = gpu_mem = gpu_power = None
        if gpu_rows:
            parts = [p.strip() for p in gpu_rows[0].split(",")]
            if len(parts) >= 3:
                try:
                    gpu_util = float(parts[0])
                    gpu_mem = float(parts[1])
                    gpu_power = float(parts[2])
                except ValueError:
                    pass

        proc_gpu_mem = 0.0
        proc_rows = _run_csv_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        for row in proc_rows:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 2:
                continue
            try:
                if int(parts[0]) == self.pid:
                    proc_gpu_mem += float(parts[1])
            except ValueError:
                continue

        return gpu_util, gpu_mem, gpu_power, proc_gpu_mem

    def _loop(self):
        while not self._stop.is_set():
            try:
                mem = self.process.memory_info()
                cpu_percent = self.process.cpu_percent(None)
                num_threads = self.process.num_threads()
            except psutil.Error:
                break

            gpu_util, gpu_mem, gpu_power, proc_gpu_mem = self._sample_gpu()
            self.samples.append(
                {
                    "time": time.time(),
                    "cpu_percent": cpu_percent,
                    "rss_mib": mem.rss / (1024**2),
                    "num_threads": num_threads,
                    "gpu_util_percent": gpu_util,
                    "gpu_memory_used_mib": gpu_mem,
                    "gpu_power_w": gpu_power,
                    "process_gpu_memory_mib": proc_gpu_mem,
                }
            )
            self._stop.wait(self.interval)

    def summary(self):
        def values(key):
            return [s[key] for s in self.samples if s.get(key) is not None]

        def mean(key):
            vals = values(key)
            return float(np.mean(vals)) if vals else None

        def max_value(key):
            vals = values(key)
            return float(np.max(vals)) if vals else None

        return {
            "sample_count": len(self.samples),
            "avg_sampled_cpu_percent": mean("cpu_percent"),
            "max_sampled_cpu_percent": max_value("cpu_percent"),
            "peak_rss_mib": max_value("rss_mib"),
            "avg_gpu_util_percent": mean("gpu_util_percent"),
            "max_gpu_util_percent": max_value("gpu_util_percent"),
            "avg_gpu_memory_used_mib": mean("gpu_memory_used_mib"),
            "max_gpu_memory_used_mib": max_value("gpu_memory_used_mib"),
            "avg_process_gpu_memory_mib": mean("process_gpu_memory_mib"),
            "max_process_gpu_memory_mib": max_value("process_gpu_memory_mib"),
            "avg_gpu_power_w": mean("gpu_power_w"),
            "max_gpu_power_w": max_value("gpu_power_w"),
        }


def run_one_epoch(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available; resource benchmark must run on GPU.")

    model = {"segnet": SegNet(), "mtan": SegNetMtan()}[args.model].to(device)

    train_set = NYUv2(
        root=args.data_path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    test_set = NYUv2(root=args.data_path.as_posix(), train=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    method_params = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=3, device=device, **method_params[args.method]
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": weight_method.parameters(), "lr": args.method_params_lr},
        ]
    )

    process = psutil.Process(os.getpid())
    sampler = ResourceSampler(
        pid=os.getpid(), physical_gpu=args.physical_gpu, interval=args.sample_interval
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()
    sampler.start()

    train_batch_count = len(train_loader)
    test_batch_count = len(test_loader)
    if args.max_train_batches > 0:
        train_batch_count = min(train_batch_count, args.max_train_batches)
    if args.max_test_batches > 0:
        test_batch_count = min(test_batch_count, args.max_test_batches)

    avg_cost = np.zeros(24, dtype=np.float32)
    conf_mat = ConfMatrix(model.segnet.class_nb)
    core_computation_time = 0.0
    train_start = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= train_batch_count:
            break
        model.train()
        optimizer.zero_grad()
        train_data, train_label, train_depth, train_normal = batch
        train_data = train_data.to(device)
        train_label = train_label.long().to(device)
        train_depth = train_depth.to(device)
        train_normal = train_normal.to(device)

        candidate_forward_state = None
        if weight_method.requires_candidate_evaluator:
            candidate_forward_state = (
                FunctionalCandidateEvaluator.capture_forward_state(model)
            )
        train_pred, features = model(train_data, return_representation=True)
        losses = torch.stack(
            (
                calc_loss(train_pred[0], train_label, "semantic"),
                calc_loss(train_pred[1], train_depth, "depth"),
                calc_loss(train_pred[2], train_normal, "normal"),
            )
        )

        core_start = time.perf_counter()
        shared_parameters = list(model.shared_parameters())
        candidate_evaluator = None
        if weight_method.requires_candidate_evaluator:
            def candidate_losses(forward):
                candidate_pred = forward(train_data, return_representation=False)
                return torch.stack(
                    (
                        calc_loss(candidate_pred[0], train_label, "semantic"),
                        calc_loss(candidate_pred[1], train_depth, "depth"),
                        calc_loss(candidate_pred[2], train_normal, "normal"),
                    )
                )

            candidate_evaluator = FunctionalCandidateEvaluator(
                model,
                shared_parameters,
                candidate_losses,
                n_tasks=weight_method.n_tasks,
                forward_state=candidate_forward_state,
            )

        weight_method.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=list(model.task_specific_parameters()),
            last_shared_parameters=list(model.last_shared_parameters()),
            representation=features,
            candidate_evaluator=candidate_evaluator,
        )
        core_computation_time += time.perf_counter() - core_start
        optimizer.step()

        if "famo" in args.method:
            with torch.no_grad():
                train_pred = model(train_data, return_representation=False)
                new_losses = torch.stack(
                    (
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                        calc_loss(train_pred[2], train_normal, "normal"),
                    )
                )
                weight_method.method.update(new_losses.detach())

        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())
        avg_cost[0] += losses[0].item() / train_batch_count
        avg_cost[3] += losses[1].item() / train_batch_count
        avg_cost[4:6] += np.array(depth_error(train_pred[1], train_depth)) / train_batch_count
        avg_cost[6] += losses[2].item() / train_batch_count
        avg_cost[7:12] += np.array(normal_error(train_pred[2], train_normal)) / train_batch_count

    torch.cuda.synchronize(device)
    train_time = time.perf_counter() - train_start
    avg_cost[1:3] = conf_mat.get_metrics()

    eval_start = time.perf_counter()
    model.eval()
    conf_mat = ConfMatrix(model.segnet.class_nb)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= test_batch_count:
                break
            test_data, test_label, test_depth, test_normal = batch
            test_data = test_data.to(device)
            test_label = test_label.long().to(device)
            test_depth = test_depth.to(device)
            test_normal = test_normal.to(device)

            test_pred = model(test_data)
            test_loss = torch.stack(
                (
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                    calc_loss(test_pred[2], test_normal, "normal"),
                )
            )
            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
            avg_cost[12] += test_loss[0].item() / test_batch_count
            avg_cost[15] += test_loss[1].item() / test_batch_count
            avg_cost[16:18] += np.array(depth_error(test_pred[1], test_depth)) / test_batch_count
            avg_cost[18] += test_loss[2].item() / test_batch_count
            avg_cost[19:24] += np.array(normal_error(test_pred[2], test_normal)) / test_batch_count

    torch.cuda.synchronize(device)
    eval_time = time.perf_counter() - eval_start
    avg_cost[13:15] = conf_mat.get_metrics()
    test_delta_m = delta_fn(avg_cost[[13, 14, 16, 17, 19, 20, 21, 22, 23]])

    total_time = time.perf_counter() - wall_start
    sampler.stop()
    cpu_end = process.cpu_times()

    cpu_user = cpu_end.user - cpu_start.user
    cpu_system = cpu_end.system - cpu_start.system
    cpu_total = cpu_user + cpu_system

    result = {
        "method": args.method,
        "display_method": args.display_method or args.method,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "model": args.model,
        "batch_size": args.batch_size,
        "train_batches": train_batch_count,
        "test_batches": test_batch_count,
        "visible_gpu": args.gpu,
        "physical_gpu": args.physical_gpu,
        "total_time_s": total_time,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "core_computation_time_s": core_computation_time,
        "cpu_user_time_s": cpu_user,
        "cpu_system_time_s": cpu_system,
        "cpu_total_time_s": cpu_total,
        "avg_cpu_percent_from_time": cpu_total / total_time * 100.0,
        "cpu_core_hours": cpu_total / 3600.0,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "test_delta_m": float(test_delta_m),
    }
    result.update(sampler.summary())
    return result


def write_outputs(result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    method = result["method"]
    json_path = out_dir / f"{method}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    csv_path = out_dir / "summary.csv"
    fieldnames = list(result.keys())
    existing = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            fieldnames = reader.fieldnames or fieldnames
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
        existing = [row for row in existing if row.get("method") != method]
    existing.append({k: result.get(k, "") for k in fieldnames})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    return json_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser("NYUv2 resource profiler")
    parser.add_argument("--method", required=True)
    parser.add_argument("--display-method", default="")
    parser.add_argument("--data-path", type=Path, default=Path("dataset"))
    parser.add_argument("--model", choices=["segnet", "mtan"], default="mtan")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--apply-augmentation", type=str2bool, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)

    parser.add_argument("--method-params-lr", type=float, default=0.025)
    parser.add_argument("--nashmtl-optim-niter", type=int, default=20)
    parser.add_argument("--update-weights-every", type=int, default=1)
    parser.add_argument("--main-task", type=int, default=0)
    parser.add_argument("--c", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--use_log", action="store_true")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--dwa-temp", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_one_epoch(args)
    json_path, csv_path = write_outputs(result, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    print(f"updated {csv_path}")


if __name__ == "__main__":
    main()
