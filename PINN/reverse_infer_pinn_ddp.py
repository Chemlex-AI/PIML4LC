import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


RR_LIST = [0.7037, 0.76296, 0.82222, 0.88148, 0.94074, 1.0]
PARAM_NAMES = [
    "ka",
    "kd",
    "qmax",
    "sa",
    "sd",
    "sq",
    "col_disp",
    "fil_diff",
    "par_diff",
]
# Match PARAM_NAMES:
# ka, kd, qmax, sa, sd, sq, D_a_GRM, k_f, D_p.
DEFAULT_LOWER = np.array(
    [1.0, 1.0, 10.0, 0.0, 0.0, 0.0, 1.0e-9, 1.0e-3, 1.0e-9],
    dtype=np.float64,
)
DEFAULT_UPPER = np.array(
    [1.0e3, 1.0e3, 1.0e4, 10.0, 10.0, 10.0, 1.0e-7, 1.0e-1, 1.0e-7],
    dtype=np.float64,
)
LOG_SCALE_MASK = np.array([True, True, True, False, False, False, True, True, True])


class PINN(nn.Module):
    def __init__(self, hidden_size):
        super(PINN, self).__init__()
        self.IL = nn.Linear(12, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.OL = nn.Linear(hidden_size // 2, 3)

    def forward(self, x):
        x = self.IL(x)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.OL(x)
        return x


def clean_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def build_model(checkpoint_path, device, hidden_size=None):
    state_dict = clean_state_dict(torch.load(checkpoint_path, map_location=device))
    if "IL.weight" not in state_dict:
        raise ValueError(
            "This script expects the 100k_pinn_ddp.py architecture with IL.weight. "
            "For checkpoints with IL_shared.weight, use the newer shared-head scripts."
        )
    input_size = state_dict["IL.weight"].shape[1]
    if input_size != 12:
        raise ValueError(f"Expected checkpoint input size 12, got {input_size}.")

    hidden_size = hidden_size or state_dict["IL.weight"].shape[0]
    model = PINN(hidden_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_h5_columns(data_path):
    with h5py.File(data_path, "r") as f:
        arrays = {name: f[name][:] for name in f.keys()}

    for name, value in arrays.items():
        if value.ndim == 1:
            arrays[name] = value.reshape(-1, 1)

    input_values = np.concatenate(
        (
            arrays["ka"],
            arrays["kd"],
            arrays["qmax"],
            arrays["sa"],
            arrays["sd"],
            arrays["sq"],
            arrays["col_disp"],
            arrays["fil_diff"],
            arrays["par_diff"],
            arrays["bul0"],
        ),
        axis=1,
    )
    return arrays, input_values


def load_vector(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64).reshape(-1)
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 2:
        data = data[:, -1]
    return data.reshape(-1)


def parse_float_list(values, expected, name):
    if values is None:
        return None
    parsed = np.array([float(value) for value in values], dtype=np.float64)
    if parsed.shape[0] != expected:
        raise ValueError(f"--{name} expects {expected} values, got {parsed.shape[0]}.")
    return parsed


def load_initial_from_json(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        values = np.array(data, dtype=np.float64)
    else:
        values = np.array([data[name] for name in PARAM_NAMES], dtype=np.float64)
    if values.shape[0] != len(PARAM_NAMES):
        raise ValueError(f"initial json must contain {len(PARAM_NAMES)} values.")
    return values


def random_initial(lower, upper, seed=None):
    rng = np.random.default_rng(seed)
    values = np.empty_like(lower, dtype=np.float64)
    log_mask = LOG_SCALE_MASK & (lower > 0)
    values[log_mask] = np.exp(
        rng.uniform(np.log(lower[log_mask]), np.log(upper[log_mask]))
    )
    values[~log_mask] = rng.uniform(lower[~log_mask], upper[~log_mask])
    return values


def build_fixed_coordinates(arrays, sample_index, axial_indices, time_indices, radius):
    rows = []
    for tt in time_indices:
        for axial_index in axial_indices:
            rows.append([tt / 10.6, axial_index / 100.0, radius])
    return np.asarray(rows, dtype=np.float64)


def build_target(arrays, sample_index, axial_indices, time_indices, target_path):
    if target_path:
        target = load_vector(target_path)
        expected = len(axial_indices) * len(time_indices)
        if target.shape[0] != expected:
            raise ValueError(f"target has {target.shape[0]} points, but expected {expected}.")
        return target

    values = []
    for tt in time_indices:
        for axial_index in axial_indices:
            values.append(arrays["bul"][sample_index, tt, axial_index])
    return np.asarray(values, dtype=np.float64)


def bounded_decode(raw, lower, upper):
    lower_t = torch.as_tensor(lower, dtype=raw.dtype, device=raw.device)
    upper_t = torch.as_tensor(upper, dtype=raw.dtype, device=raw.device)
    log_mask = torch.as_tensor(LOG_SCALE_MASK & (lower > 0), dtype=torch.bool, device=raw.device)
    ratio = torch.sigmoid(raw)
    decoded = lower_t + ratio * (upper_t - lower_t)
    decoded[log_mask] = torch.exp(
        torch.log(lower_t[log_mask])
        + ratio[log_mask] * (torch.log(upper_t[log_mask]) - torch.log(lower_t[log_mask]))
    )
    return decoded


def initial_to_raw(initial, lower, upper, device):
    initial = np.clip(initial, lower, upper)
    ratio = (initial - lower) / (upper - lower)
    log_mask = LOG_SCALE_MASK & (lower > 0)
    ratio[log_mask] = (
        (np.log(initial[log_mask]) - np.log(lower[log_mask]))
        / (np.log(upper[log_mask]) - np.log(lower[log_mask]))
    )
    ratio = np.clip(ratio, 1e-6, 1.0 - 1e-6)
    raw = np.log(ratio / (1.0 - ratio))
    return torch.tensor(raw, dtype=torch.float32, device=device, requires_grad=True)


def make_prediction(model, fixed_xt, raw_params, lower, upper):
    params = bounded_decode(raw_params, lower, upper)
    para = params.unsqueeze(0).expand(fixed_xt.shape[0], -1)
    x = torch.cat((fixed_xt, para), dim=1)
    return model(x)[:, 0], params


def optimize_parameters(args):
    device = torch.device(args.device)
    arrays, input_values = load_h5_columns(args.data_path)

    if args.sample_index < 0 or args.sample_index >= input_values.shape[0]:
        raise ValueError(f"--sample-index must be in [0, {input_values.shape[0] - 1}].")

    axial_indices = args.axial_indices or [args.outlet_index]
    invalid_axial = [idx for idx in axial_indices if idx < 0 or idx >= arrays["bul"].shape[2]]
    if invalid_axial:
        raise ValueError(f"axial indices out of range: {invalid_axial}.")

    start, stop, stride = args.time_slice
    time_indices = list(range(start, min(stop, arrays["bul"].shape[1]), stride))
    if not time_indices:
        raise ValueError("--time-slice selected zero time points.")

    lower = parse_float_list(args.lower, len(PARAM_NAMES), "lower")
    upper = parse_float_list(args.upper, len(PARAM_NAMES), "upper")
    lower = DEFAULT_LOWER.copy() if lower is None else lower
    upper = DEFAULT_UPPER.copy() if upper is None else upper
    if np.any(upper <= lower):
        raise ValueError("Bounds must satisfy lower < upper for every parameter.")
    if np.any(lower[LOG_SCALE_MASK] <= 0):
        raise ValueError("Log-scaled parameter bounds must be positive.")

    if args.initial_json:
        initial = load_initial_from_json(args.initial_json)
    elif args.use_data_initial:
        initial = input_values[args.sample_index, :9].astype(np.float64)
    else:
        initial = random_initial(lower, upper, seed=args.seed)
    initial = np.clip(initial, lower, upper)

    model = build_model(args.checkpoint, device, hidden_size=args.hidden_size)
    fixed_xt_np = build_fixed_coordinates(
        arrays,
        args.sample_index,
        axial_indices,
        time_indices,
        args.radius,
    )
    target_np = build_target(arrays, args.sample_index, axial_indices, time_indices, args.target)

    fixed_xt = torch.tensor(fixed_xt_np, dtype=torch.float32, device=device)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    raw_params = initial_to_raw(initial, lower, upper, device)
    criterion = nn.MSELoss()
    history = []

    def record(step, loss_value, params, prediction):
        pred_np = prediction.detach().cpu().numpy()
        row = {
            "step": int(step),
            "loss": float(loss_value),
            "rmse": float(np.sqrt(mean_squared_error(target_np, pred_np))),
            "r2": float(r2_score(target_np, pred_np)),
            **{name: float(value) for name, value in zip(PARAM_NAMES, params.detach().cpu().numpy())},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    lbfgs = torch.optim.LBFGS(
        [raw_params],
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    lbfgs_step = 0

    def closure():
        nonlocal lbfgs_step
        lbfgs.zero_grad()
        prediction, params = make_prediction(model, fixed_xt, raw_params, lower, upper)
        loss = criterion(prediction, target)
        loss.backward()
        if lbfgs_step % args.log_interval == 0:
            record(lbfgs_step, loss.item(), params, prediction)
        lbfgs_step += 1
        return loss

    lbfgs.step(closure)
    with torch.no_grad():
        prediction, params = make_prediction(model, fixed_xt, raw_params, lower, upper)
        final_loss = criterion(prediction, target).item()
        record(lbfgs_step, final_loss, params, prediction)

    best = min(history, key=lambda row: row["loss"])
    result = {
        "checkpoint": str(args.checkpoint),
        "data_path": str(args.data_path),
        "sample_index": args.sample_index,
        "axial_indices": axial_indices,
        "time_indices": [time_indices[0], time_indices[-1], stride],
        "target": str(args.target) if args.target else "h5:bul first output",
        "initial": {name: float(value) for name, value in zip(PARAM_NAMES, initial)},
        "best": best,
        "final": history[-1],
        "history": history,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the 100k_pinn_ddp.py PINN checkpoint and infer the 9 physical "
            "input parameters by matching the first model output cl/bul."
        )
    )
    parser.add_argument("--data-path", default="./test_data.h5")
    parser.add_argument("--checkpoint", default="./100k_pinn_6000.pt")
    parser.add_argument("--output", default="./reverse_infer_result.json")
    parser.add_argument("--target", help="Optional .npy or CSV target vector for model output[:, 0].")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--outlet-index", type=int, default=100)
    parser.add_argument("--axial-indices", nargs="+", type=int)
    parser.add_argument(
        "--time-slice",
        nargs=3,
        type=int,
        metavar=("START", "STOP", "STRIDE"),
        default=[0, 301, 1],
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=6000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--initial-json", help="JSON list or dict keyed by parameter name.")
    parser.add_argument("--use-data-initial", action="store_true", help="Initialize from the selected HDF5 sample.")
    parser.add_argument("--seed", type=int, help="Random seed for default random initialization.")
    parser.add_argument("--lower", nargs="+", help="Lower bounds in PARAM_NAMES order.")
    parser.add_argument("--upper", nargs="+", help="Upper bounds in PARAM_NAMES order.")
    parser.add_argument("--lbfgs-lr", type=float, default=0.2)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    result = optimize_parameters(args)
    print("BEST")
    print(json.dumps(result["best"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
