from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import shutil
import time
import traceback
import mph
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "index", "phi0", "phie", "ts", "te", "bul0", "D_1", "Dp", "kf",
    "ka", "kd", "qmax", "sads", "sdep", "sq",
)
GRADIENT_CHOICES = (
    (0.95, 120),
    (0.5, 210),
    (0.95, 300),
    (0.95, 210),
    (0.95, 270),
)
EXPORTS = (("数据 1", "cl.csv"), ("数据 2", "cp.csv"), ("数据 3", "q.csv"))


def sample_parameters(count: int, seed: int) -> list[dict[str, float | int]]:
    """Draw independent physical parameters; logarithmic ranges use log10."""
    rng = random.Random(seed)

    def log_uniform(low: float, high: float) -> float:
        return 10.0 ** rng.uniform(low, high)

    rows = []
    for index in range(count):
        phie, ts = rng.choice(GRADIENT_CHOICES)
        rows.append({
            "index": index,
            "phi0": 0.05,
            "phie": phie,
            "ts": ts,
            "te": 360,
            "bul0": 1.0,
            "D_1": log_uniform(-9.0, -7.0),
            "Dp": log_uniform(-9.0, -7.0),
            "kf": log_uniform(-3.0, -1.0),
            "ka": log_uniform(0.0, 3.0),
            "kd": log_uniform(0.0, 3.0),
            "qmax": log_uniform(1.0, 4.0),
            "sads": rng.uniform(0.0, 10.0),
            "sdep": rng.uniform(0.0, 10.0),
            "sq": rng.uniform(0.0, 10.0),
        })
    return rows


def load_or_create_plan(path: Path, count: int, seed: int, resume: bool):
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"No plan to resume: {path}")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected parameter columns in {path}")
            rows = list(reader)
        if len(rows) != count:
            raise ValueError(f"Plan contains {len(rows)} samples; --count is {count}.")
        for index, row in enumerate(rows):
            if int(row["index"]) != index:
                raise ValueError(f"Unexpected sample index at row {index} in {path}")
            if (float(row["phie"]), float(row["ts"])) not in GRADIENT_CHOICES:
                raise ValueError(
                    f"Unsupported (phie, ts) pair at row {index} in {path}; "
                    "create a new plan in a different --output-dir."
                )
        return rows

    if path.exists():
        raise FileExistsError(f"Plan already exists: {path}. Use --resume to keep it.")
    if any(path.parent.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path.parent}")
    rows = sample_parameters(count, seed)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_one(index: int, row: dict, model_path: Path, output_dir: Path, cores: int):
    """Solve one parameter set and publish its three exports together."""
    final_dir = output_dir / f"sample_{index:05d}"
    temp_dir = output_dir / f".sample_{index:05d}.tmp"
    temp_dir.mkdir(exist_ok=False)
    client = None
    model = None
    try:
        client = mph.start(cores=cores)
        model = client.load(str(model_path))
        for key in ("phi0", "phie", "ka", "kd", "qmax", "sads", "sdep", "sq"):
            model.parameter(key, str(row[key]))
        model.parameter("ts", f"{row['ts']} [s]")
        model.parameter("te", f"{row['te']} [s]")
        model.parameter("D_1", f"{row['D_1']} [m^2/s]")
        model.parameter("Dp", f"{row['Dp']} [m^2/s]")
        model.parameter("kf", f"{row['kf']} [m/s]")
        model.parameter("c01", str(row["bul0"]))
        model.build()
        model.mesh()
        model.solve("Study 1")
        for export_name, filename in EXPORTS:
            export_path = temp_dir / filename
            model.export(export_name, str(export_path))
            if not export_path.is_file() or export_path.stat().st_size == 0:
                raise RuntimeError(f"COMSOL did not write a nonempty {filename} export")
        os.rename(temp_dir, final_dir)
    finally:
        if model is not None:
            try:
                model.clear()
            except Exception:
                pass
            try:
                client.remove(model)
            except Exception:
                pass
        if client is not None and hasattr(client, "disconnect"):
            try:
                client.disconnect()
            except Exception:
                pass
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def worker(index, row, model_path, output_dir, cores, connection):
    try:
        run_one(index, row, model_path, output_dir, cores)
        connection.send(None)
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        connection.close()


def run_batch(rows, model_path: Path, output_dir: Path, workers: int,
              cores: int, timeout: float) -> int:
    ctx = mp.get_context("spawn")
    pending = []
    for row in rows:
        final_dir = output_dir / f"sample_{int(row['index']):05d}"
        if final_dir.exists():
            if not final_dir.is_dir() or any(
                not (final_dir / filename).is_file()
                or (final_dir / filename).stat().st_size == 0
                for _, filename in EXPORTS
            ):
                raise RuntimeError(f"Incomplete sample output: {final_dir}")
        else:
            pending.append(row)
    print(f"Plan: {len(rows)} samples; remaining: {len(pending)}", flush=True)
    if not pending:
        return 0
    running = {}
    failed = 0
    next_job = 0
    failures_path = output_dir / "failures.csv"
    with failures_path.open("a", newline="") as failure_file:
        writer = csv.writer(failure_file)
        if failure_file.tell() == 0:
            writer.writerow(("index", "error"))
        try:
            while next_job < len(pending) or running:
                while next_job < len(pending) and len(running) < workers:
                    row = pending[next_job]
                    index = int(row["index"])
                    temp_dir = output_dir / f".sample_{index:05d}.tmp"
                    if temp_dir.exists():  # Left by a timed-out or interrupted worker.
                        shutil.rmtree(temp_dir)
                    receive, send = ctx.Pipe(duplex=False)
                    process = ctx.Process(
                        target=worker,
                        args=(index, row, model_path, output_dir, cores, send),
                    )
                    process.start()
                    send.close()
                    running[index] = (process, receive, time.monotonic())
                    next_job += 1
                    print(f"Started {index} ({next_job}/{len(pending)})", flush=True)

                for index, (process, receive, started) in list(running.items()):
                    expired = process.is_alive() and time.monotonic() - started > timeout
                    if expired:
                        process.terminate()
                        process.join(5)
                        if process.is_alive():
                            process.kill()
                    if expired or not process.is_alive():
                        process.join()
                        error = receive.recv() if receive.poll() else None
                        receive.close()
                        if expired:
                            error = f"Timed out after {timeout:g} seconds"
                        elif error is None and process.exitcode != 0:
                            error = f"Worker exited with code {process.exitcode}"
                        if error is not None:
                            failed += 1
                            writer.writerow((index, error))
                            failure_file.flush()
                            print(f"Failed {index}: {error.splitlines()[-1]}", flush=True)
                        else:
                            print(f"Finished {index}", flush=True)
                        del running[index]
                if running:
                    time.sleep(0.2)
        finally:
            for process, receive, _ in running.values():
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    if process.is_alive():
                        process.kill()
                process.join()
                receive.close()
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "GRM.mph")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comsol_sim")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cores-per-job", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0 or args.cores_per_job <= 0 or args.timeout <= 0:
        parser.error("--count, --workers, --cores-per-job and --timeout must be positive")
    if args.plan_only and args.resume:
        parser.error("--plan-only and --resume cannot be used together")
    model_path = args.model.resolve()
    if not model_path.is_file():
        parser.error(f"COMSOL model does not exist: {model_path}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_or_create_plan(output_dir / "parameters.csv", args.count, args.seed, args.resume)
    if args.plan_only:
        print(f"Saved {len(rows)} parameter sets to {output_dir / 'parameters.csv'}")
        return
    failed = run_batch(rows, model_path, output_dir, args.workers,
                       args.cores_per_job, args.timeout)
    if failed:
        raise SystemExit(f"{failed} simulations failed; inspect {output_dir / 'failures.csv'} and rerun with --resume")
    print(f"Completed all {len(rows)} simulations in {output_dir}")


if __name__ == "__main__":
    main()
