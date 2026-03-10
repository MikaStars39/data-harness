import asyncio
import argparse
import warnings
import sys
from pathlib import Path
import importlib.util

# Load BatchInferenceEngine from data-harness package
def _load_batch_inference_engine():
    repo_root = Path(__file__).resolve().parents[2]
    package_name = "data_harness_runtime"
    package_dir = repo_root / "data-harness"
    offline_path = package_dir / "offline.py"

    if not offline_path.exists():
        raise FileNotFoundError(f"Cannot find `data-harness/offline.py` at {offline_path}.")

    package_spec = importlib.util.spec_from_loader(package_name, loader=None)
    package_module = importlib.util.module_from_spec(package_spec)
    package_module.__path__ = [str(package_dir)]
    sys.modules[package_name] = package_module

    # Load base first (required by offline.py)
    base_path = package_dir / "base.py"
    base_spec = importlib.util.spec_from_file_location(f"{package_name}.base", base_path)
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules[f"{package_name}.base"] = base_module
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)

    offline_spec = importlib.util.spec_from_file_location(f"{package_name}.offline", offline_path)
    offline_module = importlib.util.module_from_spec(offline_spec)
    sys.modules[f"{package_name}.offline"] = offline_module
    assert offline_spec.loader is not None
    offline_spec.loader.exec_module(offline_module)
    return offline_module.BatchInferenceEngine


# ------ Logic --------
async def run_batch_inference(args):
    """
    Standard Batch Inference execution.
    """
    BatchInferenceEngine = _load_batch_inference_engine()

    engine_args = {
        "model_path": args.model_path,
        "dp_size": args.dp_size,
        "tp_size": args.tp_size,
        "max_inflight": args.max_concurrency,
        "mem_fraction_static": args.gpu_mem,
        "enable_dp_attention": args.enable_dp_attention,
    }

    sampling_params = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
    }

    async with BatchInferenceEngine(**engine_args) as engine:
        await engine.run(
            input_file=args.input,
            output_file=args.output,
            sampling_params=sampling_params,
            resume=args.resume,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference: Run LLM batch processing.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--enable_dp_attention", action="store_true", help="Enable DP attention for multi-GPU inference")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")

    args = parser.parse_args()
    asyncio.run(run_batch_inference(args))
