"""Capture hidden states with bounded async transfers and versioned tar layouts."""
import argparse
import sys
from pathlib import Path

import os
import traceback

if 'LOCAL_RANK' in os.environ:
    cache_root=Path(os.environ.get('LEGOSPEC_COMPILER_CACHE_ROOT','./cache/compilers'))
    rank='rank_'+os.environ['LOCAL_RANK']
    for key,subdir in [('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
        base=Path(os.environ.get(key,str(cache_root/subdir)))
        if base.name!=rank:base=base/rank
        os.environ[key]=str(base.resolve())

from litespec.data.capture_engine import parse_args, run_generation
from litespec.data.capture_metrics import CaptureMetrics


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--profiling-output', type=Path,
                        help='Optional timing directory outside the hidden cache.')
    settings, remaining = parser.parse_known_args(argv)
    if '--help' in argv:
        parser.print_help()
    args = parse_args(remaining)
    if not args.output_path:
        parser.error("--output-path is required for optimized capture")
    if settings.profiling_output is not None:
        cache = Path(args.output_path or "./cache/data/regen_full").resolve()
        if settings.profiling_output.resolve().is_relative_to(cache):
            parser.error('--profiling-output must be outside --output-path')
    metrics = CaptureMetrics(profiling=settings.profiling_output is not None)
    try:
        run_generation(args, metrics)
    except BaseException:
        if 'LOCAL_RANK' in os.environ and 'RANK' in os.environ:
            traceback.print_exc();sys.stderr.flush();sys.stdout.flush()
            os._exit(1)  # torchrun cleans peers; do not mask the original exception.
        raise
    if settings.profiling_output is not None:
        metrics.finish(settings.profiling_output)


if __name__ == '__main__':
    main()
