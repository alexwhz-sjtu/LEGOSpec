"""Capture hidden states with bounded async transfers and versioned tar layouts."""
import argparse
import sys
from pathlib import Path

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
    run_generation(args, metrics)
    if settings.profiling_output is not None:
        metrics.finish(settings.profiling_output)


if __name__ == '__main__':
    main()
