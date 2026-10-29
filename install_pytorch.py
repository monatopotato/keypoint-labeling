"""Install matching PyTorch wheels for CPU or NVIDIA CUDA, then YOLO."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--cuda', choices=['cu126', 'cu130'], default='cu126',
                        help='CUDA wheel build (default: cu126); requires a compatible NVIDIA driver')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('This pinned PyTorch version requires Python 3.11 or newer. Recreate .venv with Python 3.11+.')
    if sys.platform == 'darwin' and args.device == 'cuda':
        parser.error('NVIDIA CUDA wheels are supported on Windows/Linux, not macOS.')
    build = args.cuda if args.device == 'cuda' else 'cpu'
    suffix = '' if sys.platform == 'darwin' else '+' + build
    command = [sys.executable, '-m', 'pip', 'install',
               'torch==2.14.1' + suffix, 'torchvision==0.29.1' + suffix]
    if sys.platform != 'darwin':
        command += ['--index-url', 'https://download.pytorch.org/whl/' + build]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(Path(__file__).with_name('requirements-yolo.txt'))], check=True)
    # Use a new process so switching builds cannot reuse already imported modules.
    subprocess.run([sys.executable, '-c',
                    "import torch; from ultralytics import YOLO; "
                    "print('PyTorch:', torch.__version__); "
                    "print('CUDA available:', torch.cuda.is_available()); "
                    "print('CPU ready:', (torch.ones(1) + 1).item() == 2)"], check=True)
    print('Done. Restart the server. Auto uses CUDA when available, otherwise CPU.')


if __name__ == '__main__':
    main()
