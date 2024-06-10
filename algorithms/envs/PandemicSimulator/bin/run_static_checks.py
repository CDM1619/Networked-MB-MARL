#!/usr/bin/env python3
# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import os
import subprocess
import sys
from pathlib import Path


def flake8() -> int:
    p = subprocess.run(['flake8'], stdout=subprocess.PIPE)
    print(p.stdout.decode())
    return p.returncode


def mypy() -> int:
    root_path = Path(__file__).resolve().parents[1]
    paths = [
        Path('bin'),
        Path('python/pandemic_simulator'),
        Path('scripts/tutorials'),
        Path('scripts/simulator_experiments'),
        Path('scripts/calibration'),
        Path('test/environment')
    ]
    # we need --scripts-are-modules since we specify some files on the command line and they have usages of __main__
    cmd = ['mypy', '--show-absolute-path', '--scripts-are-modules'] + [str(x) for x in paths]
    p = subprocess.run(cmd, cwd=root_path, stdout=subprocess.PIPE)
    print(p.stdout.decode())
    return p.returncode


def print_header(name: str, desired_length: int = 30) -> None:
    rem_length = desired_length - len(name) - 2  # 2 for spaces
    prefix = int(rem_length / 2) * '='
    suffix = int(rem_length / 2 + 0.5) * '='
    header = f'{prefix} {name} {suffix}'
    print(header)


def print_footer(desired_length: int = 30) -> None:
    print(desired_length * '=' + '\n')


def main() -> int:
    root_path = Path(__file__).resolve().parents[1]
    os.chdir(str(root_path))
    failures = []

    name_to_test_fn = {'flake8': flake8, 'mypy': mypy}

    for name, test_fn in name_to_test_fn.items():
        print_header(name)
        subprocess.run([name, '--version'])
        res = test_fn()
        if res != 0:
            failures.append(name)
        print_footer()

    if len(failures) == 0:
        print('Success!')
        return 0
    else:
        print(f'Static checks failed: {", ".join(failures)}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
