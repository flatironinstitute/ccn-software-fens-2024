#!/usr/bin/env python3

import click
import pathlib
import shutil
import subprocess
import re
import os


@click.command()
def main():
    repo_dir = pathlib.Path(__file__).parent.parent
    nb_dir = repo_dir / 'notebooks'
    scripts_dir = repo_dir / 'scripts'
    src_dir = repo_dir / 'src'
    subprocess.run(['python', scripts_dir / 'strip_text.py'], cwd=repo_dir)
    subprocess.run(['python', src_dir / 'workshop_utils' / 'fetch.py'], cwd=repo_dir)
    env = os.environ.copy()
    env['WGPU_FORCE_OFFSCREEN'] = '1'
    subprocess.run(['mkdocs', 'build'], cwd=repo_dir, env=env)
    gen_nb_dir = repo_dir / 'site' / 'generated' / 'for_users'
    for f in gen_nb_dir.glob('*ipynb'):
        shutil.copy(f.absolute(), (nb_dir / f.name).absolute())
    fpl_test_nb = repo_dir / 'site' / 'generated' / 'setup' / 'test_fastplotlib_installation.ipynb'
    shutil.copy(fpl_test_nb.absolute(), (nb_dir / fpl_test_nb.name).absolute())
    for f in nb_dir.glob('*ipynb'):
        nb_contents = re.sub(r'../../(../)?assets/', r'../docs/assets/',
                             f.read_text())
        f.write_text(nb_contents)


if __name__ == '__main__':
    main()
