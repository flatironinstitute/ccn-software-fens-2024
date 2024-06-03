#!/usr/bin/env python3

from rich import print as rprint
import pathlib
import sys
import subprocess

errors = 0

python_version = sys.version.split('|')[0]
if '3.11' in python_version:
    rprint(f":white_check_mark: Python version: {python_version}")
else:
    rprint(f":x: Python version: {python_version}. Create a new virtual environment.")
    errors += 1

try:
    import nemos
except ModuleNotFoundError:
    errors += 1
    rprint(":x: Nemos not found. Try running [bold]pip install nemos[/bold]")
else:
    rprint(f":white_check_mark: Nemos version: {nemos.__version__}")

try:
    import fastplotlib as fpl
except ModuleNotFoundError:
    errors += 1
    rprint(":x: fastplotlib not found. Try running [bold]pip install fastplotlib[notebook][/bold]")
except ImportError as e:
    errors += 1
    rprint(f":x: fastplotlib installation issue:")
    rprint(e)
else:
    rprint(f":white_check_mark: fastplotlib version: {fpl.__version__}")

try:
    import pynapple as nap
except ModuleNotFoundError:
    errors += 1
    rprint(":x: pynapple not found. Try running [bold]pip install pynapple[/bold]")
else:
    rprint(f":white_check_mark: pynapple version: {nap.__version__}")

p = subprocess.run(['jupyter', '--version'], capture_output=True)
if p.returncode != 0:
    errors += 1
    rprint(":x: jupyter not found. Try running [bold]pip install jupyter[/bold]")
else:
    # convert to str from bytestring
    stdout = p.stdout.decode()
    rprint(f":white_check_mark: jupyter found with following core packages: {stdout}")

repo_dir = pathlib.Path(__file__).parent.parent / 'notebooks'
gallery_dir = pathlib.Path(__file__).parent.parent / 'docs' / 'examples'
nbs = list(repo_dir.glob('*ipynb'))
gallery_scripts = list(gallery_dir.glob('*py'))
if len(nbs) == len(gallery_scripts):
    rprint(":white_check_mark: All notebooks found")
else:
    errors += 1
    missing_nb = [f.stem for f in gallery_scripts
                  if not any([f.stem == nb.stem.replace('_users', '') for nb in nbs])]
    rprint(f":x: Following notebooks missing: {', '.join(missing_nb)}")
    rprint("   Did you run [bold]python scripts/setup.py[/bold]?")

try:
    import workshop_utils
except ModuleNotFoundError:
    errors += 1
    rprint(f":x: workshop utilities not found. Try running [bold]pip install .[/bold] from the github repo.")
else:
    missing_files = []
    for f in workshop_utils.DOWNLOADABLE_FILES:
        if not workshop_utils.fetch.retriever.is_available(f):
            missing_files.append(f)
    if len(missing_files) > 0:
        errors += 1
        rprint(f":x: Following data files not downloaded: {', '.join(missing_files)}")
        rprint("   Did you run [bold]python scripts/setup.py[/bold]?")
    else:
        rprint(":white_check_mark: All data files found!")


if errors == 0:
    rprint("\n:tada::tada: Congratulations, setup successful!")
else:
    rprint("\n:worried: Unfortunately, your setup was unsuccessful.")
    rprint("Try to resolve following the suggestions above and,")
    rprint("if you are still having trouble, come to the setup help")
    rprint("in the hotel lobby at 4pm on Saturday, June 22.")
