#!/usr/bin/env python3

try:
    from rich import print
except ImportError:
    print("rich not found. Try running pip install rich.")
    print("The following will not be pretty...")
import pathlib
import sys
import subprocess

errors = 0

python_version = sys.version.split('|')[0]
if '3.11' in python_version:
    print(f":white_check_mark: Python version: {python_version}")
else:
    print(f":x: Python version: {python_version}. Create a new virtual environment.")
    errors += 1

try:
    import nemos
except ModuleNotFoundError:
    errors += 1
    print(":x: Nemos not found. Try running [bold]pip install nemos[/bold]")
else:
    print(f":white_check_mark: Nemos version: {nemos.__version__}")

try:
    import fastplotlib as fpl
except ModuleNotFoundError:
    errors += 1
    print(":x: fastplotlib not found. Try running [bold]pip install fastplotlib[notebook][/bold]")
except ImportError as e:
    errors += 1
    print(f":x: fastplotlib installation issue:")
    print(e)
else:
    print(f":white_check_mark: fastplotlib version: {fpl.__version__}")

try:
    import pynapple as nap
except ModuleNotFoundError:
    errors += 1
    print(":x: pynapple not found. Try running [bold]pip install pynapple[/bold]")
else:
    print(f":white_check_mark: pynapple version: {nap.__version__}")

p = subprocess.run(['jupyter', '--version'], capture_output=True)
if p.returncode != 0:
    errors += 1
    print(":x: jupyter not found. Try running [bold]pip install jupyter[/bold]")
else:
    # convert to str from bytestring
    stdout = '\n'.join(p.stdout.decode().split('\n')[1:])
    print(f":white_check_mark: jupyter found with following core packages:\n{stdout}")

repo_dir = pathlib.Path(__file__).parent.parent / 'notebooks'
gallery_dir = pathlib.Path(__file__).parent.parent / 'docs' / 'examples'
nbs = list(repo_dir.glob('*ipynb'))
gallery_scripts = list(gallery_dir.glob('*py'))
if len(nbs) == len(gallery_scripts):
    print(":white_check_mark: All notebooks found")
else:
    errors += 1
    missing_nb = [f.stem for f in gallery_scripts
                  if not any([f.stem == nb.stem.replace('_users', '') for nb in nbs])]
    print(f":x: Following notebooks missing: {', '.join(missing_nb)}")
    print("   Did you run [bold]python scripts/setup.py[/bold]?")

try:
    import workshop_utils
except ModuleNotFoundError:
    errors += 1
    print(f":x: workshop utilities not found. Try running [bold]pip install .[/bold] from the github repo.")
else:
    missing_files = []
    for f in workshop_utils.DOWNLOADABLE_FILES:
        if not workshop_utils.fetch.retriever.is_available(f):
            missing_files.append(f)
    if len(missing_files) > 0:
        errors += 1
        print(f":x: Following data files not downloaded: {', '.join(missing_files)}")
        print("   Did you run [bold]python scripts/setup.py[/bold]?")
    else:
        print(":white_check_mark: All data files found!")


if errors == 0:
    print("\n:tada::tada: Congratulations, setup successful!")
else:
    print(f"\n:worried: [red bold]{errors} Errors found.[/red bold]")
    print("Unfortunately, your setup was unsuccessful.")
    print("Try to resolve following the suggestions above.")
    print("If you encountered many installation errors, run [bold] pip install .[/bold] (note the dot!)")
    print("If you are unable to fix your setup yourself, please come to the setup help")
    print("in the hotel, Palais Saschen Coburg room IV, from 4 to 6 pm on Saturday, June 22, and show us the output of this command.")
