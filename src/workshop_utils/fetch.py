#!/usr/bin/env python3
"""Fetch data using pooch.

This is inspired by scipy's datasets module.
"""
REGISTRY = {
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "A0634-210617.nwb": None,
    "Achilles_10252013.nwb": None,
}

OSF_TEMPLATE = "https://osf.io/{}/download"
# these are all from the OSF project at https://osf.io/ts37w/.
REGISTRY_URLS = {
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "A0634-210617.nwb": OSF_TEMPLATE.format("28ths"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma")
}
DOWNLOADABLE_FILES = list(REGISTRY_URLS.keys())

import pathlib
from typing import List
import pooch
import click
DATA_DIR = pathlib.Path(__file__).parent.parent.parent / 'data'


retriever = pooch.create(
    path=DATA_DIR,
    base_url="",
    urls=REGISTRY_URLS,
    registry=REGISTRY,
    retry_if_failed=2,
    # this defaults to true, unless the env variable with same name is set
    allow_updates="POOCH_ALLOW_UPDATES",
    # user can use FENS_DATA_DIR to update path where we store data
    env="FENS_DATA_DIR",
)


def find_shared_directory(paths: List[pathlib.Path]) -> pathlib.Path:
    """Find directory shared by all paths."""
    for dir in paths[0].parents:
        if all([dir in p.parents for p in paths]):
            break
    return dir


def fetch_data(dataset_name: str) -> pathlib.Path:
    """Download data, using pooch. These are largely used for testing.

    To view list of downloadable files, look at `DOWNLOADABLE_FILES`.

    This checks whether the data already exists and is unchanged and downloads
    again, if necessary. If dataset_name ends in .tar.gz, this also
    decompresses and extracts the archive, returning the Path to the resulting
    directory. Else, it just returns the Path to the downloaded file.

    """
    if retriever is None:
        raise ImportError("Missing optional dependency 'pooch'."
                          " Please use pip or "
                          "conda to install 'pooch'.")
    if dataset_name.endswith('.tar.gz'):
        processor = pooch.Untar()
    else:
        processor = None
    fname = retriever.fetch(dataset_name,
                            progressbar=True,
                            processor=processor)
    if dataset_name.endswith('.tar.gz'):
        fname = find_shared_directory([pathlib.Path(f) for f in fname])
    else:
        fname = pathlib.Path(fname)
    return fname.as_posix()


@click.command()
def main():
    """Download data.

    By default, this will be in data directory in this repo. To overwrite, set
    FENS_DATA_DIR environment variable, e.g.,

    FENS_DATA_DIR=path/to/data_dir python fetch.py

    """
    fetch_data("allen_478498617.nwb")
    fetch_data("Mouse32-140822.nwb")
    fetch_data("A0634-210617.nwb")
    fetch_data("Achilles_10252013.nwb")

if __name__ == '__main__':
    main()
