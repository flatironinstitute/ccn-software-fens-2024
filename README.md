# ccn-software-fens-2024

Materials for CCN software workshop at FENS 2024

We have a slack channel for communicating with attendees, if you haven't received an invitation, please send us a note!

Note that the rest of this README is for contributors to the workshop. If you are a user, please view the [built website](https://flatironinstitute.github.io/ccn-software-fens-2024/) instead!

## Building the site locally

To build the site locally, clone this repo and install it in a fresh python 3.11 environment (`pip install -e .`). Then run `mkdocs serve` and navigate your browser to `localhost:8000`.

## Adding notebooks

Instead of directly writing tutorials as `.ipynb` files, which are hard to version control and review, we are using [mkdocs-gallery](https://smarie.github.io/mkdocs-gallery/), which converts `.py` scripts into jupyter notebooks when the documentation is built. Additionally, we have an additional file, `scripts/strip_text.py`, which is run by the Github Actions that builds the site (or by binder), but needs to be run manually if you want to see its output locally. This script creates two additional versions of each notebook: a user-facing version, which has much of the text and code removed, only retaining some basic headers, brief pointers, and introductory blocks, and a presenter-facing version, which has notes for the presenter as well as all code blocks. The following will describe the syntax for `mkdocs-gallery` and `strip_text.py`, see also `docs/examples/01_current_injection.py` for an example.

### mkdocs-gallery

- The first markdown block must be contained within the module docstring at the top of the file, i.e., be contained within `"""`. It must start with a single `#` and be the only level 1 header.
- All following markdown blocks start with `# %%` and continue until a line without `#`:

``` python
# %%
# This is a markdown cell.
# 
# This line is part of the same markdown cell.

# But this is just a regular comment, because of the empty line preceding it.
```

- You can use [admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) in the markdown blocks as well, which is helpful. See site for different possible headers.

``` python
# !!! note
#
#     This is an admonition.
```

### strip_text.py

- This script creates two additional copies of each script found in `docs/examples`, placing one of them in `docs/for_users` and the other in `docs/just_code`.
- In general, `strip_text.py` removes the contents of all markdown cells and leaves the code alone, but its behavior is dictated by the file's headers, as well as four special directives:
    1. Any content placed within `<div class="notes"> </div>` will be invisible in the fully-rendered notebook, but present in both the `for_users/` and `just_code/` versions. The idea here is that these contain bullet points summarizing the upcoming content.
    2. If a header is followed by `{.keep-text}`, then the `for_users/` and `just_code/` versions will contain the text found under that header (until we hit the next header with the same level or higher):

    ```python
    # ## Header
    # This text will be removed
    # ### Header {.keep-text}
    # This text is present
    # #### Header
    # This text is present
    # ### Header
    # This text will be removed
    ```
    
    3. If a header is followed by `{.strip-headers}`, then the `for_users/` version will remove all headers beneath it (but not itself) until we hit the next header with the same level or higher:
    
    ```python
    # ## Visible header
    # ### Visible header {.strip-headers}
    # #### Hidden header
    # #### Hidden header
    # ### Visible header
    ```

    4. If a header is followed by `{.strip-code}`, then the `for_users/` version will remove all headers beneath it (but not itself) until we hit the next header with the same level or higher:
    
    ```python
    # ## Visible header
    this_code_present
    # ### Visible header {.strip-code}
    this_code_removed
    # #### Hidden header
    this_code_removed
    # ### Visible header
    this_code_present
    ```

## binder

See [nemos Feb 2024 workshop](https://github.com/flatironinstitute/nemos-workshop-feb-2024) for details on how to set up the Binder

