# CCN software workshop at FENS 2024

WRITE INTRO TEXT HERE

## Setup

Before the workshop, please try to complete the following steps. If you are unable to do so, we will be in the hotel, Palais Saschen Coburg room IV, from 4 to 6pm on Saturday, June 22 to help. Please come by!

1. Clone the github repo for this workshop: 

`git clone https://github.com/flatironinstitute/ccn-software-fens-2024.git`

2. Create a new python 3.11 virtual environment. If you do not have a preferred way of managing your python virtual environments, we recommend [miniconda](https://docs.anaconda.com/free/miniconda/). After installing it (if you have not done so already), run `conda create --name fens2024 pip python=3.11`.
3. Activate your new environment: `conda activate fens2024`
4. Navigate to the cloned github repo and install the required dependencies. This will install pynapple, fastplotlib, and nemos, as well as jupyter and several other packages.
    ```shell
    cd ccn-software-fens-2024
    pip install .
    ```
5. Run our setup script to download data and prepare the notebooks: `python scripts/setup.py`.
6. Confirm the installation and setup completed correctly by running `python scripts/check_setup.py`.

If `check_setup.py` tells you setup was successful, then you're good to go. Otherwise, please come to the installation help session in the hotel, Palais Saschen Coburg room IV, on Saturday, so everyone is ready to get started Sunday morning.
