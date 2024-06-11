![image](assets/banner.jpg)

# Flatiron CCN workshop on neural data analysis

We are excited to see everyone at the Flatiron Center for Computational Neuroscience FENS Satellite event on using open source packages to analyze and visualize neural data! You should have received an email with logistical information, including the schedule and link to the slack channel where we will be communicating the workshop. If you did not receive this email, please let us know!

Over the course of this two-day workshop, we will walk you through the notebooks included on this site in order to demonstrate how to use pynapple, fastplotlib, and NeMoS to analyze and visualize your data!

Before the workshop, please try to follow the [setup](#setup) instructions below to install everything on your personal laptop. Additionally, if you are bringing your own data, please see the section on [converting your data to NWB](#converting-your-data-to-nwb).

The presentations for this workshop can be found [at this page](https://neurorse.flatironinstitute.org/workshops/fens-2024).

## Setup

Before the workshop, please try to complete the following steps. If you are unable to do so, we will be in the hotel, Palais Saschen Coburg room IV, from 4 to 6pm on Saturday, June 22 to help. Please come by!

1. Clone the github repo for this workshop:
   ```shell
   git clone https://github.com/flatironinstitute/ccn-software-fens-2024.git
   ```
2. Create a new python 3.11 virtual environment. If you do not have a preferred way of managing your python virtual environments, we recommend [miniconda](https://docs.anaconda.com/free/miniconda/). After installing it (if you have not done so already), run 
    ```shell
    conda create --name fens2024 pip python=3.11
    ```
3. Activate your new environment: `
    ```shell
    conda activate fens2024
    ```
4. Navigate to the cloned github repo and install the required dependencies. This will install pynapple, fastplotlib, and nemos, as well as jupyter and several other packages.
    ```shell
    cd ccn-software-fens-2024
    pip install .
    ```
5. Run our setup script to download data and prepare the notebooks:
    ```shell
    python scripts/setup.py
    ```
6. Confirm the installation and setup completed correctly by running:
    ```shell
    python scripts/check_setup.py
    ```

If `check_setup.py` tells you setup was successful, then you're good to go. Otherwise, please come to the installation help session in the hotel, Palais Saschen Coburg room IV, on Saturday, so everyone is ready to get started Sunday morning.

## Converting your data to NWB

If you plan to bring your own data to the Monday afternoon session, please convert it to the [NWB format](https://pynwb.readthedocs.io/). It will make it much easier to load it with pynapple. We recommend the following tools to convert : [NeuroConv](https://neuroconv.readthedocs.io/en/main/) or [NWBGuide](https://nwb-guide.readthedocs.io/en/latest/). If you have any questions about NWB, please consider asking them on the [NWB Helpdesk](https://github.com/NeurodataWithoutBorders/helpdesk/discussions).
