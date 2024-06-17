# Data

This directory will be populated by NWB files once you run the setup script. See the [setup instructions](https://flatironinstitute.github.io/ccn-software-fens-2024/#setup) for more details.

You can access these files either directly or by using the included `workshop_utils` (assuming you've followed the setup instructions), and you can load them using pynapple:

``` python
import pynapple as nap
import workshop_utils

data = nap.load_file(workshop_utils.fetch_data("allen_478498617.nwb"))
```

We'll explore these datasets during the workshop!
