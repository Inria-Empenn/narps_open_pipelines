from __future__ import annotations

from pathlib import Path

import pandas as pd

# # compute euclidian distance to the indifference line defined by
# # gain twice as big as losses
# % https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
# a = 0.5;
# b = -1;
# c = 0;
# x = onsets{iRun#.gain;
# y = onsets{iRun}.loss;
# dist = abs(a * x + b * y + c) / (a^2 + b^2)^.5;
# onsets{iRun}.EV = dist; % create an "expected value" regressor

def compute_expected_value(onsets: dict[str, list[float]] | pd.DataFrame | str | Path):
    """Compute expected value regressor for a run.
    
    Parameters
    ----------
    onsets : dict[str, list[float]] | pd.DataFrame | str | Path
             Events for a run. 
             Pathlike TSV file with columns 'gain' and 'loss'.
             If a dict, must have keys 'gain' and 'loss'.
             If a DataFrame, must have columns 'gain' and 'loss'.

    Returns
    -------
    onsets : pd.DataFrame
             Onsets with expected value column added.    
    """
    a = 0.5
    b = -1
    c = 0

    if isinstance(onsets, (str, Path)):
        onsets = pd.read_csv(onsets, sep='\t')

    if isinstance(onsets, dict):
        onsets = pd.DataFrame(onsets)

    x = onsets['gain']
    y = onsets['loss']

    dist = abs(a * x + b * y + c) / (a**2 + b**2)**.5
    onsets['EV'] = dist

    return onsets
