"""Module to compute consensus from six base FLPEs

Runs on a single reach or set of reaches and requires JSON data for reach retrieved by
AWS Batch index.
"""

import argparse as ap
from pathlib import Path
import json
from netCDF4 import Dataset, chartostring, num2date
import numpy as np
import os
import datetime


ALGO_METADATA = {
    'momma': {
        'qvar':'Q',
        'time':'time_str'
    },
    'metroman':{
        'qvar':'average/allq',
        'time':'time_str'
    },
    'hivdi': {
        'qvar':'reach/Q',
        'time':'time_str'
    },
    'sic4dvar':{
        'qvar':'Q_da',
        'time':'times'
    },
    'busboi':{
        'qvar':'q/q',
        'time':'time'
    },
   'sad':{
       'qvar':'Qa',
       'time':'time_str'
   },
}

FILL_VALUE = -999999999999.0
FILL_VALUE_STR = "no_data"



def normalize_time(var):
    """
    Normalize different algorithm time formats to
    'YYYY-MM-DDTHH:MM:SSZ'
    """

    t = var[:]

    # Character array: (nt, nchars) -> (nt,)
    if t.ndim == 2 and t.dtype.kind in ("S", "U"):
        t = chartostring(t)

    # String/object time
    if t.dtype.kind in ("S", "U", "O"):

        result = []

        for x in t:

            if np.ma.is_masked(x) or x is None:
                result.append(None)
                continue

            if isinstance(x, bytes):
                x = x.decode()

            x = str(x).strip()

            # Handle missing / fill-value time strings
            if x in ("", "no_data", "--", "nan", "None"):
                result.append(None)
                continue

            # Standardize to YYYY-MM-DDTHH:MM:SSZ
            x = x.rstrip("Z")

            if "." in x:
                x = x.split(".")[0]

            result.append(x + "Z")

        return np.asarray(result, dtype=object)


    # Numeric time with CF-style units
    if np.issubdtype(t.dtype, np.number):

        units = getattr(var, "units", None)
        calendar = getattr(var, "calendar", "standard")

        if units is None:
            raise ValueError(
                "Numeric time variable has no 'units' attribute"
            )

        dates = num2date(
            t,
            units=units,
            calendar=calendar
        )

        result = []

        for x, d in zip(t, dates):

            if np.ma.is_masked(x):
                result.append(None)
            else:
                result.append(
                    d.strftime("%Y-%m-%dT%H:%M:%SZ")
                )

        return np.asarray(result, dtype=object)

    raise ValueError(
        f"Unsupported time format: "
        f"shape={t.shape}, dtype={t.dtype}"
    )



def remove_rf_bad_and_recalc_consensus(reach_id, arrs, time_arrs, included_algos, rf_data, selected_metric):
    """
    For a list of discharge arrays:
    - Removes arrays with poor performance (prediction = 0) by Random Forest for selected metric
    - Recalculates consensus using the remaining arrays

    Parameters
    ----------
    reach_id : int
        SWORD reach ID.
        
    arrs : list of np.ndarray
        Discharge arrays from each algorithm.

    time_arrs : list of np.ndarray
        Time arrays corresponding to each discharge array.
        
    included_algos : list of str
        Names of the available discharge algorithms.

    rf_data : dict
        Random Forest prediction data.

    selected_metric : str
        Metric used to filter algorithms
        (e.g., "nBIAS_binary").

    Returns
    -------
    np.ndarray
        Cleaned and recalculated consensus array.
    """

    rf_arrs = []
    rf_included_algos = []
    rf_time_arrs = []


    # -------------------------------------------------
    # Find reach
    # -------------------------------------------------
    reach_idx = np.where(rf_data["reach_ids"] == reach_id)[0]

    if len(reach_idx) == 0:
        print(f"Reach {reach_id} not found in RF predictions.")
        return (
            np.full_like(arrs[0], np.nan),
            np.full_like(arrs[0], "no_data", dtype=object),
            []
        )

    reach_idx = reach_idx[0]


    # -------------------------------------------------
    # Find selected metric
    # -------------------------------------------------
    metric_idx = np.where(rf_data["metrics"] == selected_metric)[0]

    if len(metric_idx) == 0:
        raise ValueError(
            f"Metric '{selected_metric}' not found in RF predictions."
        )

    metric_idx = metric_idx[0]


    # -------------------------------------------------
    # Keep algorithms with RF prediction = 1
    # -------------------------------------------------
    for i, arr in enumerate(arrs):

        algo = included_algos[i]

        algo_idx = np.where(
            np.char.lower(rf_data["algorithms"].astype(str))
            == algo.lower()
        )[0]

        if len(algo_idx) == 0:
            print(f"Algorithm {algo} not found in RF predictions.")
            continue

        algo_idx = algo_idx[0]

        pred = rf_data["predictions"][
            reach_idx,
            algo_idx,
            metric_idx
        ]

        print(
            f"  {algo}: {selected_metric} prediction = {pred}"
        )

        if pred == 1:
            rf_arrs.append(arr)
            rf_included_algos.append(included_algos[i])
            rf_time_arrs.append(time_arrs[i])


    # -------------------------------------------------
    # No algorithms remain after RF filtering
    # -------------------------------------------------
    if not len(rf_arrs):
        print(
            f"All algorithms removed by RF for reach {reach_id} "
            f"using {selected_metric}."
        )

        return (
            np.full_like(arrs[0], np.nan),
            np.full_like(arrs[0], "no_data", dtype=object),
            []
        )


    # -------------------------------------------------
    # Compute consensus
    # -------------------------------------------------
    # consensus_arr = np.nanmedian(
    consensus_arr = np.nanmean(
        np.stack(rf_arrs, axis=0),
        axis=0
    )

    selected_time_arr = rf_time_arrs[0]

    return consensus_arr, selected_time_arr, rf_included_algos


def process_reach(reach_id, mntdir, rf_data, selected_metric):
    """
    Compute consensus for a single reach.

    Parameters
    ----------
    mntdir: Path
        path to base mount directory
    reach_id: int
        ID of reach to process
    """

    print('reach', reach_id)
    included_algos = []
    arrs = []
    time_arrs = []

    for algo, metadata in ALGO_METADATA.items():
        infile = mntdir / 'flpe' / algo / f'{reach_id}_{algo}.nc'
        
        if not os.path.exists(infile):
            continue
            
        try:
            with Dataset(infile, 'r') as ds:
                try:
                    arr = ds[metadata['qvar']][:]
                except (KeyError, IndexError) as e:

                    print(f"  Skipping {algo} for reach {reach_id}: "
                          f"Q variable could not be read ({e})")
                    continue


                # Convert masked array to NaN
                if np.ma.isMaskedArray(arr):
                    arr = arr.filled(np.nan)

                arr = np.asarray(arr, dtype=float).squeeze()


                # Invalid / empty algorithm output
                if arr.ndim != 1 or arr.size <= 1:

                    print(f"  Skipping {algo} for reach {reach_id}: "
                          f"invalid Q shape {arr.shape}")
                    continue


                # Read time variable
                time_var_name = metadata['time']

                if time_var_name not in ds.variables:
                    print(f"  Skipping {algo} for reach {reach_id}: "
                          f"time variable '{time_var_name}' not found")
                    continue


                # Normalize time
                try:
                    time = normalize_time(ds.variables[time_var_name])

                except Exception as e:
                    print(f"  Skipping {algo} for reach {reach_id}: "
                          f"could not decode time ({e})")
                    continue


                # Check Q/time consistency
                if len(time) != len(arr):
                    print(f"  Skipping {algo} for reach {reach_id}: "
                          f"Q/time length mismatch (Q={len(arr)}, time={len(time)})")
                    continue


                # Original Q validity checks
                # treat negative discharge as NaN
                arr[arr < 0] = np.nan

                # ignore algos with no nonnegative discharge
                if not np.any(arr >= 0):
                    continue


                # Everything is valid
                arrs.append(arr)
                time_arrs.append(time)
                included_algos.append(algo)


        except (IOError, OSError, KeyError) as e:
            print(f"  Skipping {algo} for reach {reach_id}: {e}")
            continue

    if not len(arrs):
        print(f"No data for reach '{reach_id}'")
        return

    # Ensure all arrays are the same length — drop any that don't match the majority
    if len(arrs) > 1:
        lengths = [len(a) for a in arrs]
        most_common_len = max(set(lengths), key=lengths.count)
        keep = [i for i, a in enumerate(arrs) if len(a) == most_common_len]
        if len(keep) < len(arrs):
            dropped = [included_algos[i] for i in range(len(arrs)) if i not in keep]
            print(f"  Dropping {dropped} for reach {reach_id}: length mismatch")
            arrs           = [arrs[i] for i in keep]
            time_arrs      = [time_arrs[i] for i in keep]
            included_algos = [included_algos[i] for i in keep]

    consensus_arr, time_arr, included_algos = remove_rf_bad_and_recalc_consensus(
        reach_id=reach_id,
        arrs=arrs, 
        time_arrs=time_arrs, 
        included_algos=included_algos,
        rf_data=rf_data,
        selected_metric=selected_metric
    )

    # Build nc file
    outdir = mntdir / 'flpe' / 'consensus'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    outfile = outdir / f'{reach_id}_consensus.nc'

    with Dataset(outfile, 'w', format="NETCDF4") as dsout:
        dsout.n_algos = str(len(included_algos))
        dsout.contributing_algos = included_algos

        # Add consensus Q
        dsout.createDimension("nt", len(consensus_arr))
        consensus_q = dsout.createVariable("consensus_q", "f8", ("nt",), fill_value=FILL_VALUE)
        consensus_q.long_name = 'consensus discharge'
        consensus_q.short_name = "discharge_consensus"
        consensus_q.tag_basic_expert = "Basic"
        consensus_q.units = "m^3/s"
        consensus_q.valid_min = -10000000.0
        consensus_q.valid_max = 10000000.0
        consensus_q.comment = "Discharge from the consensus discharge algorithm."

        # Add consensus time_str
        consensus_time_str = dsout.createVariable("time_str", str, ("nt",), fill_value="no_data")
        consensus_time_str.long_name = "time (UTC)"
        consensus_time_str.standard_name = "time"
        consensus_time_str.short_name = "time_string"
        consensus_time_str.calendar = "gregorian"
        consensus_time_str.tag_basic_expert = "Basic"
        consensus_time_str.comment = (
            "Time string giving UTC time. The format is YYYY-MM-DDThh:mm:ssZ, "
            "where the Z suffix indicates UTC time."
        )

        # Fill as needed
        consensus_arr_filled = np.where(np.isnan(consensus_arr), FILL_VALUE, consensus_arr)
        time_arr_filled = [t if t is not None else FILL_VALUE_STR for t in time_arr]

        # Write values
        consensus_q[:] = consensus_arr_filled
        consensus_time_str[:] = np.array(time_arr_filled, dtype="O")


def run_consensus(mntdir, indices, reachfile):
    """
    Run consensus algorithm on a set of reaches.

    Parameters
    ----------
    mntdir: Path
        path to base mount directory
    indices: list
        offsets of reaches to process
    """

    with open(reachfile, 'r') as fp:
        reaches = json.load(fp)
        reach_ids = [reaches[i]['reach_id'] for i in indices]

    
    rf_file = next((mntdir / 'rf' ).glob("RF_binary_pred_sword_*.nc"))

    with Dataset(rf_file, "r") as ds:

        rf_data = {
            "reach_ids": ds.variables["reach_id"][:],
            "algorithms": ds.variables["algorithm"][:],
            "metrics": ds.variables["metric"][:],
            "predictions": ds.variables["prediction"][:]
        }

    selected_metric = "KGE_binary" 
    # "NSE_binary"
    # "KGE_binary"
    # "Pearson_r_binary"
    # "nBIAS_binary"

    for reach_id in reach_ids:
        process_reach(reach_id, mntdir, rf_data, selected_metric)


def parse_range(index_str):
    """Parse a range string into a list of integers."""

    indices = []
    try:
        for part in index_str.strip().split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                indices.extend(list(range(int(start), int(end) + 1)))
            else:
                indices.append(int(part))
    except (IndexError, ValueError, TypeError):
        print(f"cannot parse range string: '{index_str}'. Must be either a single integer, "
              f"a range such as 1-100, or a comma separated list of integers and/or ranges")

    return sorted(list(set(indices)))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mntdir", type=str, default="/mnt", help="Mount directory.")
    parser.add_argument("-i", "--index", type=parse_range, required=True)
    parser.add_argument("-r", "--reachfile", type=str, default="reaches.json", help="Reach JSON file.")
    args = parser.parse_args()
    run_consensus(Path(args.mntdir), args.index, args.reachfile)
