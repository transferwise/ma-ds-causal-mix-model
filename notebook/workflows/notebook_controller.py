"""
Notebook Controller Module
--------------------------

This module provides a command-line interface to execute Jupyter notebooks through papermill,
allowing parameter injection into the notebook.

Functions
---------
parse_cli()
    Parse command-line arguments for notebook execution.

to_dict(kvs)
    Convert a list of strings in key=value format into a dictionary with evaluated types.

main()
    Main entry point for the notebook execution process.
"""

import argparse
from ast import literal_eval

import papermill as pm


def parse_cli():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace containing input notebook path ('in_nb'), output notebook path ('out_nb')
        and a list of parameter strings ('param') in key=value format.
    """
    p = argparse.ArgumentParser(description="Run a notebook with CLI-supplied parameters.")
    p.add_argument("in_nb", help="Template notebook (input)")
    p.add_argument("out_nb", help="Executed notebook (output)")
    p.add_argument(
        "-p", "--param",
        action="append",
        metavar="k=v",
        default=[],
        help="Parameter to inject (can be repeated)",
    )
    return p.parse_args()


def to_dict(kvs):
    """
    Convert a list of key=value strings to a dictionary with evaluated values.

    Parameters
    ----------
    kvs : list of str
        List of strings where each string is formatted as 'key=value'.

    Returns
    -------
    dict
        Dictionary with keys and their corresponding evaluated values if possible;
        otherwise the value remains as a string.
    """
    out = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        try:
            out[k] = literal_eval(v)  # Convert string to corresponding Python literal
        except Exception:
            out[k] = v  # Remain as string if evaluation fails
    return out


def main():
    """
    Execute the notebook with injected parameters.

    This function performs the following steps:
      1. Parses CLI arguments.
      2. Converts parameter strings to a dictionary.
      3. Executes the input notebook using papermill with the provided parameters and a timeout.
    """
    args = parse_cli()
    params = to_dict(args.param)
    pm.execute_notebook(args.in_nb, args.out_nb, parameters=params, timeout=30_000)


if __name__ == "__main__":
    main()