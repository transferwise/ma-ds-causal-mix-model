"""
Utils for the WISE project.
"""

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

import pymc as pm

def extract_all_seasonalities(df, seasonalities, trend=True):
    """Extract all seasonal components from a Prophet forecast.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    seasonalities : dict[str, dict[str, number]]
        A dictionary containing the period and fourier order of each custom
        seasonality to be extracted.

    """

    # Create "ds" column from "date" index (required by Prophet)
    df = df.reset_index()#.rename(columns={"date": "ds"})

    # Initialize a Prophet model
    model = Prophet(
        weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False
    )

    # Add each custom seasonality from the dictionary
    for name, settings in seasonalities.items():
        period_days = settings["period"]
        fourier_order = settings["fourier"]
        model.add_seasonality(
            name=name, period=period_days, fourier_order=fourier_order
        )

    # Fit the model to the historical data
    model.fit(df)

    # Create a future dataframe and predict
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)

    # Extract and compile all seasonal components
    components = [df["ds"]]
    components += [forecast[name] for name in seasonalities.keys()]

    if trend:
        components += [forecast["trend"]]

    # Combine into a DataFrame
    result_df = pd.concat(components, axis=1)

# Removed commented-out scaling code as it is not necessary for the current implementation.
    # Set "ds" column as index and rename it to "date"
    result_df = result_df.set_index("ds").rename_axis("date")

    return result_df

def build_and_sample_model(data: pd.DataFrame, formula: str, sample_kwargs=None, debug=False):
    """
    Build and sample a linear model from a formula.
    """
    # Parse the formula to get target and channels
    target, channels = formula.split('~')
    target = target.strip()
    channels = [ch.strip() for ch in channels.split('+') if ch.strip() != "1"]

    # Define coordinates
    coordinates = {"date": data.date.unique()}
    if debug:
        # print coordinates length
        print(f"Coordinates length Date: {len(coordinates['date'])}")
    if channels:  # If there are regressors, include them in coordinates
        coordinates["channel"] = channels
        if debug:
            # print coordinates length
            print(f"Coordinates length Channels: {len(coordinates['channel'])}")

    # Filter the dataset based on the formula
    with pm.Model(coords=coordinates) as linear_model:
        # Load Data in Model
        target_data = pm.Data("target", data[target].values, dims="date")
        if debug:
            # print shape of target_data
            print(target_data.eval().shape)

        # Constant or intercept
        intercept = pm.Gamma("intercept", mu=3, sigma=2)

        mu_var = 0

        if channels:  # If there are regressors, include them
            regressors = pm.Data("regressors", data[channels].values, dims=("date", "channel"))
            if debug:
                # print shape of regressors
                print(regressors.eval().shape)
            gamma = pm.Normal("gamma", mu=1, sigma=.5, dims="channel")
            mu_var += (regressors * gamma).sum(axis=-1) + intercept
        else:
            mu_var += intercept

        if debug:
            # print shape of mu_var
            print(mu_var.eval().shape)

        # Likelihood
        pm.Normal("likelihood", mu=mu_var, sigma=pm.Gamma("sigma", mu=2, sigma=3), observed=target_data, dims="date")

        # Sample
        idata = pm.sample_prior_predictive(random_seed=42)
        
        # Use provided sample_kwargs if available, otherwise use defaults
        if sample_kwargs is None:
            sample_kwargs = {
                "tune": 1000,
                "draws": 500,
                "chains": 4,
                "random_seed": 42,
                "target_accept": 0.9,
                "nuts_sampler": "numpyro",
                "progressbar": False
            }
        
        idata.extend(pm.sample(**sample_kwargs))
        pm.compute_log_likelihood(idata, progressbar=False)
        idata.extend(
            pm.sample_posterior_predictive(idata, random_seed=42)
        )

    return (idata, linear_model)

# Calculate distribution overlap
def calculate_overlap(dist1, dist2, bins=30):
    # Create a common range for both distributions
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    # Create histograms with the same bins
    hist1, bin_edges = np.histogram(dist1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), density=True)
    
    # Calculate the overlap
    overlap = np.sum(np.minimum(hist1, hist2)) * (bin_edges[1] - bin_edges[0])
    return overlap * 100  # Convert to percentage