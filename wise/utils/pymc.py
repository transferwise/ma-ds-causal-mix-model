import pymc as pm
import pandas as pd

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