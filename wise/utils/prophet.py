import pandas as pd
from prophet import Prophet

# Removed unused imports to clean up the code
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