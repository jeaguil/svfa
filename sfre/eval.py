import numpy as np
import pandas as pd
import joblib

from sfre import const


class ModelImport(Exception):
    """Checks if model exists before evaluating."""


def mean_absolute_error(predictions, actual_power):
    return 1 / len(predictions) * np.sum(np.abs(actual_power - predictions))


def root_mean_squared_error(predictions, actual_power):
    return np.sqrt(
        1 / len(predictions) *
        np.sum(np.square(np.abs(actual_power - predictions)))
    )


def eval(testing_set):
    try:
        regr = joblib.load("model.pkl")
    except:
        raise ModelImport(
            "Cannot find model.pkl, consider training the model first with 'python3 -m sfre -t' ")

    testing_set["TIMESTAMP"] = pd.to_datetime(testing_set["TIMESTAMP"])
    solar_zones = testing_set.groupby(["ZONEID"])
    model_res = pd.DataFrame(index=["MAE", "RMSE"])

    for i in solar_zones.groups:
        zone_data = solar_zones.get_group(i)
        time_ahead = zone_data["TIMESTAMP"].apply(
            lambda x: x + pd.Timedelta(hours=24)
        )
        actual_power = np.empty(shape=(0, 0), dtype=np.float64)
        for j in time_ahead:
            actual_power = np.append(
                actual_power, zone_data[zone_data["TIMESTAMP"] == j]["POWER"]
            )
        testing_variable_data = zone_data.loc[
            :, ~zone_data.columns.isin(["ZONEID", "TIMESTAMP", "POWER"])
        ]
        testing_variable_data = testing_variable_data.iloc[
            : -(len(testing_variable_data) - len(actual_power))
        ]

        predictions = regr.predict(testing_variable_data)
        model_res[i] = [
            mean_absolute_error(predictions, actual_power),
            root_mean_squared_error(predictions, actual_power),
        ]

    model_res["Overall"] = model_res.mean(axis=1)
    model_res.to_csv(const.OUTFILE_MODEL_RES)
