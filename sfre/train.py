from functools import wraps
import time
import pandas as pd
import numpy as np

from sfre import const


def timer_fn(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        before = time.time()
        fn(*args, **kwargs)
        print("{} took: {} seconds".format(
            fn.__name__.upper(), time.time() - before))

    return wrapper


class TrainModel(object):
    def __init__(self, training_set, hours_ahead):
        self.training_set = training_set

        self.training_time_Df = self.construct_time_Df(hours_ahead)

        self.y = self.construct_y()

        self.x = self.construct_x(
            training_set.drop(["POWER"], axis=1))

    def construct_x(self, ts):
        diff = ts.shape[0] - self.y.shape[0]
        x = ts.copy()
        x.drop(x.tail(diff).index, inplace=True)
        return x

    def construct_y(self):
        ts_cp = self.training_set.copy()
        tt_df_cp = self.training_time_Df["TIMESTAMP"].copy()
        tt_df_cp = tt_df_cp.astype({"TIMESTAMP": str})
        tt_l = tt_df_cp.values.tolist()
        res = ts_cp.loc[ts_cp["TIMESTAMP"].isin(tt_l)]
        res_power = res["POWER"].reset_index()
        return res_power.drop(["index"], axis=1)

    def construct_time_Df(self, hours_ahead):
        res = []
        ts_copy = self.training_set.copy()
        all_solar_zones = ts_copy.groupby(["ZONEID"])
        for i in all_solar_zones.groups:
            indiv_solar_zone = all_solar_zones.get_group(i)
            ts_modified = indiv_solar_zone.astype({"TIMESTAMP": np.datetime64})
            ts_time_col = ts_modified["TIMESTAMP"]
            indev_yts = ts_time_col.apply(
                lambda x: x + pd.Timedelta(hours_ahead, "h"))
            indev_yts = pd.merge(indev_yts, ts_time_col, on=[
                "TIMESTAMP"], how="left", indicator="Exists")
            indev_yts = indev_yts[indev_yts["Exists"] == "both"]
            indev_yts = indev_yts.drop_duplicates()
            indev_yts = indev_yts.reset_index()
            res.append(indev_yts)

        res = pd.concat(res, ignore_index=True)
        res = res.drop(["index", "Exists"], axis=1)
        return res

    def output_training_Y_df(self):
        self.y.to_csv(const.OUTFILE_TRAIN_Y)

    def output_training_X_df(self):
        self.x.to_csv(const.OUTFILE_TRAIN_X)

    @timer_fn
    def train(self):
        """Train model."""
        pass
