import pandas as pd
import numpy as np


class TrainModel(object):
    def __init__(self, training_set, hours_ahead, selected_model):
        self.training_set = training_set

        self.x = self.construct_x(training_set.drop(["POWER"], axis=1))
        self.y = self.construct_y(hours_ahead)

    def construct_x(self, ts):
        pass

    def construct_y(self, hours_ahead):
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
        return res
