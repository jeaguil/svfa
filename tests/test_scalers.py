""" 
 Compares the effect of different scalars on the weather data vs power output.
 
 Find best preprocessing scalar to be used for training the model.

 User guide from: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
 """

import logging
import pandas as pd
import numpy as np

from pathlib import Path

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from utils import check_for_training_files, setup_cli_logging


static_root_parent_path = Path(__file__).resolve().parent.parent
training_data_csv = static_root_parent_path / "data/training_set.csv"
training_df = pd.read_csv(training_data_csv)
feature_names = list(training_df.columns)

feature_mapping = {
    "VAR78": "Total column liquid water(tclw) measured in kg m^-2",
    "VAR79": "Total coluimn ice water (tciw) measured in kg m^-2",
    "VAR134": "Surface pressure (SP) measure in Pa",
    "VAR157": "Relative humidity at 1000 mbar (r) measured in %",
    "VAR164": "Total cloud cover (TCC) measured 0-1",
    "VAR165": "10-metre U wind component (10u) measured in m s^-1",
    "VAR166": "10-metre V wind component (10v) measured in m s^-1",
    "VAR167": "2-metre temperature (2T) measured in K",
    "VAR169": "Surface solar rad down (SSRD) measured in J m^-2",
    "VAR175": "Surface thermal rad down (STRD) measured in J m^-2",
    "VAR178": "Top net solar rad (TSR) measured in J m^-2",
    "VAR228": "Total precipitation (TP) measured in m",
    "POWER": "Solar farm accumulated POWER",
}


class DfReadingError(BaseException):
    pass


def _cmp_scalars():

    # check if training sets are available
    check_for_training_files()

    # Take only 2 features to make visualization easier.
    # Features [Any] for distributions.
    features = ["VAR78", "VAR79"]
    feature_idx = [feature_names.index(feature) for feature in features]
    _X_full = training_df.iloc[:, feature_idx]

    distributions = [
        ("Unscaled data", _X_full),
        ("Data after standard scaling",
         StandardScaler().fit_transform(_X_full)),
        ("Data after min-max scaling", MinMaxScaler().fit_transform(_X_full)),
        ("Data after max-abs scaling", MaxAbsScaler().fit_transform(_X_full)),
        (
            "Data after robust scaling",
            RobustScaler(quantile_range=(25, 75)
                         ).fit_transform(_X_full),
        ),
        (
            "Data after power transformation (Yeo-Johnson)",
            PowerTransformer(
                method="yeo-johnson").fit_transform(_X_full),
        ),
        (
            "Data after quantile transformation (uniform pdf)",
            QuantileTransformer(
                output_distribution="uniform").fit_transform(_X_full),
        ),
        (
            "Data after quantile transformation (guassian pdf)",
            QuantileTransformer(
                output_distribution="normal").fit_transform(_X_full),
        ),
    ]

    for i, v in enumerate(distributions, 1):
        try:
            _make_plot(i, distributions, features)
            plt.savefig("./scalars_imgs/" + v[0] + ".png")
        except IndexError:
            pass


def _create_axes(title, figsize=(16, 6)):
    # Code from sklearn user guide of comparing scalar

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # axes for first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # axes for second plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.5, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
    )


def _plot_distribution(axes, X, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    ax.scatter(X[:, 0], X[:, 1], alpha=0.5,
               marker="o", s=5, lw=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")

    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")


def _make_plot(item_indx, distributions, features):
    title, X = distributions[item_indx]
    ax_zoom_out, ax_zoom_in = _create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)

    # PLot full data at first plot
    _plot_distribution(
        axarr[0],
        X,
        hist_nbins=200,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Full data",
    )

    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outlies_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    )
    _plot_distribution(
        axarr[1],
        X[non_outlies_mask],
        hist_nbins=50,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Zoom-in",
    )


if __name__ == "__main__":
    setup_cli_logging()

    _cmp_scalars()

    logging.info("\tScalar tests saved in scalar_tests folder")
