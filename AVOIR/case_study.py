import matplotlib.pyplot as plt
import os
# os.environ["PATH"] = os.pathsep + os.path.join("..", "mlp_solver")
import pandas as pd
import numpy as np
from tqdm import tqdm
from dsl import spec, SpecTracker
from dsl.grammar import (
    create_variable as V,
    Expectation as E,
    Specification
)
from dsl.tests.database.database import Database
import logging

level = logging.CRITICAL
logger = logging.getLogger()
logger.setLevel(level)
for handler in logger.handlers:
    handler.setLevel(level)

data_fp = './dsl/tests/datasets/data/compas/compas-scores-two-years.csv'
data = pd.read_csv(data_fp)

BLUE   = "#1d70b8"      # rich blue (line, spines, labels)
GREEN  = "#006400"      # dark green (target line)
ALPHA  = 0.15           # CI translucency

def conf_plot_pretty(xs, ys, cs, hline,
                     ylab="Value of the selected node",
                     xlab="# Observed"):
    # wide, short canvas
    fig, ax = plt.subplots(figsize=(16, 10), dpi=120)

    # main curve
    ax.plot(xs, ys, color=BLUE, linewidth=3, zorder=3)

    # confidence band (no red ± curves, just the shaded area)
    ax.fill_between(xs, ys-cs, ys+cs, color=BLUE, alpha=ALPHA, zorder=2)

    # dotted green fairness / target line
    ax.hlines(hline, xs.min(), xs.max(),
              color=GREEN, linewidth=4,
              linestyle=(0, (5, 5)), zorder=1)

    # ­­­—— axes & label styling ———­
    ax.set_xlabel(xlab, fontsize=26, color=BLUE, weight="bold", labelpad=15)
    ax.set_ylabel(ylab, fontsize=26, color=BLUE, weight="bold", labelpad=20)
    ax.tick_params(axis="both", labelsize=18, colors=BLUE)

    # thick left & bottom spines, hide the others
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(4)
        ax.spines[side].set_color(BLUE)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)

    # limits & margin so the left edge is flush
    ax.set_ylim(-2.0, 4.0)
    ax.margins(x=0)

    # OPTIONAL: faint shading for the first 100 points (burn-in look)
    # ax.axvspan(xs.min(), xs.min()+100, color=BLUE, alpha=0.05)

    return fig, ax

def did_or_didnt_recid_in_2_years(df):
    select_attributes = ["id", "age", "c_charge_degree", "race",
                         "age_cat", "score_text", "sex",
                         "priors_count", "days_b_screening_arrest", "decile_score",
                         "is_recid", "two_year_recid", "c_jail_in", "c_jail_out"]
    view = df[select_attributes].copy()
    view = view[(view.days_b_screening_arrest <= 30) & (view.days_b_screening_arrest >= -30)]
    view = view[view.is_recid != -1]
    view = view[view.c_charge_degree != "O"]
    view = view[view.score_text != 'N/A']
    return view

def get_spec():
    return E(V("decile_score" ) > decile_threshold, given=(V("two_year_recid") == 0) & (V("race") == "African-American")) / \
    E(V("decile_score" ) > decile_threshold, given=(V("two_year_recid") == 0) & (V("race") == "Caucasian"))  < threshold_l

def get_spec_north():
    return E(V("two_year_recid") == 0, given=((V("decile_score" ) > decile_threshold) & (V("race") == "Caucasian"))) / \
    E(V("two_year_recid") == 0, given=((V("decile_score" ) > decile_threshold) & (V("race") == "African-American")))  > threshold_l

def get_inverse_spec():
    return E(V("decile_score") > decile_threshold, given=(V("two_year_recid") == 0) & (V("race") == "Caucasian")) / \
        E(V("decile_score") > decile_threshold, given=(V("two_year_recid") == 0) & (V("race") == "African-American")) > threshold_r

def get_spec_or():
     return (E(V("decile_score") > decile_threshold,given=(V("two_year_recid") == 0) & (V("race") == "African-American")) / \
        E(V("decile_score") > decile_threshold,given=(V("two_year_recid") == 0) & (V("race") == "Caucasian"))  < threshold_l) | \
        (E(V("decile_score") > decile_threshold,given=(V("two_year_recid") == 0) & (V("race") == "Caucasian")) / \
        E(V("decile_score") > decile_threshold,given=(V("two_year_recid") == 0) & (V("race") == "African-American")) > threshold_r)
def get_values_for_term(spec_hist, term):
    xs, values = [], []
    for x, dict_vals in spec_hist:
        xs.append(x)
        values.append(dict_vals.get(term.id))
    return xs, values
def conf_plot(xs, ys, cs, hline):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(xs, ys, color="b")
    ax.plot(xs, ys + cs, color="r", linestyle="dashed")
    ax.plot(xs, ys - cs, color="r", linestyle="dashed")
    ax.fill_between(xs, ys-cs, ys+cs, color="b", alpha=0.1)
    ax.hlines(hline, xmin=np.min(xs), xmax=np.max(xs), color="g")
    ax.set_ylim(-1.5, 1.5)
    return fig, ax

if __name__ == "__main__":
    database = Database(data)
    threshold_l = 1.1
    threshold_r = 0.9
    decile_threshold = 3.0
    delta_threshold = 0.1

    spec = get_inverse_spec()
    print(spec.left_child.expectation_term)

    my_spec = get_spec()
    database.create_materialized_view(
        "two_year_recid_results",
        query=did_or_didnt_recid_in_2_years,
        specification=my_spec,
        observation_key="id",
        progress_bar=tqdm,
        spec_args={"include_confidence": True, "delta": delta_threshold, "optimization_frequency": 5, "our_approach": True},
    )

    spec_history = database.views["two_year_recid_results"].observer.spec_val_history

    xs, values = get_values_for_term(spec_history, my_spec)
    deltas = [val.delta for val in values]

    inv_spec = get_inverse_spec()
    print(inv_spec.left_child.expectation_term)
    view_name = "two_year_recid_results_inv"
    database.create_materialized_view(
        view_name,
        query=did_or_didnt_recid_in_2_years,
        specification=inv_spec,
        observation_key="id",
        progress_bar=tqdm,
        spec_args={"include_confidence": True, "delta": delta_threshold, "optimization_frequency": 5, "our_approach": True}
    )

    inv_spec_history = database.views[view_name].observer.spec_val_history

    xs, values = get_values_for_term(inv_spec_history, inv_spec)
    deltas = [val.delta for val in values]

    print(inv_spec.left_child.expectation_term)

    xs, values = get_values_for_term(inv_spec_history, inv_spec.left_child.expectation_term)
    ys, cs = zip(*[(value.val, value.epsilon) for value in values])

    xs, ys, cs = (np.array(t) for t in (xs, ys, cs))

    inv_spec_v = get_inverse_spec()
    print(inv_spec_v.left_child.expectation_term)
    view_name_v = "two_year_recid_results_inv_v"
    database.create_materialized_view(
        view_name_v,
        query=did_or_didnt_recid_in_2_years,
        specification=inv_spec_v,
        observation_key="id",
        progress_bar=tqdm,
        spec_args={"include_confidence": True, "delta": delta_threshold, "optimization_frequency": 5, "our_approach": False}
    )

    inv_spec_history_v = database.views[view_name_v].observer.spec_val_history

    xs_v, values_v = get_values_for_term(inv_spec_history_v, inv_spec_v)
    deltas_v = [val.delta for val in values_v]

    xs_v, values_v = get_values_for_term(inv_spec_history_v, inv_spec_v.left_child.expectation_term)
    ys_v, cs_v = zip(*[(value_v.val, value_v.epsilon) for value_v in values_v])

    xs_v, ys_v, cs_v = (np.array(t) for t in (xs_v, ys_v, cs_v))

    # fig, ax = conf_plot(xs_v, ys_v, cs_v, inv_spec_v.left_child.threshold.val)
    # ax.set_ylim(-2, 4)
    # ax.set_xlabel("# Observed", fontsize=24)
    # ax.set_ylabel("E[hrisk|recid=0 & r=C] / E[hrisk|recid=0 & r=AA]", fontsize=24)
    # ax.tick_params(axis='both', which='major', labelsize=24)

    # fig.savefig("plots/compas-propublica-et.png")
    fig, ax = conf_plot_pretty(xs, ys, cs, inv_spec.left_child.threshold.val)
    fig.tight_layout()
    fig.savefig("plots/compas-propublica-et.png", bbox_inches="tight")   # if you need a file

    print(inv_spec.left_child.expectation_term.left_child)
    xs_l, values_l = get_values_for_term(inv_spec_history, inv_spec.left_child.expectation_term.left_child)
    print(inv_spec.left_child.expectation_term.right_child)
    xs_r, values_r = get_values_for_term(inv_spec_history, inv_spec.left_child.expectation_term.right_child)

    deltas_l = [value.delta for value in values_l]
    deltas_r = [value.delta for value in values_r]
    print(deltas_l[-1], deltas_r[-1])

    print(get_spec_north())

    inv_spec = get_spec_north()
    view_name = "two_year_recid_results_inv"
    database.create_materialized_view(
        view_name,
        query=did_or_didnt_recid_in_2_years,
        specification=inv_spec,
        observation_key="id",
        progress_bar=tqdm,
        spec_args={"include_confidence": True, "delta": delta_threshold, "optimization_frequency": 5, "our_approach": True}
    )

    inv_spec_history = database.views[view_name].observer.spec_val_history

    xs, values = get_values_for_term(inv_spec_history, inv_spec)
    deltas = [val.delta for val in values]

    xs, values = get_values_for_term(inv_spec_history, inv_spec.left_child.expectation_term)
    ys, cs = zip(*[(value.val, value.epsilon) for value in values])

    xs, ys, cs = (np.array(t) for t in (xs, ys, cs))

    # fig, ax = conf_plot(xs, ys, cs, inv_spec.left_child.threshold.val)
    # ax.set_ylim(-2, 4)
    # ax.set_xlabel("# Observed", fontsize=18)
    # ax.set_ylabel("E[recid=0|hrisk & r=C] / E[recid=0|hrisk & r=AA]", fontsize=24)
    # ax.tick_params(axis='both', which='major', labelsize=24)
    fig, ax = conf_plot_pretty(xs, ys, cs, inv_spec.left_child.threshold.val)
    fig.tight_layout()
    fig.savefig("plots/compas-northpointe-et.png", bbox_inches="tight")   # if you need a file
    print(inv_spec.left_child)

    # fig.savefig("plots/compas-northpointe-et.png")
    