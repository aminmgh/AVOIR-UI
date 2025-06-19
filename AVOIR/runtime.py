from typing import ClassVar
import streamlit as st
from stqdm import stqdm
from dsl import spec, SpecTracker
import seaborn as sns
import matplotlib.pyplot as plt
from dsl.grammar import (
    create_variable as V,
    Expectation as E,
    RETURN_VARIABLE as r
)
from dsl.grammar import Specification
from dsl.visualization import add_vega_chart, create_viz_spec
from dsl.tests.datasets import get_dataset_names, get_dataset_attr_list, get_dataset_attr_dict, Dataset, get_dataset
from dsl.tests.models import get_models, get_model, ModelBasedTest, InvalidTargetError, InvalidTrainError, _view_maintenance
import pandas as pd
import time
import numpy as np

class CustomDatasetModel(ModelBasedTest):
    def __init__(self, dataset, output_var, input_vars, attrs, model: ClassVar[ModelBasedTest]):
        self.output_var = output_var
        self.input_vars = input_vars
        self.attrs = attrs
        self.dataset: Dataset = Dataset(dataset.train, dataset.test, attrs, target=output_var, inputs=input_vars)

        self.model = model(self.dataset)

    def update_spec(self, new_spec):
        if not isinstance(new_spec, Specification):
            raise ValueError("Input provided was not parsed into a specification")

        @spec(new_spec, include_confidence=True)
        def decision_func(**kwargs):
            x = kwargs.get("x")
            # prediction = self.model.model.predict(x.reshape(1, -1)) # TODO??
            prediction = self.model.predict(x.reshape(1, -1))
            return prediction[0]
        self.f_x = decision_func

    @property
    def data(self):
        test_data = []
        for _, d in self.dataset.test.iterrows():
            x = d.loc[self.dataset.inputs].to_numpy()
            data_dict = {
                "x": x
            }
            for ind, attr in enumerate(self.dataset.attributes_dict.keys()):
                data_dict[attr] = d[attr]
            test_data.append(data_dict)
        return test_data


def provide_model_eval_interface(model_obj, spec_code):
    model_obj.update_spec(eval(spec_code))  # TODO DANGEROUS
    start_time = time.perf_counter()
    model_obj.run_eval_loop()
    end_time = time.perf_counter()   
    elapsed_time = end_time - start_time
    return elapsed_time
    

if __name__ == "__main__":
    runtimes = {}
    datasets = ["Compas", "Adult Income"]
    fairness_metrics = ["Demographic Parity", "Equalized Odds", "Equal Opportunity"]
    thresholds = np.arange(0.01, 2.25, 0.4)
    model = "Linear Regression"
    fairness_specs_templates = {
        "Demographic Parity": 'E(r, given=(V("{group1_attr}") == {group1_value})) / E(r, given=(V("{group2_attr}") == {group2_value})) < {threshold}',
        "Equalized Odds": '(E(r, given=(V("{group1_attr}") == {group1_value}) & (V("{label_attr}") == {label_value})) / E(r, given=(V("{group2_attr}") == {group2_value}) & (V("{label_attr}") == {label_value})) < {threshold}) & (E(r, given=(V("{group1_attr}") == {group1_value}) & (V("{label_attr}") == {label_value_not})) / E(r, given=(V("{group2_attr}") == {group2_value}) & (V("{label_attr}") == {label_value_not})) < {threshold})',
        "Equal Opportunity": 'E(r, given=(V("{group1_attr}") == {group1_value}) & (V("{label_attr}") == {label_value})) / E(r, given=(V("{group2_attr}") == {group2_value}) & (V("{label_attr}") == {label_value})) > {threshold}',
    }
    dataset_attributes = {
        "Adult Income": {
            "group1_attr": "sex_Male",
            "group1_value": 1,
            "group2_attr": "sex_Female",
            "group2_value": 1,
            "label_attr": "high_income",
            "label_value": 1,
            "label_value_not": 0
        },
        "Compas": {
            "group1_attr": "race_African-American",
            "group1_value": 1,
            "group2_attr": "race_Caucasian",
            "group2_value": 1,
            "label_attr": "two_year_recid",
            "label_value": 1,
            "label_value_not": 0
        },
    }
    # Decide how many permutations to do for COMPAS
    num_permutations = 10  # For example

    # ------------------------------------------------------------------
    # First, handle Adult Income *without* permutations
    # ------------------------------------------------------------------
    adult_ds = get_dataset("Adult Income")
    adult_output_var = adult_ds.target
    adult_input_vars = adult_ds.attributes

    for metric in fairness_metrics:
        # We'll store once in runtimes[("Adult Income", metric)] ...
        # Each entry is a list of runtimes in the same order as thresholds
        runtimes[("Adult Income", metric)] = []

        for th in thresholds:
            # Prepare the specification for Adult
            spec_code = fairness_specs_templates[metric].format(
                **dataset_attributes["Adult Income"], threshold=th
            )
            # Build the model
            adult_model_obj = CustomDatasetModel(
                adult_ds,
                adult_output_var,
                adult_input_vars,
                adult_input_vars,
                get_model(model)
            )
            # Evaluate
            rt = provide_model_eval_interface(adult_model_obj, spec_code)
            runtimes[("Adult Income", metric)].append(rt)

    # ------------------------------------------------------------------
    # Then, handle COMPAS with permutations
    # ------------------------------------------------------------------
    original_compas = get_dataset("Compas")  # just as reference if needed

    for perm_idx in range(num_permutations):
        # We'll create a fresh copy of the COMPAS dataset
        compas_ds = get_dataset("Compas")

        combined = pd.concat([compas_ds.train, compas_ds.test], ignore_index=True)
        combined = combined.sample(frac=1).reset_index(drop=True)  # Shuffle entire data
        # Now re-split: for example, keep the same ratio of train/test
        n_train = len(compas_ds.train)
        compas_ds.train = combined.iloc[:n_train].reset_index(drop=True)
        compas_ds.test = combined.iloc[n_train:].reset_index(drop=True)
        compas_output_var = compas_ds.target
        compas_input_vars = compas_ds.attributes

        for metric in fairness_metrics:
            # We store results in runtimes[("Compas", metric, perm_idx)]
            # each entry is a list of runtimes in the same order as thresholds
            runtimes[("Compas", metric, perm_idx)] = []

            for th in thresholds:
                spec_code = fairness_specs_templates[metric].format(
                    **dataset_attributes["Compas"], threshold=th
                )
                model_obj = CustomDatasetModel(
                    compas_ds,
                    compas_output_var,
                    compas_input_vars,
                    compas_input_vars,
                    get_model(model)
                )
                rt = provide_model_eval_interface(model_obj, spec_code)
                runtimes[("Compas", metric, perm_idx)].append(rt)

    # ------------------------------------------------------------------
    # Now produce multiple plots: 
    # One figure per permutation, comparing Adult Income vs that COMPAS permutation
    # across *all* fairness metrics. 
    # 
    # If you prefer separate plots per fairness metric, you can nest further.
    # ------------------------------------------------------------------
    for perm_idx in range(num_permutations):
        # Create a new figure for this permutation
        plt.figure(figsize=(10, 6))
        plt.title(f"Runtimes of Different Fairness Metrics across Adult Income and Compas")

        # We’ll offset multiple fairness metrics in one figure
        # or we can do separate subplots. Here let's just do multiple lines.
        color_map = plt.cm.tab10(np.linspace(0, 1, len(fairness_metrics) * 2))
        color_idx = 0
        
        for metric in fairness_metrics:
            # Get adult times
            adult_times = runtimes[("Adult Income", metric)]
            # Get compas perm times
            compas_times = runtimes[("Compas", metric, perm_idx)]

            # Plot Adult
            plt.plot(
                thresholds, 
                adult_times,
                marker='o',
                label=f"[Adult] {metric}",
                color=color_map[color_idx],
            )
            color_idx += 1
            
            # Plot Compas
            plt.plot(
                thresholds, 
                compas_times,
                marker='o',
                label=f"[Compas] {metric}",
                color=color_map[color_idx],
            )
            color_idx += 1

        plt.xlabel('Threshold')
        plt.ylabel('Time (s)')
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # If you also want an "overall" single figure for all permutations, 
    # just add more plotting logic below as you prefer.
    # ------------------------------------------------------------------



    

