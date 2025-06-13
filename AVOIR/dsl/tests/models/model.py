from typing import List
from ...dsl import SpecTracker
from ..datasets import Dataset
import numpy as np
import pdb
from collections import namedtuple

TimestampedVal = namedtuple("TimestampedVal", ["value", "timestep", "stale", "prob", "upper", "lower"])

class InvalidTargetError(Exception):
    pass

class InvalidTrainError(Exception):
    pass

class ModelBasedTest:
    model = None
    f_x = None

    def __init__(self, dataset: Dataset):
        valid_targets = np.all([
            self.is_valid_target(y.to_numpy())
            for y in [dataset.train_Xy[1], dataset.test_Xy[1]]
        ])
        if not valid_targets:
            raise InvalidTargetError("Invalid target provided to ModelBasedTest")

        valid_trains = np.all([
            self.is_valid_train(x)
            for x in [dataset.train_Xy[0], dataset.test_Xy[0]]
        ])
        if not valid_trains:
            raise InvalidTrainError("Invalid training data provided to ModelBasedTest")

        self.init_data_and_model(dataset)

    def init_data_and_model(self, dataset: Dataset):
        raise NotImplementedError(
            ("Test must be initialized with: "
            "model, X_test, y_test")
        )

    @property
    def data(self):
        raise NotImplementedError("Must define data to be used during simulation")

    def is_valid_target(cls, values: np.ndarray) -> bool:
        """
        Determines if a given value can be used as a target value by this model
        """
        raise NotImplementedError("Must define what a valid target looks like in a ModelBasedTest")

    def is_valid_train(cls, values) -> bool:
        """
        Determines if the given training data can be utiltlized by this model test
        """
        return True

    def update_spec(self, new_spec):
        raise NotImplementedError("update_spec should be implmented and should set f_x")

    def get_tabular_rep(self):
        spec_obj = SpecTracker.get_spec(self.f_x)
        tabular_rep = spec_obj.tabular_representation()

        for row in tabular_rep:
            new_row = []
            if row["type"] == 'Expectation':
                for t, hist in self.f_x.spec_val_history:
                    hist_val = hist.get(row["id"])
                    new_row.append(TimestampedVal(value=hist_val.val, timestep=t, stale=False, prob=None, upper= hist_val.epsilon + hist_val.val, lower= hist_val.val - hist_val.epsilon))
                row["vals"] = new_row
            elif row["type"] == 'Specification' or row["type"] == 'ETerm':
                for t, hist in self.f_x.spec_val_history:
                    hist_val = hist.get(row["id"])
                    new_row.append(TimestampedVal(value=hist_val.val, timestep=t, stale=False, prob=None, upper= hist_val.val, lower= hist_val.val))
                row["vals"] = new_row
            elif row["repr"] == 'return' or '=' in row["repr"]:
                for i in range (0, len(row["vals"]), 10):
                    new_row.append(TimestampedVal(value=row["vals"][i].value, timestep=row["vals"][i].timestep, stale=False, prob=None, upper= row["vals"][i].value, lower= row["vals"][i].value))
                row["vals"] = new_row

                    

        # NOTTODO: for each row in tabular rep, find the bounded observations of the spec_boj
        # TODO: Method 2: for each term in tabular rep, update the term to include the value from self.f_x.spec_val_history
        #def get_values_for_term(spec_hist, term):
        #    """This function takes the specification history and a particular term as input and returns two outputs
         #       `xs`: The number of observations seen so far (at a.point in history)
          #      `values`: The values at that point in history. Each value will have an associated `val`, `epsilon` and `delta`.
           # """
           # xs, values = [], []
            #for x, dict_vals in spec_hist:
             #   xs.append(x)
              #  values.append(dict_vals.get(term.id))
            #return xs, values
        # update all TimestampedValue in tabular_replist using epsilon from self.f_x.spec_val_history
        # similar to how it is done in get_values_For_term
        # for example, 
        #  time_0, hist_vals_0 = self.f_x.spec_val_history[0]
        # hist_vals_0.get(tabular_rep[0...n-1]["id"]).epsilon, .delta, .val
        return tabular_rep

    def eval_spec(self):
        SpecTracker.get_spec(self.f_x).eval()

    def predict(self, x):
        return self.model.predict(x)
    
    def predict_list(self, input_list):
        outputs = []
        for item in input_list:
            outputs.append(self.predict(item))
        return outputs
                         
    
    def run_eval_loop(self, progress_bar=None):
        iteration_data = self.data
        if progress_bar:
            iteration_data = progress_bar(iteration_data)
        for datum in iteration_data:
            self.f_x(**datum)
            
