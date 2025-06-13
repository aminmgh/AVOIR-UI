from typing import List, Set, Dict
from collections import namedtuple
import numpy as np
import uuid
from statistics import StatisticsError
import pdb

from .errors import NoObservedOutcomesError

TimestampedVal = namedtuple("TimestampedVal", ["value", "timestep", "stale", "prob", "upper", "lower"]) #TODO: Add epsilon here


class Expression:
    def eval(self, *args, **kwargs):
        raise NotImplementedError("Expressions must implement eval")

    def eval_bounded_at_delta(self, delta, call_id=None):
        raise NotImplementedError("")


class HistoryLoggingExpression(Expression):

    def __init__(self):
        self.history = []
        self.active_value_index = {}
        self.call_ids = set()

    def active_value(self, with_value_key):
        if with_value_key not in self.active_value_index:
            return None
        if self.active_value_index[with_value_key] is None:
            return None

        return self.history[self.active_value_index[with_value_key]].value

    @property
    def fresh_values(self) -> List[TimestampedVal]:
        return [self.history[value_idx]
                for value_idx in self.active_value_index.values()
                if value_idx is not None]

    def retire_values(self, with_value_key):
        if with_value_key not in self.active_value_index or with_value_key is None:
            return

        value_index = self.active_value_index[with_value_key]

        if value_index is None:
            return

        old_value = self.history[value_index]

        self.history[value_index] = TimestampedVal(
            value=old_value.value,
            timestep=old_value.timestep,
            stale=True,
            prob=old_value.prob,
            upper=old_value.upper,
            lower=old_value.lower
        ) 
        self.active_value_index[with_value_key] = None

    def add_active_value(self, tsv: TimestampedVal, value_key):
        self.active_value_index[value_key] = len(self.history)
        self.history.append(tsv)

    def edit_active_value(self, tsv_callback, value_key):
        try:
            value_idx = self.active_value_index[value_key]
        except KeyError:
            raise KeyError(f"Please provide a value key for an active value. Value key provided: {value_key}")

        active_value = self.history[value_idx]
        self.history[value_idx] = tsv_callback(active_value)

    def eval_cached(self, call_id=None):
        """
        Returns the most value of the recent evaluation since `call_id` if one
        exists. Otherwise, return a fresh evaluatation
        """
        fresh = self.fresh_values
        if len(fresh) == 0 or call_id is None:
            return self.eval(call_id)

        most_recent = self.fresh_values[-1]

        return most_recent.value if most_recent.timestep >= call_id and call_id is not None else self.eval(call_id)

    @staticmethod
    def cached_eval(instance_func):
        def wrapper(self, *args, **kwargs):
            assert isinstance(self, HistoryLoggingExpression)

            if "call_id" not in kwargs:
                return instance_func(self, *args, **kwargs)

            call_id = kwargs["call_id"]
            fresh = self.fresh_values
            if len(fresh) == 0 or call_id is None:
                return instance_func(self, *args, **kwargs)

            most_recent = self.fresh_values[-1]

            if call_id is None or most_recent.timestep < call_id:
                return instance_func(self, *args, **kwargs)

            return most_recent.value
        return wrapper

    #@staticmethod
    #def cached_eval_bounded(instance_func):
    #    def wrapper(self, *args, **kwargs):
    #        assert (isinstance(self, HistoryLoggingExpression))
    #        if "call_id" not in kwargs:
    #            return instance_func(self, *args, **kwargs)

    #        call_id = kwargs["call_id"]
    #        fresh = self.fresh_values
    #        if len(fresh) == 0 or call_id is None:
    #            return instance_func(self, *args, **kwargs)

    #        most_recent = self.fresh_values[-1]

    #        if most_recent.prob is None:
    #            return instance_func(self, *args, **kwargs)
    #        elif most_recent.timestep < call_id:
    #            return instance_func(self, *args, **kwargs)
    #        raise NotImplementedError
    #        #return ProbabilisticEvaluation(most_recent.value, most_recent.prob)
    #    return wrapper

    def eval_bounded_cached(self, call_id=None):
        """
        Returns the most value of the recent evaluation since `call_id` if one
        exists. Otherwise, return a fresh evaluatation
        """
        raise NotImplementedError
        #fresh = self.fresh_values
        #if len(fresh) == 0 or call_id is None:
        #    return self.eval_bounded(call_id)

        #most_recent = self.fresh_values[-1]

        #if most_recent.prob is None:
        #    return self.eval_bounded(call_id)
        #elif most_recent.timestep < call_id:
        #    return self.eval_bounded(call_id)

        #return ProbabilisticEvaluation(most_recent.value, most_recent.prob)

    def update_values(self, call_id, value_key=None, timestep=None):
        """
        Updates the log of evaluation values that this object keeps.
        :param call_id: an identifier that uniquely identifies an update to values.
        if a timestep is not provided, call_id is assumed to be the timestep
            - update_values is idempotent wrt call_id
        :param value_key: uniquely identifies the value being updated.
        if a new key is provided, a new value is added. if an existing key is
        provided, the existing value is marked stale and a new value is then added.
        """
        if call_id in self.call_ids:
            return None

        try:
            val = self.eval_cached(call_id)
            if isinstance(val, np.generic):
                val = val.item()  # np.asscalar(val)

            if value_key is not None:
                self.retire_values(with_value_key=value_key)
            else:
                value_key = str(uuid.uuid4()) if value_key is None else value_key
            
            self.add_active_value(TimestampedVal(
                value=val,
                timestep=call_id if timestep is None else timestep,
                stale=False,
                prob=None,
                upper=None,
                lower=None
            ), value_key)
            self.call_ids.add(call_id)

            return value_key
        except NoObservedOutcomesError:
            return None

    def add_bounds(self, bound_producer, value_key=None, selector=lambda x: x):
        try:
            value = bound_producer()
        except (NoObservedOutcomesError, StatisticsError):
            return None

        prob = selector(value)

        if value_key is None or prob is None:
            return None

        # clip prob to [0, 1]
        prob = max(0, min(1, prob))

        self.edit_active_value(lambda tsv: TimestampedVal(value=tsv.value, timestep=tsv.timestep, stale=False, prob=prob, upper=tsv.upper, lower=tsv.lower), value_key)

        return value
    