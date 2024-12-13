from typing import Dict, List

from .dataset import Dataset
from .adult_income import (
    load_data as load_adult_income,
    attributes_dict as adult_inc_attr_dict,
    attributes as adult_inc_attrs
)

from .boston_housing import (
    load_data as load_boston,
    attributes_dict as boston_attr_dict,
    attributes as boston_attrs
)

from .ratemyprofessors import (
    load_data as load_rmp,
    attributes_dict as rmp_attr_dict,
    attributes as rmp_attrs
)

from .compas import (
    load_data as load_compas,
    attributes_dict as compas_attr_dict,
    attributes as compas_attrs
)

_boston = "Boston Housing Prices"
_income = "Adult Income"
_rmp = "Rate My Professors"
_compas = "Compas"
_dnames = [_income, _rmp, _compas]
#_boston, 

def get_dataset_names():
    return _dnames


def check_valid_dataset(dataset: str):
    # TODO make decorator or change dnames to enum
    if dataset not in _dnames:
        raise ValueError("Dataset not available")


def get_dataset_attr_list(dataset: str) -> List[str]:
    check_valid_dataset(dataset)
    return {
        _boston: boston_attrs,
        _income: adult_inc_attrs,
        _rmp: rmp_attrs,
        _compas: compas_attrs
    }[dataset]


def get_dataset_attr_dict(dataset: str) -> Dict[str, str]:
    check_valid_dataset(dataset)
    return {
        _boston: boston_attr_dict,
        _income: adult_inc_attr_dict,
        _rmp: rmp_attr_dict,
        _compas: compas_attr_dict
    }[dataset]


def get_dataset(dataset: str) -> Dataset:
    check_valid_dataset(dataset)
    return {
        _boston: load_boston,
        _income: load_adult_income,
        _rmp: load_rmp,
        _compas: load_compas
    }[dataset]()
