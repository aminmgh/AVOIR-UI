from typing import Dict, Tuple
import pandas as pd
import numpy as np
from os import path
from .dataset import Dataset


DATA_DIR = path.join(path.dirname(__file__), "data", "ACSIncome")
DATA_FILE = "ACSIncome.csv"

TRAIN_PORTION = 0.06

original_attributes = {
    'AGEP': 'Age as an integer from 0 to 99',
    'COW': 'Class of worker',
    'SCHL': 'Educational attainment',
    'MAR': 'Marital status',
    'OCCP': 'Occupation',
    'POBP': 'Place of birth. There are over 200 categories, including the 50 US states and several countries',
    'RELP': 'Relationship to householder',
    'WKHP': 'Usual hours worked per week in the past 12 months. Values are an integer from 1 to 99.',
    'SEX': 'Sex code: 1.Male 2.Female',
    'RAC1P': 'Race code 1. White alone 2. Black or African American alone 3. American Indian alone 4. Alaska Native alone 5. American Indian and Alaska native tribes specified; or American Indian or Alaska Native, not specified and no other races 6. Asian alone 7. Native Hawaiian and Other Pacific Islander alone 8. Some Other Race alone 9.Two or More races',
    'target': 'Total annual income per person'
}

attributes = ['AGEP',
              'COW',
              'SCHL',
              'MAR',
              'OCCP','POBP',
              'RELP',
              'WKHP','SEX_1.0', 'SEX_2.0', 
              'RAC1P_1.0',
              'RAC1P_2.0',
              'RAC1P_3.0',
              'RAC1P_4.0',
              'RAC1P_5.0',
              'RAC1P_6.0',
              'RAC1P_7.0',
              'RAC1P_8.0',
              'RAC1P_9.0',
              'target']


attributes_dict = dict((attr, "") for attr in attributes)

for attr, description in original_attributes.items():
    attributes_dict.update({
        matched_attr:description
        for matched_attr in attributes
        if matched_attr.startswith(attr)
    })

def _read_data():
    data_fp = path.join(DATA_DIR, DATA_FILE)
    data = pd.read_csv(data_fp)

    data['target'] = (data['target'] >= 50000).astype(int)

    columns_to_convert = ['SEX', 'RAC1P']
    data[columns_to_convert] = data[columns_to_convert].astype(str)
    view = data[original_attributes.keys()].copy()

    view = pd.get_dummies(view)

    train_size = int(len(view) * TRAIN_PORTION)
    train = view.iloc[:train_size]
    test = view.iloc[train_size:2 * train_size]
    #test = view

    return train, test


def load_data() -> Dataset:
    train, test = _read_data()

    return Dataset(
        train=train,
        test=test,
        attributes_dict=attributes_dict,
        target="target"
    )
