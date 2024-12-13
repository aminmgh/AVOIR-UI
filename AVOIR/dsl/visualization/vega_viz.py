"""A python wrapper of streamlit to visulize Vega Specs using iframes. Currently no interaction"""
import json
from typing import Dict, List
from os import path
import streamlit.components.v1 as components
import pdb

_SPEC_FILENAME = "visualization_spec_template2.vg.json"
_SPEC_PATH = path.join(path.dirname(__file__), _SPEC_FILENAME)

_DEFAULT_WIDTH = 3000
_DEFAULT_HEIGHT = 1200

def add_vega_chart(viz_spec: Dict, height=_DEFAULT_HEIGHT, width=_DEFAULT_WIDTH):
    """Extends the underlying streamlit library until vega chart support added to streamlit"""
    html = """ 
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            .error {
                color: red;
            }
          </style>
          <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@5"></script>
        </head>
        <body bgcolor='white'>
      <div id="vis"></div>
      <script>""" + \
           f'vegaEmbed("#vis", {json.dumps(viz_spec)})' + \
           """
              // result.view provides access to the Vega View API
              .then(result => console.log(result))
              .catch(console.warn);
          </script>
        </body>
        </html>
        """
    components.html(html, height=height, width=width, scrolling=True)

def transform_data_for_viz(dvs: List[Dict]):
    def row_with_seperated_vals(row):
        new_row = row
        new_row["timesteps"] = [tsv.timestep for tsv in row["vals"]]
        new_row["values"] = [tsv.value for tsv in row["vals"]]
        new_row["probs"] = [tsv.prob for tsv in row["vals"]]
        return new_row

    data_with_seperated_vals = list(map(row_with_seperated_vals, dvs))
    return data_with_seperated_vals



# TODO fix the random numbers taken off width and height that are needed to prevent scrolling problem
def create_viz_spec(dvs: List[Dict], height=_DEFAULT_HEIGHT-40, width=_DEFAULT_WIDTH-50) -> Dict:
    # TODO: Allow some other properties to be specified in args (eg. height, fontsize etc)
    viz_spec = json.load(open(_SPEC_PATH, "r"))
    viz_xformed_data = transform_data_for_viz(dvs)
    viz_spec["data"][0]["values"] = viz_xformed_data
    viz_spec["width"] = width
    viz_spec["height"] = height
    return viz_spec

    # return {
    #     "$schema": "https://vega.github.io/schema/vega/v5.json",
    #     "description": "A visualization of a fairness specification written in our grammar",
    #     "width": 700,
    #     "height": 700,
    #     "padding": 5,

    #     "signals": [
    #         {
    #             "name": "labels", "value": True,
    #             "bind": {"input": "checkbox"}
    #         },
    #         {
    #             "name": "layout", "value": "tidy",
    #             "bind": {"input": "radio", "options": ["tidy", "cluster"]}
    #         },
    #         {
    #             "name": "links", "value": "diagonal",
    #             "bind": {
    #                 "input": "select",
    #                 "options": ["line", "curve", "diagonal", "orthogonal"]
    #             }
    #         },
    #         {
    #             "name": "separation", "value": False,
    #             "bind": {"input": "checkbox"}
    #         },
    #         {
    #             "name": "timestamp", "value": 0,
    #             "bind": {
    #                 "input": "range",
    #                 "min": 0,
    #                 "max": 2,
    #                 "step": 1
    #             }
    #         },
    #         {
    #             "name": "tooltip",
    #             "value": {},
    #             "on": [
    #                 {"events": "symbol:mouseover", "update": "datum"},
    #                 {"events": "symbol:mouseout", "update": "{}"}
    #             ]
    #         }
    #     ],

    #     "data": [
    #         {
    #             "name": "tree",
    #             "values": dvs,
    #             "transform": [
    #                 {
    #                     "type": "filter",
    #                     "expr": "datum.idx == timestamp"
    #                 },
    #                 {
    #                     "type": "stratify",
    #                     "key": "id",
    #                     "parentKey": "parent_id"
    #                 },
    #                 {
    #                     "type": "tree",
    #                     "method": {"signal": "layout"},
    #                     "size": [{"signal": "height"}, {"signal": "width"}],
    #                     "separation": {"signal": "separation"},
    #                     "as": ["y", "x", "depth", "children"]
    #                 }
    #             ]
    #         },
    #         {
    #             "name": "links",
    #             "source": "tree",
    #             "transform": [
    #                 {"type": "treelinks"},
    #                 {
    #                     "type": "linkpath",
    #                     "shape": {"signal": "links"},
    #                     "orient": "horizontal"
    #                 }
    #             ]
    #         }
    #     ],

    #     "scales": [
    #         {
    #             "name": "color",
    #             "type": "linear",
    #             "range": {"scheme": "magma"},
    #             "domain": {"data": "tree", "field": "depth"},
    #             "zero": True
    #         }
    #     ],

    #     "marks": [
    #         {
    #             "type": "path",
    #             "from": {"data": "links"},
    #             "encode": {
    #                 "update": {
    #                     "path": {"field": "path"},
    #                     "stroke": {"value": "#ccc"}
    #                 }
    #             }
    #         },
    #         {
    #             "type": "symbol",
    #             "from": {"data": "tree"},
    #             "encode": {
    #                 "enter": {
    #                     "size": {"value": 900},
    #                     "stroke": {"value": "#fff"}
    #                 },
    #                 "update": {
    #                     "x": {"field": "x"},
    #                     "y": {"field": "y"},
    #                     "fill": {"scale": "color", "field": "depth"}
    #                 },
    #                 "hover": {
    #                     "fill": {"value": "green"}
    #                 }
    #             }
    #         },
    #         {
    #             "type": "text",
    #             "from": {"data": "tree"},
    #             "encode": {
    #                 "enter": {
    #                     "text": {"field": "repr"},
    #                     "fontSize": {"value": 12},
    #                     "baseline": {"value": "middle"}
    #                 },
    #                 "update": {
    #                     "x": {"field": "x"},
    #                     "y": {"field": "y"},
    #                     "dx": {"signal": "datum.children ? -15 : 15"},
    #                     "align": {"signal": "datum.children ? 'right' : 'left'"},
    #                     "opacity": {"signal": "labels ? 1 : 0"}
    #                 }
    #             }
    #         },
    #         {
    #             "type": "text",
    #             "encode": {
    #                 "enter": {
    #                     "text": {"field": "vals"},
    #                     "fontSize": {"value": 12},
    #                     "baseline": {"value": "middle"}
    #                 },
    #                 "update": {
    #                     "x": {"signal": "tooltip.x"},
    #                     "y": {"signal": "tooltip.y"},
    #                     "dy": {"value": 20},
    #                     "dx": {"value": 20},
    #                     "align": {"signal": "tooltip.children ? 'top':'bottom'"},
    #                     "text": {"signal": "tooltip.vals"},
    #                     "fillOpacity": [
    #                         {"test": "isNaN(tooltip.repr)", "value": 0},
    #                         {"value": 1}
    #                     ]
    #                 }
    #             }
    #         }
    #     ]
    # }
