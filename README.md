# Accountability Project

## Setup environment

The packages required and the environment can be setup by either using `pip`
```
pip install rquirements.txt
```
or `pipenv` (recommended)
```
pipenv install 
```

## Setup optimizers

We use `IPOPT`, an executable is provided. 
Add the executable path to your `PATH` to allow RBR to use `IPOPT`. 
One way to add it is to run this in the `RBR` directory (note, NOT in  `RBR/RBR`)
```
export PATH=$PATH:$(pwd)/nlp_solver
```

To test that `IPOPT` is setup correctly, run `which ipopt` and you should get the path of the executable as the output.

Note: You may need to follow system specific instructions to install [GLPK](https://www.gnu.org/software/glpk/) and [IPOPT](https://coin-or.github.io/Ipopt/) in case the runtime invironment is not x86_64 linux.

## Pretrained models
Please download and extract the  `outputs` directory from [dropbox](https://www.dropbox.com/sh/n5o4vswnkxv34zr/AABthgLMaYL3MuA0KC39Z1G8a?dl=0) to the `RBR/RBR` directory to reproduce the `ratemyprofs.ipynb`.
The other results do not require this directory.


## Reproducing main results

Start a jupyter server in the `RBR` directory with
```
jupyter notebook
```

The main results of the paper (case studies) can be reproduced from the following notebooks:
* `ratemyptors.ipynb`
* `adult_view_maintenance_with_bounds.ipynb`
* `compas_view_maintenance_with_bounds.ipynb`


### Progress bars

```
jupyter nbextension enable --py widgetsnbextension
```

### Vega jupyter extention
```bash
jupyter nbextension install --sys-prefix --py vega
jupyter nbextension enable vega --py --sys-prefix
```
or 
```bash
conda install vega
```



### Streamlit
For interactive demo

```bash
cd src
streamlit run streamlit_interactive_viz.py
```

Note: The local url does not seem to support auto config loading. Use the network url for correct view.

Run with `streamlit run streamlit_interactive_viz.py --logger.level=ERROR` to avoid logging intermediate results

Needs at least python 3.7

Need to install a solver for pyomo. We use [IPOPT](https://coin-or.github.io/Ipopt/)

## Acknowledgment

The authors acknowledge support from National Science Foundation (NSF) grant \#2112471 (AI-EDGE) and a grant from Cisco Research (US202581249).  Any opinions and findings are those of the author(s) and do not necessarily reflect the views of the granting agencies.

