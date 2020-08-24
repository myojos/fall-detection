# ojos-fall-detection
A repository containing the code related to the fall detection models used by OJOS

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── results            <- Everything that can be shown to the outside world
    │   ├── models         <- Trained and serialized models, model predictions, or model summaries
    │   └── reports        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                     the creator's initials, and a short `-` delimited description, e.g.
    │   │                     `1.0-jqp-initial-data-exploration` (<step>-<ghuser>-<description>.ipynb).
    │   │
    │   ├── exploratory    <- Initial explorations
    │   └── reports        <- More polished work that can be exported as HTML to the reports directory
    │
    ├── environment.yml    <- The conda environment file for reproducing the analysis environment, e.g.
    │                         generated with `conda env export > environment.yml`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so ojosfd can be imported
    └── ojosfd             <- Source code for use in this project. 
        ├── __init__.py    <- Makes it a Python module
        │
        ├── datasets       <- Pytorch datasets for this project
        │
        ├── utils          <- Pytorch modules that are useful for training (trasnformers, pipelines, etc...)
        │
        ├── models         <- Pytorch models
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
     
--------


