# predictive-modeling-assisted-adaptive-design


This repository contains code for the implmentation in the paper
**"Improving Interim Decision-Making: Predictive Modeling of Censored Survival Outcomes with Machine Learning."**.

The code implements the proposed method and provides a  demo.

## Repository Structure

```
.
├── environment.yml   # Conda environment configuration
├── simulation_demo.ipynb        # Demonstration notebook for running the method
└── Prediction_Assisted_CP_utils.py          # Utility functions used in the implementation
```

## Setup

We recommend using **Conda** to reproduce the computational environment. Create the environment using:

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

## Running the Demo

To run the demo:

1. Launch Jupyter Notebook.
2. Open `simulation_demo.ipynb`.
3. Run the cells sequentially.

## Notes

* All data used in the paper are generated through simulation.
* No external datasets are required.
* The notebook demonstrates the core workflow used in the study.

## Contact

If you have questions about the code, please contact qingkai [dot] dong [at] uconn [dot] edu.
