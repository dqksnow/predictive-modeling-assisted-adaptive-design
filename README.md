# Predictive-Modeling–Assisted Interim Decision-Making

Core code and an example for *Predictive-Modeling–Assisted Interim Decision-Making in Adaptive Trials with Censored Survival Outcomes*.

`implementation.py` contains the log-rank effect, conditional power, and five methods: Baseline, Weibull AFT, log-normal AFT, CQRNN, and Ye–Shao–Yi. `simulation_demo.ipynb` generates an interim dataset and displays their results. No external data are needed.

## Run

```bash
conda env create -f environment.yml
conda activate pai_public_core
jupyter lab simulation_demo.ipynb
```

Run the notebook cells in order. The example uses true HR=1, 140 interim events out of 200 final events (tau = 0.7), and seed=2026.
