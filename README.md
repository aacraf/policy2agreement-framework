# Policy2Agreement: reproducibility package

This package reproduces the proof of concept of the paper *Policy2Agreement: A Vision for Compiling Governance Policies into Negotiated Agreements among Organizational Agents* (Section 4 and Table 1). The notebook negotiates the agreements of the paper's running example and then executes them in simulated federated learning (FL) with differential privacy (DP).

## Contents

| File | Content |
|---|---|
| `running_example.ipynb` | The notebook, with the outputs of a full run |
| `p2a_negotiation.py` | Phases 2 to 4 for the running example: the domain rule, the compiler of Eq. (1), and MiCRO-Min under SAOP |
| `data/breast_w.csv` | Breast Cancer Wisconsin (Original): 699 biopsies, 9 features, 241 malignant |
| `data/odrl/` | The running example's policies in ODRL, as the paper's prototype reads them |
| `results/table1_prototype.json` | Agreements and mean step counts recorded with the paper's prototype |
| `results/prototype_utilities.json` | The adjusted utilities the prototype compiled for every setting |
| `results/llm_rule_proposals.csv` | The recorded language-model rule proposals of Section 4 |
| `results/fl_dp_results.json` | Results of the run shown in the notebook (rewritten on each run) |
| `requirements.txt` | Pinned package versions |

## How to run

You need Python 3.12. Once the packages are installed, no network access is needed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace running_example.ipynb
```

You can also open `running_example.ipynb` in Jupyter and run all cells. A full run uses one core and took between one and three minutes on a laptop in our tests.

Every run uses fixed seeds. The negotiation is plain Python and gives the same results on any platform. The federated training reproduces the stored outputs exactly on Linux (x86-64) with the pinned versions; with other linear-algebra libraries the last digits of a training run can differ, which could move a reported share by one run.

## What it reproduces

As in the paper's Table 1, the negotiations apply the consortium's obligation t ≥ 15 but not the regulator's prohibition ε > 0.6, which the prototype does not yet handle.

| Claim in the paper | Notebook |
|---|---|
| Section 3: the rule drops H1's stated ideal from first to 37th of the 48 outcomes, and its two next-best options rise to joint first (û ≈ 0.83) | Part 2 |
| Section 3: an agent accepts only outcomes inside its acceptance set (every agreement of Table 1 does) | Part 2 |
| Table 1: each agreement, each party's adjusted utility at it, and the mean number of steps (proposals or acceptances after the first proposal) over 6 turn orders × 20 seeds | Part 2 |
| Table 1: every agreement is Pareto-optimal over Ω_F, with respect to the adjusted utilities | Part 2 |
| Table 1: the share of 50 simulated FL runs whose F1 score reaches the agreed target, with 95% intervals | Parts 3, 4 and 6 |
| Section 4: every μ below 5/6 yields ω* = (0.5, 0.92, 10), while μ = 0.9 does not (checked on a grid of μ) | Part 2 |
| Section 4: the consortium's obligation t ≥ 15 splits the agreement by turn order until H1 states that privacy comes first | Part 2 |
| Section 4: the implausible favorites reach their F1 target in 32% and 40% of runs, and the agreements reached with the rule in 88% to 92% | Part 6 |
| Section 4: a rule calibrated on these runs sets μ ≈ 0.37 and again yields ω* (also checked on 50 fresh seeds) | Part 5 |
| Section 4: a language model flagged the same pair of H1 with μ = 0.5 and matched the hand-written rules on 12 of the 13 agents with a valid answer | Part 7 (recorded answers, not re-run) |

## Relation to the prototype

The paper's prototype reads ODRL policies, runs Phase 2 in the EYE reasoner and negotiates with its own MiCRO-Min engine. So that this package installs with pip alone, `p2a_negotiation.py` encodes the running example's norms and domain rule in Python and ports the prototype's compiler of Eq. (1) and its negotiation engine line by line. The notebook checks this port three ways:

- the constraints it uses are exactly those in the ODRL policies (`data/odrl/`);
- every adjusted utility it compiles equals the prototype's (`results/prototype_utilities.json`), as an exact equality of floating-point numbers;
- every agreement and mean step count equals the prototype's record (`results/table1_prototype.json`).

Like the prototype's, the engine uses a reservation value of 0. An agent could therefore accept an outcome outside its acceptance set once it has conceded that far, so the notebook checks that no agreement of Table 1 does.

The language-model rule proposals of Section 4 need an API key and a model that changes over time, so the package includes their recorded answers (`results/llm_rule_proposals.csv`) instead of re-running them. The ODRL ingestion and the EYE reasoner are not included.

## Federated learning and differential privacy

Each run is a 5-fold cross-validation. The four training folds are split, stratified, between the two hospitals H1 and H2 (about 280 patients each), and the orchestrator R averages their models (FedAvg). Each hospital trains a logistic regression with DP-SGD (Poisson sampling, per-example clipping and Gaussian noise), one local epoch per round. Privacy is record-level (ε, δ)-DP with δ = 10⁻⁵ over all t rounds. It is accounted with Rényi DP at integer orders 2 to 512 and converted with the bound of Canonne, Kamath and Steinke (2020), and the notebook checks the accountant against two values from Google's dp-accounting library.

Hyperparameters were tuned for each (ε, t) on seeds 0 to 19 and are stored in the notebook. Setting `RETUNE = True` redoes the tuning, which runs 24,000 trainings instead of 750. All reported results use the separate seeds 100 to 149, and the calibration check uses seeds 200 to 249. The tuning uses the same patients, so a deployment would tune on public or synthetic data. NumPy's generator is not a cryptographic source, so the code simulates the DP guarantee rather than providing it.

## Data

`data/breast_w.csv` is the Breast Cancer Wisconsin (Original) dataset as distributed by OpenML (data id 15, "breast-w", https://www.openml.org/d/15), exported unchanged. It has 699 biopsies with nine cytology features on a 1 to 10 scale, of which 241 are malignant, and 16 missing values. The notebook rescales the features with the public bounds of that scale and fills the missing values with its midpoint. The dataset is licensed under CC BY 4.0.

- Wolberg, W. (1990). Breast Cancer Wisconsin (Original) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z
- Wolberg, W. H., Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. *Proceedings of the National Academy of Sciences* 87(23), 9193–9196.
