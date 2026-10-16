# Policy2Agreement: reproducibility package

This package reproduces the proof of concept of the paper *Policy2Agreement: A Vision for Compiling Governance Policies into Negotiated Agreements among Organizational Agents* (Section 4 and Table 1). The notebook negotiates the agreements of the paper's running example and then executes them in simulated federated learning (FL) with differential privacy (DP).

## What Policy2Agreement does

When organizations collaborate, for instance to train a model on data that each of them controls, their software agents must agree on a configuration: here, a privacy budget ε, a target F1 score q for the trained model and a number of training rounds t. Each party, and each authority that governs the parties, already has a governance policy, a set of norms that constrains the configuration. Written independently, these policies conflict. Policy2Agreement compiles them into a negotiation among the agents and routes each conflict to whoever has standing over it, in four phases:

1. **Policy ingestion.** A language model translates each policy into norms, under human review. This phase is not exercised here: the norms enter as data.
2. **Conflict detection and resolution.** A direct conflict, where one norm rules out what another allows, is resolved by standing: the authority's norm prevails over the parties' (lex superior) and narrows them (curtailment), and what remains is the feasible set Ω_F. An indirect conflict, which only domain knowledge reveals, is made explicit by a domain rule that the authority approves: it flags a party's implausible combination of constraints with a multiplier μ.
3. **Preference compilation.** Each party's constraints are read as directions of preference, which define an ideal point, and Eq. (1) scores every configuration by its distance from that point. Outside a party's acceptance set the adjusted utility û is 0; on a region its rule flags, û is μ times the utility.
4. **Negotiation.** The agents negotiate with MiCRO-Min under the Stacked Alternating Offers Protocol. MiCRO-Min reads only the ordering of the utilities, so no agent discloses a number. An outcome that every agent accepts is the agreement; if none exists, the case escalates to the authority.

The notebook then executes each agreement in simulated federated learning with differential privacy and checks whether the trained model reaches the agreed F1 score. It documents the training setup and its limits in its own Parts 3 and 8.

### The running example

Two hospitals H1 and H2 and an orchestrator R agree on (ε, q, t), with ε ∈ {0.3, 0.5, 0.6, 0.8}, q ∈ {0.80, 0.85, 0.90, 0.92} and t ∈ {10, 15, 20}: 48 configurations.

| Issuer | Norms |
|---|---|
| H1 (party) | ε ≤ 0.3 and q ≥ 0.92 |
| H2 (party) | ε ≤ 0.5 and t ≤ 15 |
| R (party) | q ≥ 0.85 and t ≤ 20 (in the variant, t ≥ 15) |
| Consortium (authority) | obliges t ≥ 15 |
| Regulator (authority) | forbids ε > 0.6 |

Read as preferences, all three parties rank (0.3, 0.92, 10) first. That configuration breaks the consortium's obligation, and q = 0.92 is implausible under ε = 0.3. Leaving the authorities' norms aside, a domain rule that flags H1's pair of constraints with μ = 0.2 moves the agreement to ω* = (0.5, 0.92, 10). Executed, the favorite reaches its F1 target in 32% of runs and ω* in 88%. Like the paper's prototype, the notebook applies the consortium's obligation but not the regulator's prohibition, which the prototype does not yet handle.

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

## Relation to the prototype

The paper's prototype reads ODRL policies, runs Phase 2 in the EYE reasoner and negotiates with its own MiCRO-Min engine. So that this package installs with pip alone, `p2a_negotiation.py` encodes the running example's norms and domain rule in Python and ports the prototype's compiler of Eq. (1) and its negotiation engine line by line. The notebook checks this port three ways:

- the constraints it uses are exactly those in the ODRL policies (`data/odrl/`);
- every adjusted utility it compiles equals the prototype's (`results/prototype_utilities.json`), as an exact equality of floating-point numbers;
- every agreement and mean step count equals the prototype's record (`results/table1_prototype.json`).

Like the prototype's, the engine uses a reservation value of 0. An agent could therefore accept an outcome outside its acceptance set once it has conceded that far, so the notebook checks that no agreement of Table 1 does.

The language-model rule proposals of Section 4 need an API key and a model that changes over time, so the package includes their recorded answers (`results/llm_rule_proposals.csv`) instead of re-running them. The ODRL ingestion and the EYE reasoner are not included.

## Data

`data/breast_w.csv` is the Breast Cancer Wisconsin (Original) dataset as distributed by OpenML (data id 15, "breast-w", https://www.openml.org/d/15), exported unchanged. It has 699 biopsies with nine cytology features on a 1 to 10 scale, of which 241 are malignant, and 16 missing values. The notebook rescales the features with the public bounds of that scale and fills the missing values with its midpoint. The dataset is licensed under CC BY 4.0.

- Wolberg, W. (1990). Breast Cancer Wisconsin (Original) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z
- Wolberg, W. H., Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. *Proceedings of the National Academy of Sciences* 87(23), 9193–9196.
