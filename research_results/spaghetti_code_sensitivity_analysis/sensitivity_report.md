# Activity-Aware Comparative Hyperparameter Sensitivity Analysis

## Executive result

The comparison explicitly avoids dilution by inactive or spurious parameters. The primary local sensitivity is the mean absolute score response conditional on a parameter having a non-zero local score effect. It is accompanied by the total L1 effect and an effective sensitivity dimension; appending any number of exact zero-effect parameters changes none of these three quantities.

The spaghetti controller's active-parameter mean response is 43.855 score points, which is higher than the final controller's 15.310. The corresponding spaghetti/final ratio is 2.864.

Across joint ±10% perturbations of all structurally referenced parameters, the spaghetti score standard deviation is 44.183, compared with 28.050 for the final controller. These results distinguish local main-effect sensitivity from interaction-driven robustness.

Taken together, the activity-aware evidence supports the conclusion that the spaghetti controller is more hyperparameter-sensitive than the final controller in the tested ±10% neighborhood.

## Methods

Each declared scalar behavioral parameter was perturbed one at a time by ±10% around its current value. Both low- and high-demand simulations were run for every variant. In addition, 24 deterministic random joint trials per controller perturbed every parameter that has at least one source-code reference after its assignment.

A parameter is **score-active** when either local perturbation changes the aggregate score beyond numerical tolerance. A **metric-only active** parameter changes at least one reported metric but not the rounded aggregate score. A **locally dormant** parameter is referenced by the controller but produces no observed response within ±10%. A **structurally unused** parameter has no reference after assignment and is treated as spurious.

The declared-parameter mean is reported only as a diagnostic because it can be made arbitrarily small by appending unused parameters. The primary activity-aware measures are: (i) mean absolute score change over score-active parameters, (ii) the sum of all parameter effects (L1), and (iii) effective dimension, defined as the participation ratio of the non-negative effects. The joint trials exclude structurally unused parameters but retain locally dormant referenced parameters so that interactions are not discarded.

A trial is classified as feasible when the high-demand case has zero collisions, zero teleports, and average speed of at least 8 m/s. The controller files were hashed before and after all experiments.

## Source-integrity check

| Controller | Unchanged | SHA-256 |
| --- | --- | --- |
| final | True | c0c44be6874dd2b83988f9bcaa500ac5a1bd9044fff3ce1887ef2d961bed565f |
| spaghetti | True | e5387949bedcc90ade16909d7dd62855e840a2b71014493e2c7665b58e8e7f0e |

## Baseline performance

| Controller | Score | Case-0 collisions | Case-0 speed | Case-1 collisions | Case-1 speed | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| spaghetti | 75.130 | 0 | 12.13 | 1 | 10.62 | False |
| final | 96.028 | 0 | 12.32 | 0 | 11.77 | True |

## Parameter activity accounting

| Controller | Declared | Referenced | Score-active | Metric-only | Locally dormant | Structurally unused |
| --- | --- | --- | --- | --- | --- | --- |
| spaghetti | 23 | 22 | 18 | 0 | 4 | 1 |
| final | 9 | 9 | 7 | 0 | 2 | 0 |

## Activity-aware aggregate sensitivity

| Controller | Active mean abs(Δscore) | Total L1 effect | Effective dimension | Max abs(Δscore) | Active normalized mean | Joint score SD | Joint normalized SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spaghetti | 43.855 | 789.392 | 12.764 | 114.912 | 0.584 | 44.183 | 0.588 |
| final | 15.310 | 107.170 | 3.168 | 50.528 | 0.159 | 28.050 | 0.292 |

For reference, the dilution-prone declared-parameter means are 34.321 for spaghetti and 11.908 for final. These values are not used for the primary comparison.

## Most sensitive score-active parameters

### Spaghetti controller

| Rank | Parameter | Mean abs(Δscore) | Low score | High score | Case-1 collisions low/high | Case-1 speed low/high |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PLATOON_DECEL | 114.912 | -95.510 | 15.945 | 11/3 | 3.81/1.90 |
| 2 | COOPERATION_RANGE | 89.166 | 44.961 | -73.035 | 3/7 | 10.09/1.34 |
| 3 | SWAP_COORDINATION_DECEL | 70.741 | 12.000 | -3.222 | 2/4 | 1.55/1.68 |
| 4 | COOPERATION_WINDOW | 65.430 | 75.130 | -55.731 | 1/6 | 10.62/1.52 |
| 5 | SPEED_MATCH_FACTOR | 55.535 | 75.130 | -35.941 | 1/7 | 10.62/2.33 |
| 6 | SAFE_TTC | 53.625 | 31.802 | 11.208 | 2/5 | 2.22/9.50 |
| 7 | GAP_SEARCH_DECEL | 51.202 | -26.897 | 74.752 | 7/1 | 5.25/10.56 |
| 8 | PLATOON_DETECT_RANGE | 46.570 | 28.062 | 29.057 | 4/2 | 10.10/1.87 |
| 9 | URGENCY_START_DIST | 37.289 | 62.213 | 13.468 | 2/5 | 10.67/9.72 |
| 10 | MAX_ACCEL_BOOST | 35.250 | 31.615 | 48.144 | 2/3 | 2.37/10.40 |
| 11 | MIN_GAP | 32.702 | 75.130 | 9.726 | 1/4 | 10.62/2.17 |
| 12 | CRITICAL_DIST | 32.036 | 61.377 | 24.810 | 2/3 | 10.43/5.41 |
| 13 | ENTRY_TIME | 30.863 | 75.130 | 13.403 | 1/3 | 10.62/1.71 |
| 14 | OFFSET_BUFFER | 29.925 | 25.735 | 64.673 | 3/2 | 8.62/10.74 |
| 15 | MIN_ACCEL | 28.610 | 28.101 | 64.937 | 4/1 | 9.87/9.75 |
| 16 | SWAP_COORDINATION_ACCEL | 15.347 | 75.141 | 44.448 | 1/1 | 10.62/1.94 |
| 17 | LANE_CHANGE_GAP_BASE | 0.133 | 75.130 | 74.864 | 1/1 | 10.62/10.62 |
| 18 | SWAP_DETECT_RANGE | 0.055 | 75.019 | 75.130 | 1/1 | 10.59/10.62 |

### Final controller

| Rank | Parameter | Mean abs(Δscore) | Low score | High score | Case-1 collisions low/high | Case-1 speed low/high |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | COOP_BRAKE | 50.528 | 60.645 | 30.356 | 0/1 | 2.84/1.24 |
| 2 | ENTRY_MERGE_DISTANCE | 25.972 | 78.335 | 61.778 | 1/0 | 11.19/3.14 |
| 3 | ENTRY_DELAY | 17.588 | 96.028 | 60.853 | 0/0 | 11.77/3.00 |
| 4 | MERGE_BASE_GAP | 8.959 | 94.583 | 79.555 | 0/1 | 11.83/11.36 |
| 5 | MERGE_TTC | 2.285 | 91.461 | 96.031 | 0/0 | 11.20/11.68 |
| 6 | MERGE_COOLDOWN | 1.830 | 96.028 | 92.368 | 0/0 | 11.77/11.65 |
| 7 | RESERVATION_DISTANCE | 0.008 | 96.028 | 96.044 | 0/0 | 11.77/11.78 |

## Inactive and dormant parameters

These parameters are shown explicitly rather than being allowed to dilute the primary active-set mean.

| Controller | Parameter | Classification | Usage count |
| --- | --- | --- | --- |
| spaghetti | ENTRY_DISTANCE | locally_dormant | 1 |
| spaghetti | LANE_CHANGE_COOLDOWN | locally_dormant | 5 |
| spaghetti | TARGET_LANE_PATIENCE | locally_dormant | 1 |
| spaghetti | GAP_SEARCH_ACCEL | locally_dormant | 2 |
| spaghetti | PLATOON_ACCEL | structurally_unused | 0 |
| final | FOLLOW_GAP | locally_dormant | 2 |
| final | REACTION_TIME | locally_dormant | 2 |

## Comparative inference

Using only score-active parameters, the estimated difference in mean absolute local response (spaghetti minus final) is 28.545 score points. The parameter-level bootstrap 95% interval is [9.752, 46.524], and the exploratory one-sided permutation p-value is 0.0056.

The spaghetti/final ratios are 2.864 for active local mean, 7.366 for total L1 effect, 1.575 for absolute joint score SD, and 2.013 for baseline-normalized joint SD.

The active-set comparison answers how strongly a parameter matters when it matters locally. The L1 comparison answers how much total local sensitivity is distributed across the controller. The joint comparison answers how much the score varies when the referenced parameter vector moves simultaneously. These are distinct properties and should not be collapsed into a single parameter-count-dependent average.

The permutation and bootstrap calculations treat named parameters as exchangeable observations, which is only an exploratory approximation. The effect sizes, activity classifications, and raw simulation responses should carry more interpretive weight than the p-value.

## Robustness and feasibility

| Controller | OAT active-variant feasible rate | Joint feasible rate | Joint mean case-1 collisions | Joint mean case-1 speed | Joint score range |
| --- | --- | --- | --- | --- | --- |
| spaghetti | 0.0% | 0.0% | 5.417 | 2.485 | [-98.575, 33.758] |
| final | 57.1% | 29.2% | 0.542 | 6.764 | [16.856, 95.969] |

Feasibility rates must be interpreted relative to the baseline. If a baseline is already infeasible, its feasibility rate measures whether perturbations repair that state, not the probability of losing a feasible operating regime.

## Limitations

OAT responses are local to ±10% and can classify thresholded parameters as locally dormant even when larger changes would activate them. Joint trials partially address interactions but are finite empirical samples rather than exhaustive bounds. Results are specific to the two supplied demand scenarios, simulation seed, evaluator, and current tuned baselines.

## Reproducibility

Perturbation magnitude: ±10%; joint trials per controller: 24; random seed: 20260729; workers: 4. Raw OAT and joint trials are supplied as CSV files; the complete raw and analyzed results are supplied as JSON.
