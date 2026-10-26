# Bias report

Slices: sex, age_group, visit, dbs, hoehn_yahr, pd_subtype

## Data level

### sex

| sex | rows | participants | dyskinetic | off | on |
|---|---|---|---|---|---|
| female | 32 | 2 | 0.094 | 0.562 | 0.344 |
| male | 86 | 3 | 0.372 | 0.523 | 0.105 |

### age_group

| age_group | rows | participants | dyskinetic | off | on |
|---|---|---|---|---|---|
| 55-64 | 89 | 4 | 0.393 | 0.427 | 0.18 |
| 65+ | 29 | 1 | 0.0 | 0.862 | 0.138 |

### visit

| visit | rows | participants | dyskinetic | off | on |
|---|---|---|---|---|---|
| inpatient | 116 | 5 | 0.302 | 0.526 | 0.172 |
| mixed | 2 | 1 | 0.0 | 1.0 | 0.0 |

### dbs

| dbs | rows | participants | dyskinetic | off | on |
|---|---|---|---|---|---|
| no | 25 | 1 | 0.12 | 0.44 | 0.44 |
| yes | 93 | 4 | 0.344 | 0.559 | 0.097 |

### hoehn_yahr

| hoehn_yahr | rows | participants | dyskinetic | off | on |
|---|---|---|---|---|---|
| 2.0 | 7 | 1 | 0.0 | 1.0 | 0.0 |
| 3.0 | 86 | 3 | 0.372 | 0.523 | 0.105 |
| 4.0 | 25 | 1 | 0.12 | 0.44 | 0.44 |

### pd_subtype

| pd_subtype | rows | participants | dyskinetic | off | on |
|---|---|---|---|---|---|
| akinetic-rigid | 46 | 2 | 0.065 | 0.674 | 0.261 |
| equivalence-type | 72 | 3 | 0.444 | 0.444 | 0.111 |

## Model level (baseline: off vs not off)

Overall recall: 0.889

### sex

| group | n | positives | recall | precision | false_positive_rate |
|---|---|---|---|---|---|
| male | 50 | 45 | 0.889 | 0.909 | 0.8 |

Recall gap: nan -> within limit

### age_group

| group | n | positives | recall | precision | false_positive_rate |
|---|---|---|---|---|---|
| 55-64 | 21 | 20 | 0.9 | 0.947 | 1.0 |
| 65+ | 29 | 25 | 0.88 | 0.88 | 0.75 |

Recall gap: 0.020 -> within limit

### visit

| group | n | positives | recall | precision | false_positive_rate |
|---|---|---|---|---|---|
| inpatient | 50 | 45 | 0.889 | 0.909 | 0.8 |

Recall gap: nan -> within limit

### dbs

| group | n | positives | recall | precision | false_positive_rate |
|---|---|---|---|---|---|
| yes | 50 | 45 | 0.889 | 0.909 | 0.8 |

Recall gap: nan -> within limit

### hoehn_yahr

| group | n | positives | recall | precision | false_positive_rate |
|---|---|---|---|---|---|
| 3.0 | 50 | 45 | 0.889 | 0.909 | 0.8 |

Recall gap: nan -> within limit

### pd_subtype

| group | n | positives | recall | precision | false_positive_rate |
|---|---|---|---|---|---|
| akinetic-rigid | 21 | 20 | 0.9 | 0.947 | 1.0 |
| equivalence-type | 29 | 25 | 0.88 | 0.88 | 0.75 |

Recall gap: 0.020 -> within limit
