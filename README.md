# Explainable Hotel Recommendation System using Collaborative Filtering and Aspect-Level Preferences

This repository contains the implementation pipeline for my Master's dissertation on **hotel recommendation using collaborative filtering and aspect-level preference information**.

The project investigates whether explicit hotel aspects can improve collaborative recommendation performance and whether those same aspects can instead be used to provide interpretable post-hoc explanations for hotel recommendations.

The implementation is provided as a single end-to-end Jupyter Notebook covering data preparation, model training, evaluation, hybrid experiments, robustness analysis, and explanation generation.

---

## Research Objectives

The project addresses three research questions:

**RQ1:** How effectively can collaborative filtering methods recommend relevant hotels in a highly sparse user–hotel interaction setting?

**RQ2:** Can explicit hotel aspect information improve the ranking performance of a collaborative recommender?

**RQ3:** Can aspect-level preference information provide useful and interpretable explanations for collaborative hotel recommendations?

The project therefore evaluates recommendation accuracy and explanation separately rather than assuming that information useful for explanation must also improve ranking.

---

## Dataset

The project uses the **HotelRec** dataset introduced by Antognini and Faltings (2020).

HotelRec is a large-scale hotel recommendation dataset derived from TripAdvisor reviews. It contains user–hotel interactions together with overall ratings, review information, and hotel sub-ratings.

The implementation uses the following hotel aspects:

- Service
- Location
- Value
- Cleanliness
- Rooms

The raw working dataset used by the notebook contains approximately:

- **22.5 million reviews**
- **2.03 million users**
- **362,000+ hotels**

After cleaning and resolving repeated user–hotel interactions, approximately **20.4 million unique user–hotel interactions** remain.

### Dataset Reference

Antognini, D. and Faltings, B. (2020)  
*HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset*.  
Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020), pp. 4917–4923.

Dataset repository:  
https://github.com/diegoantognini/HotelRec

Paper:  
https://aclanthology.org/2020.lrec-1.605/

> **Note:** The original HotelRec data contains explicit ratings. In this project, qualifying positive ratings are transformed into preference evidence for the Top-K recommendation experiments.

---

## Repository Contents

The repository is intentionally kept simple.

```text
.
├── hotel_recommendation_pipeline.ipynb
├── raw_dataset/
└── README.md
```

### `hotel_recommendation_pipeline.ipynb`

The main notebook contains the complete experimental pipeline, including:

1. Data loading and validation
2. Data cleaning and preprocessing
3. User–hotel interaction preparation
4. Temporal train/test construction
5. Evaluation candidate generation
6. Popularity baseline
7. Item-Based Collaborative Filtering
8. Alternating Least Squares (ALS)
9. Hyperparameter and sensitivity experiments
10. Aspect profile construction
11. Aspect-only recommendation
12. ALS–aspect hybrid experiments
13. Top-K aspect reranking
14. Aspect signal diagnostics
15. Post-hoc explanation generation
16. Statistical significance testing
17. Robustness analysis

---

## Recommendation Pipeline

The main recommendation architecture is:

```text
User Interaction History
          |
          v
         ALS
          |
          v
 Ranked Hotel Recommendations
```

ALS learns latent representations from historical user–hotel interactions and produces the final collaborative ranking.

The explanation component operates separately:

```text
User Aspect Profile
        +
Recommended Hotel Aspect Profile
        |
        v
Aspect Contribution Analysis
        |
        v
Post-hoc Explanation
```

This separation is important because the experiments test whether aspect information should influence ranking rather than assuming that a hybrid model must perform better.

---

## Data Preparation

The raw dataset is processed using **DuckDB** because of its size.

The preprocessing pipeline includes:

- validation of user and hotel identifiers;
- removal of invalid records;
- handling of repeated user–hotel interactions;
- selection of required rating and aspect fields;
- construction of deterministic temporal interaction histories;
- separation of training and held-out interactions;
- filtering of the collaborative hotel catalogue;
- construction of positive preference interactions;
- creation of common evaluation candidate sets.

The final evaluation is based on a **warm-start recommendation setting**, meaning that users and evaluated hotels have suitable training information available to the collaborative models.

---

## Evaluation Framework

All final recommendation models are evaluated using a common candidate-set framework.

The main evaluation contains:

- **37,946 evaluation users**
- **87,428 eligible hotels**
- **1 held-out positive hotel per user**
- **99 sampled negative hotels per user**
- **100 candidates per user**

The same candidate sets are reused across models to provide a controlled comparison.

Deterministic tie-breaking is applied when recommendation scores are equal.

### Evaluation Metrics

Performance is measured using:

- Hit Rate at 5 (**HR@5**)
- Hit Rate at 10 (**HR@10**)
- Hit Rate at 20 (**HR@20**)
- Normalised Discounted Cumulative Gain at 5 (**NDCG@5**)
- Normalised Discounted Cumulative Gain at 10 (**NDCG@10**)
- Normalised Discounted Cumulative Gain at 20 (**NDCG@20**)

Hit Rate measures whether the held-out relevant hotel appears within the Top-K recommendations.

NDCG additionally considers the position of the relevant hotel, giving greater credit when it appears closer to the top of the ranking.

> The reported results are based on sampled candidate sets rather than full-catalogue ranking. This should be considered when interpreting the absolute metric values.

---

## Recommendation Models

### Popularity Baseline

A non-personalised popularity model is included as a simple baseline. Hotels are ranked according to positive interaction frequency in the training data.

### Item-Based Collaborative Filtering

ItemCF models relationships between hotels based on user interaction patterns.

Sparse cosine similarity is used and only the strongest neighbours are retained to make computation practical at the scale of the dataset.

Neighbourhood sizes of **50, 100 and 200** are examined, with **200 neighbours** used for the final comparison.

### Alternating Least Squares

ALS is the primary collaborative recommendation model.

Positive interactions are constructed from ratings of **4 or 5**, with stronger weighting given to 5-star interactions.

The final configuration uses:

```text
Factors:        64
Regularisation: 0.05
Iterations:     15
Random seed:    42
```

Factor sizes of 32, 64 and 128 and multiple regularisation settings are evaluated before selecting the final configuration.

---

## Aspect-Based Recommendation

The project investigates five explicit hotel aspects:

```text
Service
Location
Value
Cleanliness
Rooms
```

Training information is used to construct user and hotel aspect profiles.

The profiles are centred and normalised to represent **relative aspect preferences and alignment**, rather than treating raw aspect averages as direct measures of psychological importance.

An aspect compatibility score is then calculated between a user and candidate hotel.

The aspect component is evaluated in three main ways:

1. As an independent aspect-only recommender
2. Combined directly with ALS scores
3. Used to rerank hotels already selected by ALS

This allows the project to determine empirically whether explicit aspects improve collaborative ranking.

---

## Main Results

The final common-catalogue comparison is:

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.2700 | 0.1798 | 0.3801 | 0.2152 | 0.5235 | 0.2513 |
| ItemCF-200 | 0.2822 | 0.2419 | 0.3245 | 0.2554 | 0.4071 | 0.2760 |
| **ALS-64** | **0.4405** | **0.3157** | **0.5814** | **0.3612** | **0.7275** | **0.3982** |
| Aspect-only | 0.0600 | 0.0357 | 0.1187 | 0.0544 | 0.2328 | 0.0829 |

**ALS-64 achieved the strongest overall recommendation performance across all evaluated cut-offs.**

---

## ALS and Aspect Integration

Several hybrid configurations were tested by combining normalised ALS and aspect scores.

The general formulation was:

```text
Hybrid Score =
α × ALS Score + (1 - α) × Aspect Score
```

Multiple values of `α` were evaluated.

The results showed that performance generally improved as greater weight was returned to ALS, with **pure ALS providing the strongest overall ranking performance**.

A second experiment used aspect information only to rerank the ALS Top-20 recommendations. Increasing the aspect contribution again reduced ranking performance.

These results indicate that, in the evaluated setting, explicit aspect information contains preference information but does not provide sufficient additional ranking signal to improve the stronger collaborative ALS model.

---

## Aspect Signal Analysis

Although aspect-based ranking was substantially weaker than ALS, additional diagnostics showed that the aspect signal was not completely random.

Across the evaluation:

```text
Mean positive-hotel aspect score: 0.4466
Mean negative-hotel aspect score: 0.4020
Mean score margin:                0.0446
Positive-margin rate:             61.72%
Mean pairwise win rate:           0.5345
Mean positive-hotel rank:         47.09 / 100
```

Aspect information also became more informative for users with richer positive interaction histories.

This suggests that explicit aspects capture some meaningful preference alignment, even though the signal is not strong enough to improve ALS ranking.

---

## Post-hoc Explanation Layer

Because aspect information did not improve the collaborative ranking, the final system retains **ALS-64 as the ranking model** and uses aspects separately for explanation.

For each ALS-recommended hotel, the explanation layer compares the user's relative aspect profile with the recommended hotel's aspect profile.

The contribution of individual aspects can then be used to produce an interpretable explanation of user–hotel alignment.

Example:

```text
This recommended hotel also aligns strongly with your
observed preferences for value and cleanliness.
```

These explanations should **not** be interpreted as causal explanations of the internal ALS model.

For example, the system should not claim:

```text
ALS recommended this hotel because of cleanliness.
```

ALS learns latent factors and does not explicitly make its ranking decision using these aspect labels.

The aspect layer instead provides a **post-hoc description of observable alignment** between the user and an already recommended hotel.

---

## Explanation Evaluation

The explanation layer was evaluated on a sample of **5,000 users** using the Top-5 ALS recommendations.

Approximately **25,000 explanations** were expected, of which **24,625** had sufficient aspect information.

Key results include:

```text
Explanation coverage:              98.50%
Positive overall aspect alignment: 85.54%
Primary contribution positive:     99.63%
Primary aspect share >= 40%:       85.83%
Primary aspect share >= 50%:       62.51%
```

The most frequently identified primary aspects were:

```text
Value:        38.00%
Cleanliness:  22.89%
Rooms:        18.76%
Location:     15.84%
Service:       4.52%
```

The results indicate that aspect profiles can provide high-coverage interpretable alignment information even though they do not improve the underlying ALS ranking.

---

## Statistical Validation

Model differences are additionally evaluated using paired user-level statistical tests.

The analysis includes:

- **McNemar tests** for differences in Hit Rate;
- **Wilcoxon signed-rank tests** for differences in NDCG;
- **Bootstrap confidence intervals** for NDCG differences.

The final comparisons between ALS and the main alternative models are statistically significant at **p < .001**.

Bootstrap analysis also shows positive ALS improvements in NDCG@10 over Popularity, ItemCF and the Aspect-only model.

---

## Key Finding

The central finding of the project is that **information useful for explanation is not necessarily information that improves recommendation ranking**.

In the evaluated HotelRec setting:

```text
Collaborative interaction information
             |
             v
          ALS-64
             |
             v
      Hotel Ranking
```

provides the strongest recommendation performance.

Explicit hotel aspects are instead retained as:

```text
User Aspect Profile
        +
Hotel Aspect Profile
        |
        v
Aspect Alignment
        |
        v
Post-hoc Explanation
```

The final architecture therefore separates **ranking** from **explanation**.

---

## Requirements

The notebook requires Python 3 and commonly used data-science and recommendation libraries, including:

```text
numpy
pandas
duckdb
scipy
scikit-learn
implicit
sparse-dot-topn
matplotlib
```

Depending on the notebook version, additional standard libraries may also be imported.

For exact reproducibility, it is recommended to create an environment using the same package versions used during the original experiments.

---

## Running the Project

1. Obtain the HotelRec dataset from the official source.(https://github.com/diegoantognini/HotelRec)
2. Place the required raw dataset inside the `raw_dataset/` directory.
3. Update the dataset path in the notebook if necessary.
4. Install the required Python dependencies.
5. Open the Jupyter Notebook.
6. Run the notebook sequentially from the beginning.

Because the raw dataset and intermediate matrices are large, the complete pipeline requires substantial storage and memory. Runtime will depend on the available hardware.

The notebook should be executed in order because later stages depend on datasets, mappings, candidate sets and model outputs generated during earlier stages.

---

## Reproducibility

Where applicable, random operations use fixed seeds, including a seed of:

```python
42
```

The evaluation candidate sets are generated once and reused across models to ensure that model comparisons use the same evaluation conditions.

The temporal split and ranking procedures also use deterministic ordering and tie-breaking where required.

---

## Limitations

The implementation should be interpreted with several limitations in mind.

The final evaluation uses one positive hotel and 99 sampled negatives rather than ranking against the complete hotel catalogue. The main evaluation is also focused on warm-start recommendation and therefore does not represent a complete cold-start solution.

Aspect availability varies across reviews, and the five available aspects cannot represent every factor that may influence hotel choice. Furthermore, the post-hoc aspect explanations describe observable alignment between user and hotel profiles; they do not reveal the internal causal reasoning of ALS.

Finally, explanation quality is evaluated computationally rather than through a human user study. Future work could therefore investigate whether users find the generated explanations useful, understandable and trustworthy.

---

## Final System

The final experimental system can be summarised as:

```text
                    ┌──────────────────────┐
                    │ User Interaction     │
                    │ History              │
                    └──────────┬───────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │       ALS-64         │
                    └──────────┬───────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │ Ranked Hotel         │
                    │ Recommendations      │
                    └──────────┬───────────┘
                               │
                               v
          ┌────────────────────────────────────────┐
          │ User Aspect + Hotel Aspect Profiles    │
          └───────────────────┬────────────────────┘
                              │
                              v
          ┌────────────────────────────────────────┐
          │ Aspect Contribution / Alignment        │
          │ Analysis                               │
          └───────────────────┬────────────────────┘
                              │
                              v
          ┌────────────────────────────────────────┐
          │ Post-hoc Recommendation Explanation    │
          └────────────────────────────────────────┘
```

---

## Academic Context

This repository accompanies a Master's dissertation investigating the performance and interpretability of collaborative hotel recommendation.

The repository is intended primarily for **academic reproducibility and demonstration of the experimental pipeline**.

---

## References

Antognini, D. and Faltings, B. (2020) ‘HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset’, *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pp. 4917–4923.

Hu, Y., Koren, Y. and Volinsky, C. (2008) ‘Collaborative Filtering for Implicit Feedback Datasets’, *2008 Eighth IEEE International Conference on Data Mining*, pp. 263–272. doi:10.1109/ICDM.2008.22.

Koren, Y., Bell, R. and Volinsky, C. (2009) ‘Matrix Factorization Techniques for Recommender Systems’, *Computer*, 42(8), pp. 30–37. doi:10.1109/MC.2009.263.

Zhang, Y. and Chen, X. (2020) ‘Explainable Recommendation: A Survey and New Perspectives’, *Foundations and Trends in Information Retrieval*, 14(1), pp. 1–101. doi:10.1561/1500000066.

---

## Author

**Sirish Konda**  
Master's Dissertation Project
