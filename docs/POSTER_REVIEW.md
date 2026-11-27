# Three-column poster: methods and result review

The poster follows course Template C (BlueColumns), at 80 × 100 cm. It retains the hot-topic prediction experiment and adds shared-post clustering and author-removal sensitivity. Conclusion states observed findings; Future Work contains proposals, not completed experiments.

| Poster statement | Implementation / evidence | Check |
|---|---|---|
| 254,624 posts, 199,851 users, 91 days; 1,422 eligible topics | `data/weibo_release_manifest.json`, `results/experiments/topic_design.json`, `src/mining_experiment.py` | Input hash and original descriptive inputs checked locally; topic threshold is 30 posts / 5 users. |
| 57-dimensional fused representation; selected KMeans, k=7 | `src/mining_experiment.py`, `results/experiments/selected_clustering.json` | 16 text dimensions plus 41 behavior dimensions. KMeans/Ward comparison selects the seed-28 fused silhouette winner. |
| C3: 155 topics; silhouette 0.797 vs. 0.078 | `EXPERIENCE.md`, original local `topic_features.csv` and `topic_representations.npz` | Recomputed silhouette in the original 57-D feature space. Display C3 means raw cluster label 2. |
| 35 hashtags share 103 posts; another 35 share 102 of them; 5 authors; median 135 extracted tags | Original normalized Weibo input, project hashtag extraction rule | Recomputed exact post-ID sets and author/tag counts locally. Counts depend on the project's extraction rule. |
| 18/155 C3 topics have top-quartile HMI | Original feature table and public `results/experiments/topic_tsne_display.csv` | Count verified. The display CSV contains no topic or author identifiers; its row order must not be joined to the original feature table by position. t-SNE coordinates are a display export, not a new clustering method. |
| HMI formula and components | `src/mining_experiment.py`, `docs/METHODS.md` | Mean of percentile ranks of duplicate share, author HHI, peak-day share and same-minute share. Not a manipulation probability. |
| Posting shares and absolute score changes | `src/misbot_author_influence.py`, `results/experiments/author_influence_summary.csv`, `author_influence_validation.json` | All plotted values agree with saved results. Small-gap n=1,031; large-gap n=143. After top-5 removal, score-change medians use 900 / 143 nonempty topics. Shares use full cohorts. |
| 131/143 (91.6%) move closer after top-1 removal | Same author-influence evidence | Reference is the original equal-author score, held fixed. Raw private style-score tables are unavailable in this checkout, so this review did not rerun the complete style scoring/removal pipeline. |
| 54 itemsets at 0.3% support | `results/experiments/pattern_method_comparison.csv` | Apriori and FP-growth saved results agree. Not rerun in this poster review. |
| AP 0.148 → 0.237; hits 44 → 84/175 | `results/experiments/forecast_metrics.csv`, `forecast_summary.json` | Primary Windows test run, rounded to 3 decimal places. This is ranking AP, not classification accuracy. No RF retraining in this review. |
| HMI block +0.0013 AP; candidate coverage 30.7%; overall recall 3.38% | Same forecast evidence | 763/2,485 candidate coverage, 84/2,485 overall recall. HMI block includes its four rolling components. |
| Past-only prediction; six train-fitted activity/coordination groups; no MisBot | `src/forecast_experiment.py`, `docs/METHODS.md` | This is the original full RF experiment. It differs from the seven retrospective text/behavior groups and from the supplementary seven-group validation experiment. |

The normalized Weibo SHA-256 matches the public release manifest. Neither t-SNE separation, high HMI, shared posts nor style sensitivity establishes malicious coordination or CIB. The poster does not claim that a few accounts posted hundreds of times: the checked sample has at most 42 posts per author within an eligible topic.

The source, root PDF, `output/pdf/Poster.pdf`, preview and two vector charts are updated together. Report content is unchanged. `src/validate_deliverables.py` checks the new headings, geometry, matching PDF copies and vector charts. Original PDFs remain available in Git history.
