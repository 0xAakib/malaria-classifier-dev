# Answers on the study design

<!-- Answer every question under its **Answer:** line. Keep the ## headings as they are: the review finds your answers by them. -->

## Q1

Why does the study match optimizer steps instead of epochs across label fractions? What would a 2% run look like if epochs were matched?

The study matches optimizer steps instead of epochs because when using the same number of epochs, a model trained on 2% of the data would receive far fewer gradient updates (optimizer steps) than a model trained on 100% of the data. If epochs were matched, the 2% run would have roughly 50x fewer optimizer steps, causing a drop in performance not due to data size but insufficient training time for the model to converge. This would confound the effect of fewer labels with less training.


## Q2

BBBC041 is over 95% uninfected. Why is plain accuracy the wrong metric, and what is used instead?

BBBC041 is over 95% uninfected, so a naive classifier that always predicts "uninfected" would achieve >95% accuracy without learning anything meaningful. This makes plain accuracy misleading. The study uses Balanced Accuracy (average of per-class recall) and MCC (Matthews Correlation Coefficient) instead, as these metrics properly account for class imbalance.


## Q3

What would go wrong if the train/test split were made per cell instead of per source image?

If the train/test split is made per cell, cells from the same slide or source image can end up in both train and test sets. This means the model is tested on data it has effectively already seen during training, causing the model performance to be overestimated.


## Q4

The four methods differ in architecture and pretraining at once. What can the results NOT claim?

<!-- Tick one option: change its [ ] to [x]. -->

- [ ] A: Which of these four specific models is most label-efficient
- [x] B: Whether ViTs are more label-efficient than CNNs in general
- [ ] C: Nothing; every claim is valid


## Q5

Give one result that would count as a publishable null result for this study.

We expect pretrained models to be more efficient. But even if the curves for the from-scratch CNN and the DINOv2 model look exactly the same, it would still be a publishable null result because it would show that pretraining does not provide any advantage in this specific setting.

