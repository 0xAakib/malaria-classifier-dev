# Answers on the study design

<!-- Answer every question under its **Answer:** line. Keep the ## headings as they are: the review finds your answers by them. -->

## Q1

Why does the study match optimizer steps instead of epochs across label fractions? What would a 2% run look like if epochs were matched?

The study matches optimizer steps instead of epochs to ensure that the model trained on small number of data also get as much learning steps as the model trained on large number of data.


## Q2

BBBC041 is over 95% uninfected. Why is plain accuracy the wrong metric, and what is used instead?

BBBC041 is over 95% uninfected, so the model can just learn to say uninfected everytime and will get the 95% accuracy, which is not a good metric.


## Q3

What would go wrong if the train/test split were made per cell instead of per source image?

**Answer:**


## Q4

The four methods differ in architecture and pretraining at once. What can the results NOT claim?

<!-- Tick one option: change its [ ] to [x]. -->

- [ ] A: Which of these four specific models is most label-efficient
- [ ] B: Whether ViTs are more label-efficient than CNNs in general
- [ ] C: Nothing; every claim is valid


## Q5

Give one result that would count as a publishable null result for this study.

**Answer:**

