- Understand why nudge applied %age is so low even when successful retrieval (i.e. > threshold for cosine sim) are much higher (~40%) --> Because of the nudge classifier, which determinese a sigmoided usefulnessm of mean 0.605. We set a threshold at 0.6, which filters a lot of the nudges that fit the cosine metric

- Understand how to get the nudge applied %age up --> Lower nudge classifier threshold, or even ablate it.
- Understand its impact

- Augment learning rate? between 2x and 5x --> Do that after determining whether the ablation is useful.
- Plot mean nudge norm, nudge applied %age and retrieval success as an average of the last 200 attempts (not more, not less) and find a way to get rid of the starting noise.

- KEEP THE NUDGED ACC GOING UP!!!!!
- track hyperparams carefully for each change.

- limit run size to 10k params for hyperparam picking with automated stopping, then run a final run?

- carefully annotate each run, and do a hyperparam sweep on dimensions, index size, etc
- get rid of guided accuracy on results? or at least on test set