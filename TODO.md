- Understand why nudge applied %age is so low even when succesful retrieval (i.e. > threshold for cosine sim) are much higher (~40%)
- Understand how to get the nudge applied %age up
- Understand its impact

- Augment learning rate? between 2x and 5x
- Plot mean nudge norm, nudge applied %age and retrieval success as an average of the last 200 attempts (not more, not less) and find a way to get rid of the starting noise.

- KEEP THE NUDGED ACC GOING UP!!!!!
- track hyperparams carefully for each change.

- limit run size to 10k params for hyperparam picking with automated stopping, then run a final run?

- carefully annotate each run, and do a hyperparam sweep on dimensions, index size, etc
- get rid of guided accuracy on results? or at least on test set