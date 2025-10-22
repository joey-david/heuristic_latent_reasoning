Lean kNoT TODO (status)

Completed:
- [x] Build hidden-state cache: `knot/pipeline.py` now caches single-pass latents (answer, `h_T`, token count, metadata) to JSONL per split.
- [x] Stand up FAISS indexing: `knot/retrieval.py` supports saving/loading indices; the pipeline exposes a `build_index` step.
- [x] Implement vote aggregators: configurable uniform/softmax weighting with base-answer prior λ.
- [x] Gate (optional but quick): logistic regression gate (scikit-learn) with feature dumps for inspection.
- [x] Evaluation harness: `knot_report.py` + `configs/report.yaml` compare Direct/CoT/Coconut/kNoT (metrics, tokens, flip matrix plots).
- [x] Drop CLI args for kNoT entrypoints; everything is driven via config files (`configs/runner.yaml`, `configs/report.yaml`).
- [x] Cleanup: removed legacy heuristic nudging modules, refreshed README with the lean pipeline and preserved Coconut notes.

Pending (requires running inference):
- [ ] Lock down baselines: reuse `coconut.py` harness to record current GSM8K / StrategyQA / Date Understanding accuracy (Direct vs CoT vs Coconut).
- [ ] Execute extraction/index/eval runs once you're ready to spend tokens (`python knot.py`).
- [ ] Generate updated reports after the first sweep (`python knot_report.py`).

Notes: plumbing is ready; the next actions just need dataset paths + model checkpoints before launching inference.
