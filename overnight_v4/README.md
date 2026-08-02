# Overnight suite v4

This suite does not switch the main checkout. It prepares four detached,
pinned worktrees and then runs the following sequence:

1. exp022 proxy seed 0;
2. select exp021 or exp022 using the registered loss-vs-training-time rule;
3. exp023 FP8 integration benchmark only;
4. exp016 causal A and conditional systems B;
5. selected BF16 full seed 0 at exactly 2,700,083,200 tokens.

No FP8 proxy/full and no exp016-C training stage exist in this launcher.
Failure of exp023 or exp016 does not block the independent BF16 full. An
incomplete exp022 proxy selects the already completed exp021 result. A failed
full leaves the instance running for inspection.
