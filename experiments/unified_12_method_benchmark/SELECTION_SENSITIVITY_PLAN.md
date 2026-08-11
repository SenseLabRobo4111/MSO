# Teacher-checkpoint selection sensitivity plan

Status: declared on 10 August 2026 while the teacher stage of the frozen
full-objective campaign was still running. No sensitivity student run had been
started at declaration time.

## Rationale

The active prospective reconstruction selects the teacher checkpoint by
minimum validation unknown-region BCE. Through epoch 189 that rule still chose
epoch 0, whereas unknown-region F1 reached its current maximum at epoch 167.
The archived historical checkpoint, by contrast, records epoch 499 and its
callback has no validation monitor. Thus minimum-BCE selection is a declared
reconstruction choice, not a recovered historical rule.

## Frozen comparison

The active minimum-BCE branch remains unchanged and all its results must be
reported. A separate sensitivity branch may be launched only after the teacher
finishes and the final-epoch teacher checkpoint has been frozen by SHA-256.

The sensitivity branch will:

- use the same completed teacher trajectory but the epoch-499 final teacher;
- keep the same architecture, projections, critic initialisation policy,
  training split, loss equations, optimiser, augmentation and five student
  seeds (11, 23, 37, 53 and 71);
- keep validation BCE as the per-student checkpoint rule so that only teacher
  selection changes;
- evaluate all five students with the same frozen evaluator;
- retain every seed and report the paired per-seed difference between teacher
  selection branches with no test-set-driven tuning;
- label the result as a checkpoint-selection sensitivity analysis, not as an
  authenticated historical reproduction.

No F1-selected teacher branch is declared here because the current campaign
does not retain every historical epoch checkpoint and the F1 peak already
observed would make that choice post-hoc. A future F1-selected experiment would
require a new teacher run and a separately frozen protocol.
