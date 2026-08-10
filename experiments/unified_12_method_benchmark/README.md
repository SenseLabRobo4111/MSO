# Unified 12-method occupancy-completion benchmark

This is a new prospective benchmark over the preserved 6,841/1,463 archive.
It is not the unrecovered historical 5,385/1,356 split and must not be mixed
with the archived values currently shown in Main Figure 3.

The first remote stage freezes the dataset inventory and exact upstream source
revisions for eight additional methods. Training and evaluation adapters must
subsequently enforce the same input raster, measured-cell copy-back rule,
metric implementation, hardware profiler and split for all twelve methods.

## Complete MSO objective reconstruction

The uniform-objective benchmark intentionally gives every architecture the
same BCE/Dice objective.  It is therefore not a test of the complete MSO
training method.  `run_full_mso_campaign_remote.sh` schedules a separate,
strictly labelled reconstruction after the uniform benchmark finishes:

1. train one width-32 teacher from deterministic random initialisation for 500
   epochs on the current train split;
2. freeze the validation-best teacher and a shared, fixed-seed set of four
   1-by-1 feature projections;
3. train five width-4, 342,771-parameter students with the reported masked
   adversarial, ADE20K ResNet50Dilated feature, masked L2 and four-tap
   distillation objectives;
4. evaluate every seed with the same probability-output evaluator as the
   uniform benchmark; and
5. save raw 500-epoch curves, seed-level tables, hashes and SVG/PDF/PNG/TIFF
   visualisations.

The reported Adam settings (learning rate 0.001, betas 0.5/0.999), batch size
16, 500 epochs, loss weights 10/30/1/5 and flip/90-degree rotation augmentation
are frozen in `full_mso_config.json`.  The adversarial mask is max-pooled to
the critic grid so that the real, generated and generator logistic terms are
each averaged over patches intersecting the unknown region, as defined by the
paper equations.  The archived LaMa R1 regulariser remains an all-real-patch
gradient penalty and is recorded separately in the protocol.

This remains a prospective paper-equation reconstruction, not an
authentication of the lost historical trainer.  The preserved candidate proves
the teacher/student topology and critic state tree, while the archived LaMa
source and ADE20K weights recover the perceptual objective.  The historical FFC
critic forward function was not found; the declared channel-mean patch readout
is therefore a reconstruction.  No historical teacher, projection, student or
critic weights initialise the prospective runs.  The current 6,142/699/1,463
split is also not the unrecovered manuscript split.  It is a sample-hash split,
not a verified building- or floorplan-disjoint split, so its results are a
within-archive benchmark rather than evidence of building-level generalisation.
