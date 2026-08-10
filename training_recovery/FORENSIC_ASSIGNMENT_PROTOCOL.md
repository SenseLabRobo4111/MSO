# Preserved-target floorplan-assignment protocol

Frozen before the new assignment audit: 2026-08-07

The preserved sample identifiers contain worker IDs, not building IDs.  This
audit therefore attempts an image-geometric assignment from each preserved
`local_map_0.png` target to one of the 156 source `GT.bmp` floorplans.  It is a
forensic test, not permission to infer identities from filenames.

Only the preserved `train` directory is used while the method is assessed.
The preserved `test` directory is not accessed.  Each source is represented by
both its directly thresholded dark-obstacle raster and the visible preprocessing
implemented in the accessible KTH reader/simulator.  The latter is only a
candidate source transform; it is not asserted to be the missing historical
pipeline.

ORB supplies a broad candidate shortlist.  SIFT and a similarity-transform
RANSAC then verify the shortlist.  For every sample the audit records top-1 and
top-2 source IDs, feature matches, inliers, inlier ratio, median reprojection
error, recovered scale and rotation, binary agreement, and bidirectional edge
agreement after warping the source into the target frame.

An initial one-sample implementation smoke exposed the standard repeated-line
degeneracy in which RANSAC can collapse many matches onto one point with a
near-zero scale.  Before any cohort audit, transforms are therefore declared
mathematically invalid unless their uniform scale is in `[0.1, 10]` and their
inlier support spans at least 64 target pixels and 20 source pixels.  Invalid
transforms receive zero verified inliers.  This is a transform-validity check,
not a relaxation of the acceptance thresholds below; the superseded smoke is
retained separately in the remote evidence directory.

A sample is accepted only when all of the following frozen gates hold:

- at least 40 SIFT-RANSAC inliers;
- SIFT inlier ratio at least 0.35;
- median inlier reprojection error at most 3 source pixels;
- top-1 minus top-2 inlier margin at least 25 and top-1/top-2 ratio at least
  1.5;
- bidirectional edge F1 within three pixels at least 0.55; and
- valid-overlap binary IoU at least 0.45.

These deliberately conservative gates favor leaving samples unassigned over
manufacturing building labels.  A floorplan/building split may be constructed
only if a substantial fraction passes and accepted assignments remain stable
under audit.  Ambiguous and unassigned samples must be excluded.  If the gates
fail broadly, the result is evidence that the current 156-floorplan collection
or visible preprocessing cannot authenticate the preserved archive.
