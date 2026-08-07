# Preserved archive inventory

`preserved_archive_6841_1463.csv` inventories the closest preserved archive.
The existing directory labels are repeated in `split` and
`archive_partition`, with `split_status` fixed to
`preserved_archive_label_not_manuscript_split`. The numeric name prefix is
explicitly stored as `source_worker_prefix`. None of these fields is a
semantic building/floorplan group.

This inventory is not the manuscript's reported 5,385/1,356 split. Use
`../tools/build_split_manifest.py` with author-verified semantic metadata to
create a new reconstructed canonical split.
