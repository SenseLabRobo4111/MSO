# Public raw-data access record

## Verified access

The raw ROS 2 records are available in a [public read-only Google Drive
folder](https://drive.google.com/drive/folders/1mbCuIISidEy87mmWPbZTtiRKfW54Fhii?usp=sharing). The folder contains 18 SQLite
bag files, 18 associated `metadata.yaml` files, `README_DATA.txt` and
`RAW_ROSBAG_SHA256SUMS.md`.

The 36 data files total 4,438,635,447 bytes (4.134 GiB); including the two root
documents, the folder contains 38 objects totalling 4,438,641,736 bytes.
Upload verification reported 38 matching objects and zero transfer
differences. An unauthenticated session returned HTTP 200 for the folder,
displayed the expected folder name and downloaded `README_DATA.txt` with the
same SHA-256 value as the staged source.

## Evidence groups

| Group | Folder | Bag files | Bag bytes | Intended review use | Evidence boundary |
|---|---|---:|---:|---|---|
| Controlled single-robot physical comparisons | `Exp/arena1/` | 8 | 2,689,044,480 | Inspect the retained IG-Hector, MapEx, UPEN and MSO recordings and reproduce the archived map-raster diagnostic | One arena; unequal included-run counts; the metric is not region-of-interest-clipped arena coverage |
| Archived two-robot registration and deployment recordings | `Exp/experiment_video/rosbag/` | 10 | 1,749,336,064 | Re-run the documented negative offline registration audit and inspect recorded topics | No online transform proposals, gate decisions or commit events; reconstructed transforms are offline |

The folder does not supply missing historical training splits, the manuscript
checkpoint, simulation seed-level records or unrecorded registration events.
The SHA-256 inventory gives the exact relative paths, byte sizes and hashes for
all 36 data objects.

## Access and reuse boundary

The folder is publicly viewable and downloadable without an account. The raw
records are not covered by the repository's BSD-3-Clause software licence, and
no separate data-reuse licence is asserted. Requests for uses beyond
manuscript verification should be directed to Fei Qiao
(`qiaofei@tsinghua.edu.cn`). No expiry date is configured.
