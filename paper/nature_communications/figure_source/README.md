# Nature Communications figure source

Run from the submission directory:

```powershell
py -3 figure_source/export_drawio_figures.py
py -3 figure_source/make_system_figures.py
py -3 peer_review_data/mrpb/plot_exploration_results.py
```

The first command exports the author-supplied `fig1` and `fig2` pages directly
from `source_assets/figs.drawio` (or the identical project-level source when
present) on a true-white canvas. It changes no diagram element and writes
600-dpi PNG and LZW-compressed TIFF files.

The second command creates six-panel Main Figs. 3, 4 and 8 as editable SVG and
PDF files, 300-dpi PNG previews and 600-dpi LZW-compressed TIFF files. In Main
Fig. 3, panels a--f show operation entries versus FID, parameters versus LPIPS,
exact resource ratios, a 4-by-5 raw metric matrix, a 5-by-7
direction-adjusted change matrix and one unified runtime axis. Inputs are the
exact retained model/runtime values, event-level transform audit, aggregate
scaling/KTH CSVs and physical endpoint records packaged with the manuscript.

The third command regenerates the full-width source-backed metric row used in
Main Fig. 6. It draws all nine retained coverage records and the six records
with predictive occupied-cell precision. The `MSO multi, ORB` and `MSO no
merge` records coincide through the displayed horizon and are distinguished
with sparse markers, without offsetting either trace. The legacy OBS curve is
not regenerated because no retained CSV exists. No interpolation, smoothing,
resampling or synthetic endpoint is applied.

No synthetic error bars, seed-level dispersion or significance tests are added. Figure captions must retain the stated evidence boundaries.
