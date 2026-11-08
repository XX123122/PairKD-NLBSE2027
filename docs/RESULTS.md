# Evidence and interpretation

The CSV files preserve accepted author results; three are byte-identical copies. The hardware inventory replaces local absolute workspace paths with author_archive/ while preserving metric/device/checkpoint fields. Original and repository hashes are listed in `provenance/INPUTS.json`. They contain aggregated metrics and runtime records, not raw dataset rows or prediction arrays.

`T4_20261008_ALL3.csv`: original PKP2 and two narrow three-layer candidates, one public pass each in the same T4 session. PKP2 Mean F1 .9084194916924127, time 33.16337677800004 s, GFLOPs 444468.6284385284, reference score .9087877777639457. N13 has higher F1 but a lower reference score. n=1 does not establish timing variance.

`T4_20261007_ALL20.csv`: all ten paired rounds for PKP2/PKP3; keep all rounds. PKP2 mean reference score .9074469936, PKP3 .8966633048. Do not substitute the newer single time for a ten-round mean.

`THRESHOLD_POLICY_COMPARISON.csv`: saved-output comparisons of existing operating points, not a new public threshold fit.

`T4_PLOT_HARDWARE_AUDIT.csv`: the 15 plotted public deployment configurations are T4 measurements across different sessions. Training/internal CAL/AUDIT used RTX 4090; local weight edits used RTX 3050 Laptop GPU. The paper's weight-edit comparisons use internal rows and do not replace public T4 results.

Reference score: S=.6 MeanF1+.2 max(1-t/1000,0)+.2 max(1-GFLOPs/3000000,0). Historical executable accounting uses two forwards per batch and divides by two. Forward-only timing excludes model load, preprocessing, H2D, sigmoid and reporting. Profiler output is not complete operator accounting. Hidden evaluation and final organizer acceptance remain separate.
