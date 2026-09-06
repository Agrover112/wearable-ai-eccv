# Documentation index

Start here:

- `REPRODUCIBILITY_GUIDE.md`: rebuild the environment, final model, and test
  container from the preserved records.
- `IMPLEMENTATION_CATALOG.md`: map implementation families to source files.
- `RUN_LOG.md`: chronological experiment outcomes.
- `STORAGE_CLEANUP_AUDIT.md`: protected artifacts, removed caches, and cleanup
  receipts.
- `SCRATCH_ASSET_RESTORATION.md`: recreate the deleted dataset, model snapshots,
  exploratory source checkout, and final image build inputs.
- `PAPER_PLANNING_SKELETON.md`: meeting worksheet for choosing the paper story,
  research questions, evidence, figures, and claims.
- `PROJECT_WEBSITE_CONTENT_MAP.md`: planning-only website structure and asset
  review checklist.

`reproducibility_manifests/` contains generated machine-readable records.
`cleanup_records/` contains the before/after inventory for destructive storage
cleanup. Historical Slurm launchers, scheduler logs, model caches, datasets,
and prediction outputs are intentionally excluded from this repository. Dated
meeting briefs and experiment notes remain available here as project history.
The final container source and recovery instructions are under
`../../test_submission/qwen35_27b_dual_view_fusion_v2/`.
