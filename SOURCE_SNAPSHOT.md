# LEGOSpec shared-storage snapshot

This repository contains the current LiteSpec_exp source and generated project artifacts. The Python package remains `litespec`; original documentation and license are retained.

All regular project files up to 10 MiB are included, including results, logs, progress/verification receipts, reports, experiment configurations, and historical source snapshots. Environment directories, credentials, rebuildable compilation caches and absolute symlinks are excluded. Files above 10 MiB are excluded and listed in `PUBLISH_MANIFEST.json`; notably, full benchmark manifests and large W&B binary logs are not included. This is a source/results snapshot rather than a full model or dataset backup.

Runtime queue/PID files record historical processes and must not be treated as live state after cloning. Host-specific absolute paths need adaptation. DFlare+Parallel training was stopped at the user's request on 2026-10-21: CE last logged 23,850/59,724 steps; CE+TV did not start; subsequent testing is held. No completed result or final model is claimed for these two jobs.

The source directory, its Git branch/remotes, protected code/configuration and existing results were not modified for publishing. File hashes and exclusion reasons are recorded in `PUBLISH_MANIFEST.json`.
