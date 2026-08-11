# Prospective v2 immutable lock reference

The machine-readable final lock is stored on a private experiment host. Its
independently verified SHA-256 bindings and test-disabled status are recorded
without publishing host identifiers, private filesystem paths or the source
floorplan inventory.

The remote lock root and directories are mode `0555`; all lock, inventory,
split, environment, source, and positive-control files are mode `0444`.
`v2_lock.sha256` verifies `v2_lock.json`, whose SHA-256 is
`9a4a308144abd3f43d37e7c702717fdac56590ebc5497667c4946463c679dd8b`.

The final source bundle contains 18 hashed files and no legacy generator CLI
with seed, sample-count, attempt-count, optimizer, epoch, resume, or hash-skip
overrides.  Geometry helpers live in a no-entrypoint module.  The earlier
`v2_lock` directory on the host is superseded and must not be executed.

This record does not claim historical recovery.  No prospective-v2 smoke,
full generation, training, or test evaluation had been run when the lock was
frozen.  The numeric-prefix grouping remains operational only and is not
verified semantic building identity.
