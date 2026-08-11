# Required 304,000-parameter model artifact audit

Audit date: 2026-08-11.

## Conclusion

No verified 304,000-parameter MSO artifact or matching exact architecture was
found. The field package therefore records:

```text
artifact_status: missing_required_artifact
collection_ready: false
```

No model binary is included in this subtree. The label `mso_304k` identifies a
prospective treatment and exact 304,000-parameter gate; it is not evidence that
such an artifact has been recovered.

## Search boundary

The audit covered the active local MSO workspaces, the accessible remote MSO
training-host search roots, the retained inference candidates, and the
approximately 20 GB weight archive supplied for recovery. The inspected
archive has SHA-256:

```text
d82f18a605fed210efaafb421b4235776cf0226d056868d3e89ad0721c07cab8
```

The archive's `iros/model` groups contain four artifacts in each of six model
classes. Their observed trainable-parameter counts are:

| Group | Artifacts | Trainable parameters per artifact |
|---|---:|---:|
| `m` | 4 | 18,684,048 |
| `s` | 4 | 8,094,816 |
| `n` | 4 | 4,509,384 |
| `xn` | 4 | 1,997,872 |
| `xxn` | 4 | 847,576 |
| `xxxn` | 4 | 511,128 |

Six additional configuration checkpoints resolve to:

| Exact trainable parameters | Checkpoints |
|---:|---:|
| 709,184 | 2 |
| 322,371 | 2 |
| 342,771 | 2 |

None equals 304,000.

## Known 342,771-parameter candidates

The two retained generator candidates have the following artifact digests:

```text
da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8
022689a97336b47fe0c3a39d85e120a4d07e99ba3cc34e4f2be5422c1e515e2c
```

They strictly target the recovered 342,771-parameter architecture, are not
identified as the manuscript checkpoint, and are explicitly denied by
`validate_campaign.py`. Renaming or rounding either artifact would not create a
304,000-parameter model.

## Why the historical table row is insufficient

The manuscript source contains a historical `w/o FFC` resource row reported as
`304K`. It is a rounded aggregate table value, not an inference artifact,
architecture definition, state dictionary, split manifest or selection record.

The four identified FFC blocks in the 342,771-parameter reconstruction account
for 303,160 parameters. Mechanical removal would leave only 39,611 parameters,
not 304,000, and would still not establish the historical forward graph or
weights. The experiment package therefore does not synthesize a replacement by
deleting modules.

## Conditions for changing NO-MODEL status

The status may change only if all of the following are available together:

1. an authoritatively identified artifact with a distribution right;
2. its exact architecture implementation and input/output contract;
3. strict state loading with no missing or unexpected keys;
4. an actual trainable-parameter count of exactly 304,000;
5. a finite forward pass on a frozen 256 x 256 fixture;
6. artifact, fixture, provenance and selection-record SHA-256 digests;
7. a deterministic output fingerprint;
8. a machine-readable report from an explicitly reviewed loader in
   `verify_model_artifact.py`.

Hand-editing `model_lock.yaml` is insufficient because the campaign validator
recomputes file digests and checks the verifier report fields.

## Distribution boundary

Possession of a weight file does not establish permission to publish it. Before
any future artifact is committed, confirm ownership, training-data obligations,
third-party terms and the intended model licence. If public distribution is not
authorized, retain only the digest and access procedure in the public branch
and provide the artifact through an approved controlled channel.
