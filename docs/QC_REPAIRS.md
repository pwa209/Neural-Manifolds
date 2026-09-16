# QC compatibility repairs

These repairs are label-blind signal diagnostics, not cohort admission or a
scientific result gate. Original releases remain sealed and unchanged.

## Invalid EDF clock field

Eight source EDFs in DREAM set 2 contain an invalid seconds value of 60.
For sampled QC only, a temporary copy replaces the eight clock bytes with a
parser placeholder. The reader then discards measurement date. All other header
bytes and the entire remaining file are checked unchanged. Original/copy hashes,
the changed byte range and the unavailable absolute clock are recorded.

The placeholder is **not** an estimate of the true start time. These copies must
not be used for absolute-time event alignment. No waveform repair, interpolation,
resampling or relabelling is involved.

## Non-UTF-8 EDF annotations

Only the specific invalid-annotation-byte error triggers a Latin-1 decoding retry
in sampled QC. The retry and unverified annotation semantics are recorded. This
does not authorize interpreting annotation text as experimental labels or events.
The normal reader's default remains unchanged. See the
[MNE EDF reader documentation](https://mne.tools/stable/generated/mne.io.read_raw_edf.html).

## RAR archives

The cluster's old p7zip can list but not decode this RAR5 archive. Libarchive
decoded five signals but raised block-checksum errors on two others. A current
7-Zip decoder is therefore prestaged separately and checksum-bound using
`configs/archive_reader.yaml`; no binary is redistributed in Git. The version
and download route were verified on the [official download page](https://www.7-zip.org/download.html).
Set `NEURAL_MANIFOLDS_7ZIP` to the extracted executable and
`NEURAL_MANIFOLDS_7ZIP_SHA256` to its pinned hash. A mismatch fails before use.
Without these settings, libarchive remains a compatibility fallback, not a
guarantee that all RAR5 blocks are supported. Unsafe/duplicate names,
links, encryption and malformed metadata fail explicitly. Each requested signal
is streamed to a generated temporary name, bounded by its inventoried size and
scratch allowance. Extraction must finish successfully, including integrity
checks, and the decoded bytes must independently match the inventoried CRC32.
The temporary recording is removed after inspection. Unsupported
multi-file recording formats remain explicit rather than flattened incorrectly.

The final cluster canary inspected all seven DREAM set 5 recordings with the
pinned decoder and no unavailable rows. Thus the two libarchive block-checksum
failures were decoder incompatibilities, not evidence of damaged source signals.

## Defective upstream ZIPs

DREAM sets 10 and 11 contain four recordings with decompression, CRC or local
header failures. A fresh cluster hash check matches both the sealed SHA-256
manifest and the recorded upstream MD5 for each archive. A live public metadata
check still reports the same version and checksums. Re-downloading identical
archives is therefore not a demonstrated repair.

CRC failures are not bypassed, waveforms are not reconstructed, and affected
recordings remain unavailable unless a verified replacement becomes available.
Other recordings continue independently. Successful job exit does not imply that
every recording was readable or scientifically eligible.
