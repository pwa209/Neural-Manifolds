# Server dependency lock

`server-py311-cu126.lock` is the complete, transitive Python 3.11 lock for the
Ubuntu 22.04 CUDA 12.6 accelerator runtime. It contains only pinned
wheel-resolvable packages, cryptographic hashes, and explicit PyPI/PyTorch CUDA
12.6 indexes. The lock also
includes the test and build tooling used by the remote bootstrap's collection
check and no-build-isolation project install.

Generate it with the exact command recorded in the first two comment lines of
the lock. The target is `x86_64-manylinux_2_28`; changing Python, CUDA, platform,
or `pyproject.toml` requires a newly reviewed lock and a new SHA-256.

Current SHA-256:

```text
e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e
```

The remote bootstrap verifies this hash before creating a content-addressed
environment and installs the lock with `--require-hashes --no-deps` and
`--only-binary=:all:`. Do not hand-edit package versions or hashes.
