# Releasing cuspa

cuspa is published to pypi.org and pypi.nvidia.com as three projects: `cuspa`
(the source distribution), `cuspa-cu12`, and `cuspa-cu13` (x86_64 and aarch64
wheels). NVIDIA's release service publishes the files from the cuspa GitHub
release, which is immutable once published. Users therefore install exactly
the files that CI built and that passed the security checks.

## Before releasing

- The wheel builds and the other required checks pass on `main`.
- The release notes in `docs/release-notes/<version>.md` are complete, and
  sections without entries are removed. For a final release, replace
  `the-future` in the heading with the release date, for example
  ``### 0.1.0 {small}`2026-11-03` ``. A release candidate keeps `the-future`.
- Immutable releases are enabled in the repository settings.
- The `publish` environment requires a maintainer's approval. It and the
  `release-staging` environment hold the release service credentials.
- The maintainer who runs the security checks has access to NVIDIA's internal
  release tooling.

## Cutting a release

1. **Tag the release.** Tag a commit on `main` and push the tag. Use `vX.Y.Z`,
   or `vX.Y.ZrcN` for a release candidate. The package version comes from the
   tag.

   ```bash
   git tag v0.1.0rc1 origin/main
   git push origin v0.1.0rc1
   ```

2. **Prepare Release** builds and tests the wheels and the source distribution,
   checks them with twine and abi3audit, and attaches them to a draft GitHub
   release together with `SHA256SUMS` and `manifest.json`.
3. **Publish the draft.** Review the release notes, then publish the draft.
   Publishing makes the release immutable and starts **Publish Release**, which
   checks the release against the Prepare Release build and runs a dry run of
   the upload for each project. It also starts **Docs**, which adds the
   release to the documentation site; see [DOCS.md](DOCS.md#versions).
4. **Run the security checks.** With NVIDIA's internal release tooling, register
   the published files for scanning, wait for the scans to finish, and run the
   release readiness check. A pass sets the `release/review` commit status for
   the reviewed manifest. A failure lists the unmet requirements; resolve them
   and run the readiness check again.
5. **Approve the upload.** Approve the `publish` environment in the Publish
   Release run. The workflow requires the `release/review` status, repeats the
   dry run, uploads each project, and waits until pypi.org and pypi.nvidia.com
   serve every file with the SHA-256 that CI built.
6. **Check the release.** In a fresh environment on a GPU machine, install the
   wheel for each CUDA version and import it:

   ```bash
   [uv] pip install cuspa-cu12==0.1.0rc1  # or cuspa-cu13
   python -c "import cuspa; print(cuspa.__version__)"
   ```

Publish within 90 days of tagging. Publish Release compares the release with the
Prepare Release artifact, which GitHub keeps for 90 days.

## If Publish Release cannot reach the release service

The workflows reach the release service through a network tunnel. If it is
unavailable, a maintainer can run the same checks and the upload with NVIDIA's
internal release tooling.

## When something fails

- Uploads happen one file at a time and are not rolled back. A rerun uploads
  only the files that neither index serves yet, and stops if a file is on only
  one index.
- A published release cannot change. If a file is wrong, fix it and release a
  new version.
- PyPI rejects new files for a version more than 14 days after the first file of
  that version was uploaded.
