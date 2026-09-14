# Contributing

## How to contribute

1. **Open an issue** before starting a large change.
2. **Fork** the repository and create a topic branch for your change.
3. Keep changes focused and add tests.
4. **Sign off** every commit (see below) and open a pull request.

Before submitting a change, run:

```bash
pre-commit run --all-files
python -m pytest -q
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=native' python -m build --wheel
python -m twine check dist/*
```

GPU tests require CuPy. Adapter tests are enabled when their optional
dependencies are installed.

## Pull request CI

cuspa uses NVIDIA's ephemeral self-hosted runners. For security, workflows on
those runners do not execute directly from `pull_request` events. After
reviewing the latest changes, a maintainer starts CI by commenting on the pull
request with its latest commit SHA:

```text
/ok to test <SHA>
```

NVIDIA's `copy-pr-bot` copies that exact commit to a temporary
`pull-request/<number>` branch. CI then builds the CUDA 12 and CUDA 13 wheels on
NVIDIA CPU runners and tests those same wheel artifacts on NVIDIA GPU runners.
Every new commit requires a new review and `/ok to test <SHA>` comment. The
temporary branch is removed when the pull request is closed or merged.

## Sign off commits

* Sign off every commit to certify that you have the right to submit the work
  under this project's license.

* Add the sign-off with `--signoff` or `-s`:

  ```bash
  git commit --signoff -m "Fix polygon assignment"
  ```

  This appends:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the [Developer Certificate of Origin](https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
