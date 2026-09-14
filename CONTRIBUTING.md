# Contributing to cuspa

External contributions are open for bug fixes, documentation improvements, and
focused features submitted through pull requests. All project participants
must follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Do not report security vulnerabilities through public issues, discussions, or
pull requests. Follow the private reporting process in
[SECURITY.md](SECURITY.md).

## Contribution scope

Open a [GitHub issue](https://github.com/NVIDIA/cuspa/issues/new/choose) before
starting a large change, changing the public API, adding a dependency, or
changing compatibility or performance behavior. Keep pull requests focused and
include tests and documentation for changed behavior.

New source files must include the NVIDIA SPDX copyright and Apache-2.0 license
headers used by the existing source files.

## Development setup

Install the CUDA and compiler requirements described in the
[installation guide](docs/install.md). Install CuPy for your CUDA runtime, then
install cuspa and the contribution tools. Choose one CuPy package:

```bash
python -m pip install "cupy-cuda12x[ctk]>=14"  # CUDA 12
# or: python -m pip install "cupy-cuda13x[ctk]>=14"  # CUDA 13
python -m pip install -e ".[test]"
python -m pip install pre-commit build twine
pre-commit install
```

## Tests and checks

Run pre-commit for every change:

```bash
pre-commit run --all-files --show-diff-on-failure
```

For code changes, run the tests on a supported GPU:

```bash
python -m pytest -q
```

For native or packaging changes, build and validate a local wheel:

```bash
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=native' python -m build --wheel
python -m twine check --strict dist/*.whl
```

GPU tests require CuPy. Adapter tests are enabled when their optional
dependencies are installed.

For documentation changes, build the documentation and check its links:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs docs/_build/html
sphinx-build -b linkcheck -W --keep-going docs docs/_build/linkcheck
```

## Pull request process

1. Fork the repository and create a topic branch from `main`.
2. Write commit messages that explain what changed and why.
3. Sign off every commit as described below.
4. Run the applicable tests and checks.
5. Open a pull request targeting `main`. Explain the purpose of the change and
   link the related issue when applicable.

cuspa maintainers review pull requests and may request changes. A maintainer
merges a pull request after the required checks pass.

## Pull request CI

cuspa uses NVIDIA's ephemeral self-hosted runners. For security, workflows on
those runners do not execute directly from `pull_request` events. After
reviewing the latest changes, a maintainer starts CI by commenting on the pull
request with its latest commit SHA:

```text
/ok to test <SHA>
```

NVIDIA's `copy-pr-bot` copies that exact commit to a temporary
`pull-request/<number>` branch. CI builds CUDA 12 and CUDA 13 wheels for x86_64
and aarch64 on NVIDIA CPU runners, then tests the x86_64 wheel artifacts on
NVIDIA GPU runners. Every new commit requires a new review and
`/ok to test <SHA>` comment.

## Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be
    accepted.

* To sign off on a commit, use the `--signoff` or `-s` option:

  ```bash
  git commit -s -m "Fix polygon assignment"
  ```

  This appends:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

  The name and email in the sign-off must match the commit author.

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
