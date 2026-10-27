# Documentation site

The cuspa documentation is published with GitHub Pages at
https://nvidia.github.io/cuspa/. Sphinx builds it from `docs/`.

## Versions

The site holds one copy of the documentation per version:

- `latest/` is the documentation on `main`.
- `<version>/`, for example `0.1.0/`, is the documentation of a published
  GitHub release, built from its tag. Every final release has one. A release
  candidate has one while it is newer than the newest final release. Draft
  releases have none.
- `stable/` is the newest final release.

The site root redirects to `stable/`, or to `latest/` until the first final
release. So do the top-level addresses the site used before it had versions,
such as `install.html`. The version switcher in the header reads the list of
versions from `switcher.json` at the site root. Pages of other versions than
`stable/` show a banner that links to it.

## How the site is published

The **Docs** workflow (`.github/workflows/docs.yml`) runs on every push to
`main`, when a release is published, and when started by hand. On a
GitHub-hosted runner, `ci/build_docs_site.py` builds `latest/` from `main` and
rebuilds every published release from its tag. GitHub's Pages actions then
deploy the site.

Sphinx builds `latest/` with `-W`, so a warning fails the build and the
published site stays as it was. Releases build without `-W`, so a new warning
in an old release cannot block the site. Each release build installs the
documentation dependencies as they were on its tagged commit (uv's
`--exclude-newer`), so new versions of Sphinx or the theme cannot break it.

Publishing a release starts **Publish Release**, and its start triggers
**Docs** through `workflow_run`. That run uses `main`, which the
`github-pages` environment accepts.

CI also builds the documentation and checks its links on every pull request in
the **Docs / Sphinx** job, so most problems show up before merge.

## API reference

Every public function and class has its own page. Sphinx autosummary
generates the pages into `docs/api/generated/`, which is not committed, from
the lists in `docs/api/containers.md`, `docs/api/tl.md`, and `docs/api/io.md`.
When you add a public function or class, add it to the matching list. The
template for class pages is `docs/_templates/autosummary/class.rst`.

## Release notes

The release notes live in `docs/release-notes/`, one file per version.
`docs/release-notes/index.md` includes them, newest first. See
[CONTRIBUTING.md](CONTRIBUTING.md#release-notes) for adding an entry and
[RELEASE.md](RELEASE.md#before-releasing) for finishing the notes of a release.

## Turning on GitHub Pages

A repository admin sets Pages to deploy from GitHub Actions once. In the
repository, open **Settings → Pages → Build and deployment → Source** and choose
**GitHub Actions**. With the GitHub CLI:

```bash
gh api --method POST repos/NVIDIA/cuspa/pages -f build_type=workflow
```

GitHub manages a `github-pages` environment for the deployments. By default it
accepts deployments only from `main`.

Then publish the site by pushing to `main` or by starting the workflow:

```bash
gh workflow run docs.yml --repo NVIDIA/cuspa --ref main
```

Check that the site is live:

```bash
gh api repos/NVIDIA/cuspa/pages --jq '.status, .html_url'
```

## Previewing locally

Build the documentation as described in [CONTRIBUTING.md](CONTRIBUTING.md), then
serve it:

```bash
[uv] pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs docs/_build/html
python -m http.server --directory docs/_build/html 8000
```

Open http://localhost:8000 in a browser.

To build the whole site, list the release tags to include. This needs
[uv](https://docs.astral.sh/uv/) and the tags in your clone:

```bash
python ci/build_docs_site.py docs/_build/site v0.1.0
python -m http.server --directory docs/_build/site 8000
```

The version switcher reads `switcher.json` from the published site, so a local
preview lists the published versions.

## Troubleshooting

- **The deploy job fails with "Get Pages site failed" or "Not Found".** Pages
  is not turned on, or its source is not GitHub Actions. See
  [Turning on GitHub Pages](#turning-on-github-pages).
- **Environment protection rules reject the deploy job.** The `github-pages`
  environment accepts only the branches listed under **Settings → Environments →
  github-pages**. `main` must be one of them.
- **The build fails on a warning.** Fix the warning shown in the log, and
  reproduce it locally with the commands above.
- **The site still shows old content.** A deployment can take a minute or two
  to appear after the workflow finishes.
- **A release is missing from the version switcher.** Check that the release
  is published and not a draft, and that a release candidate is newer than
  the newest final release. Then start the workflow by hand.
