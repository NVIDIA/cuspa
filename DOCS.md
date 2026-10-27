# Documentation site

The cuspa documentation is published with GitHub Pages at
https://nvidia.github.io/cuspa/. Sphinx builds it from `docs/`, and the site
shows the documentation on `main`.

## How the site is published

The **Docs** workflow (`.github/workflows/docs.yml`) runs on every push to
`main` and can also be started by hand. It builds the HTML with Sphinx on a
GitHub-hosted runner and deploys it with GitHub's Pages actions. Sphinx runs
with `-W`, so a warning fails the build and the published site stays as it was.

CI also builds the documentation and checks its links on every pull request in
the **Docs / Sphinx** job, so most problems show up before merge.

## API reference

Every public function and class has its own page. Sphinx autosummary
generates the pages into `docs/api/generated/`, which is not committed, from
the lists in `docs/api/containers.md`, `docs/api/tl.md`, and `docs/api/io.md`.
When you add a public function or class, add it to the matching list. The
template for class pages is `docs/_templates/autosummary/class.rst`.

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
