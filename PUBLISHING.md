# Getting this on GitHub and demoing it

## A. Push it (pick one route)

### Route 1 — browser only, no git

1. Download `mhw-phyto.zip` from the chat and unzip it.
2. github.com -> **+** -> **New repository**. Name it, leave it **empty**
   (no README, no .gitignore — you already have both).
3. On the empty repo page: **uploading an existing file**.
4. Drag in the *contents* of the unzipped folder, not the folder itself.
   Drag `src`, `scripts`, `tests`, `notebooks`, `vendor`, `outputs`,
   `.github`, and the loose files together in one go.
5. Commit.

The catch: GitHub's web uploader silently skips dotfiles in some browsers.
Afterwards, confirm `.github/workflows/pipeline.yml` exists. If it doesn't,
create it via **Add file -> Create new file**, type
`.github/workflows/pipeline.yml` as the name, and paste the contents in.
Without it you get no CI and no green badge.

### Route 2 — command line

```bash
cd mhw-phyto
git init -b main
git add .
git commit -m "MHW/phytoplankton pipeline: detection validated against reference"
git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
git push -u origin main
```

If pushing over HTTPS, GitHub wants a personal access token as the password,
not your account password: Settings -> Developer settings -> Personal access
tokens -> Fine-grained -> repo scope. Or install the `gh` CLI and run
`gh auth login`, then `gh repo create --source=. --public --push`.

## B. Wire up the placeholders

Three files contain `YOUR-USER/YOUR-REPO`. Replace them or the badges 404:

```bash
grep -rl "YOUR-USER/YOUR-REPO" . | xargs sed -i 's|YOUR-USER/YOUR-REPO|actual-user/actual-repo|g'
```

On macOS use `sed -i ''` instead of `sed -i`.

## C. What a visitor sees

- **Landing page**: badges, then the pipeline figure, then the results table.
  The figure renders because `outputs/walking_skeleton.png` is committed and
  referenced with a relative path. This is why `outputs/` is deliberately
  *not* in `.gitignore`.
- **Actions tab**: every push runs the tests, the reference cross-validation,
  and the full pipeline. The run page prints `summary.json` and attaches the
  figures as a downloadable artifact.
- **Green tick** next to the latest commit.

## D. Demoing it

**For a judge with a laptop and no patience:** the Colab badge. It clones the
repo, installs two packages, and runs everything in the browser in about two
minutes. Nothing to install. Run it once yourself first so the placeholders
are fixed and the outputs are cached.

**For a judge reading over your shoulder:** the README landing page. Figure
above the fold, results table below it.

**Live at the podium:** `python scripts/run_walking_skeleton.py` takes well
under a minute and prints the acceptance test result. Rehearse it. Have the
Colab tab already open as a fallback in case the venue wifi dies.

**Do not** demo by opening `.py` files in the GitHub file browser. Nobody has
ever been convinced by scrolling source code.

## E. Order to show things in

1. The question: does a heatwave help or harm phytoplankton? Both, depending
   where — and that is the interesting part.
2. Validation: our detector matches the canonical Hobday implementation at
   99.7%, and our response analysis recovers a planted signal while correctly
   finding nothing in the control region.
3. The result: the regime map.
4. The forecast, *with* its baselines visible.
5. The caveats, before anyone asks: surface-only chlorophyll, cloud gaps,
   photoacclimation.

Leading with validation rather than results is unusual and it works. It tells
a judge that the numbers that follow can be trusted.
