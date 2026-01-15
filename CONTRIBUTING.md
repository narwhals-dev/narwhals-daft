# Contributing

Clone this repository with the `--recursive` flag.

```console
git clone git@github.com:narwhals-dev/narwhals-daft.git narwhals-daft-dev --recursive 
cd narwhals-daft-dev
```

Make a virtual environment and activate it.

```console
uv venv -p 3.12
. .venv/bin/activate
```

Install `pre-commit`.
```console
uv pip install pre-commit
pre-commit install
```

```console
uv pip install -e . --group tests
```

and also fetch and install Narwhals as a git submodule:

```console
git submodule update --init --recursive
uv pip install -e narwhals
```
Keep daft to the latest version:

```console
uv pip install daft -U
```

To run the tests:

```console
python run_tests.py
```

Any additional arguments you pass will be passed down to pytest, e.g.

```console
python run_tests.py -x
```

To run type-checking:

```console
pyright narwhals_daft
```

If your changes add new functionalities, update the tests:

```console
python update_run_tests.py
```

## Updating the Narwhals submodule

Run

```console
. utils/submodule_update.sh
```

Check that tests still run, update `run_tests.py` if necessary, and open a PR.
