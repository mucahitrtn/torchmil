# Contributing

Contributions (pull requests) are welcome! Here's how to do it:


## Getting started

First fork the library on GitHub. Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/torchmil.git
cd torchmil
pip install .
```

Then install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

These hooks use ruff to format and lint the code, and pyright to typecheck it.

## Making changes

**If you're making changes to the code.** Make your changes and make sure to add tests for them. The tests are located in the `tests` directory. Verify that the tests run correctly by running:

```bash
pip install pytest pandas
pytest tests
```

Then push your changes to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!


**If you're making changes to the documentation.** Make your changes and build the docs to verify that they render correctly:

```bash
pip install '.[docs]'
mkdocs build
mkdocs serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser. Then push your changes to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!