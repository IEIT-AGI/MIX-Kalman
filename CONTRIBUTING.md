# Contributing to MIX-Kalman

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. fork and pull the latest MIX-Kalman repository
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

Note: If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.

## Code style

### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style. And we use [pre-commmit hook](https://pre-commit.com/)  to identify simple issues before submission to code review.

The pre-commit hooks for linting and formatting are as follows:

- [flake8](https://gitlab.com/pycqa/flake8.git): a package three tools: PyFlakes, Pep8, McCabe. It can check the not standard place of code.
- [yapf](https://github.com/pre-commit/mirrors-yapf): check and format Python files.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.
- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks): other check hooks like `trailing-whitespace`, `check-yaml`, `end-of-file-fixer` and so on.

The config for a pre-commit hook is stored in [.pre-commit-config](.pre-commit-config.yaml).

Style configurations of flake8, yapf and isort can be found in [.flake8](.flake8), [.style.yapf](.style.yapf), [.isort.cfg](.isort.cfg)

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip install pre-commit  # install  pre-commit package manager
pre-commit  # initialize and install the pre-commit hooks
```

From the repository folder

```shell
pre-commit install  # set up the git hook scripts
```

After this on every commit check code linters and formatter will be enforced.

>Before you create a PR, make sure that your code lints and is formatted by yapf.
