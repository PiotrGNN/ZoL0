# Contributing to Trading System

We love your input! We want to make contributing to Trading System as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `develop`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Code Style

This project uses several tools to maintain code quality:

- `black` for code formatting
- `flake8` for style guide enforcement
- `mypy` for type checking
- `isort` for import sorting
- `pre-commit` hooks for automated checks

Install development dependencies and set up pre-commit:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Testing

We use pytest for our test suite. Run tests with:

```bash
pytest tests/
```

For coverage report:

```bash
pytest --cov=./ tests/
```

## Documentation

- Use Google-style docstrings
- Update relevant documentation files in `docs/`
- Keep API documentation up to date
- Include docstring examples where appropriate

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with details of any API changes
3. The PR may be merged once you have the sign-off of two other developers
4. All CI checks must pass before merging

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker]

We use GitHub issues to track public bugs. Report a bug by [opening a new issue]().

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.