# Contributing

Thanks for considering a contribution to SMS-MH-AVIE.

## Before opening a pull request

1. Open an issue first for substantial feature changes or architecture changes.
2. Keep changes focused and explain user-visible behavior, assumptions, and limitations.
3. Never commit real student records, roll numbers, credentials, `.env` files, local databases, downloaded raw data, trained model caches, or screenshots containing personal data.
4. Add or update tests for changed behavior. The current integration suite expects the API to be running at `http://localhost:8000`.
5. Run the relevant backend tests and frontend production build; include the actual commands and outcomes in your pull request.
6. Do not describe model output as validated, fair, causal, or suitable for official decisions without evidence and appropriate review.

## Development checks

From the repository root, with backend dependencies installed and the API running in a separate terminal:

```bash
python -m pytest tests/test_suite.py -q
```

In `frontend/`:

```bash
npm ci
npm run build
```

## Pull requests

Include the purpose, summary of changes, tests run, known limitations, and any data/model implications. Do not include secrets or personal student data in issues or pull requests.
