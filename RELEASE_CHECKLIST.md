# Public release checklist

## Owner decisions — required before public release

- [ ] Choose a repository name and approve the short description/topics.
- [ ] Select a project-code licence and add the corresponding `LICENSE`. No licence has been selected in this release-preparation change set; do not assume that public visibility grants reuse rights.
- [ ] Review UCI dataset terms and attribution on the official dataset page; retain a link and date of review in the release notes.
- [ ] Decide whether to distribute trained model artifacts. The current recommendation is to regenerate/cache them locally rather than commit `.joblib` files.
- [ ] Confirm the repository contains no real student data, identifiers, credentials, local databases, or private timetable uploads.
- [ ] Decide whether the project is a demo/research release only (recommended) and approve the limitations language.

## Technical checks

- [ ] Run backend tests against a freshly started API and record the exact result.
- [ ] Run `npm ci` and `npm run build` in `frontend/`.
- [ ] Run the split evaluation command and inspect `reports/` output; interpret metrics cautiously because target construction overlaps with input features.
- [ ] Perform a manual UI smoke test: load dashboard, submit one demo prediction, exercise attendance calculations, upload only a synthetic timetable, and inspect browser/network errors.
- [ ] Restrict CORS, add authentication/authorization, and define data retention before any real-data or internet-facing deployment.
- [ ] Review the complete diff, `.gitignore`, dependency lockfiles, large files, and secret scan.
- [ ] Confirm repository visibility and default branch; create a version tag and release notes only after owner approval.
- [ ] Push to GitHub only after the owner has reviewed the diff and explicitly approved publication.
