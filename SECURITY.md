# Security and privacy

This repository is an educational prototype, not a security-audited student information system.

## Reporting a vulnerability

Please do not publish credentials, personal student records, or exploitable details in a public issue. Contact the repository owner privately through a verified contact method listed on the owner's GitHub profile. No response-time SLA is promised by this repository.

## Current limitations to consider

- The API configures permissive CORS (`allow_origins=["*"]`). Restrict origins before any network-facing deployment.
- The roster endpoint stores records in a local JSON file and the application has no verified authentication/authorization layer. Do not expose it to the public internet with real student data.
- Uploaded timetable files and OCR text may contain personal information. Review retention and access controls before using non-demo data.
- Runtime datasets and model artifacts are downloaded/generated and cached locally. Only load serialized model artifacts from trusted sources; joblib/pickle-based formats can execute code during deserialization.
- Predictions and attendance calculations are not official grades, admission decisions, exam eligibility rulings, or disciplinary recommendations.

Please use synthetic or properly de-identified demo data while evaluating the project.
