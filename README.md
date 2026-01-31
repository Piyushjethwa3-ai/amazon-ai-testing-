## CI/CD Pipeline

This project includes an automated CI/CD pipeline that:

- Installs dependencies
- Runs automated robustness tests using pytest
- Generates prediction outputs as CSV
- Produces robustness test reports
- Fails builds on test regression

The pipeline runs automatically on every push and pull request.
