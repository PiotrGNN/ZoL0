# my-express-app

This is a TypeScript Express application.

## Project Structure

```text
my-express-app
├── src
│   ├── app.ts          # Entry point of the application
│   ├── controllers     # Contains controllers for handling requests
│   │   └── index.ts    # Exports IndexController
│   ├── routes          # Contains route definitions
│   │   └── index.ts    # Exports setRoutes function
│   └── types           # Contains TypeScript type definitions
│       └── index.ts    # Exports Request and Response interfaces
├── package.json        # npm configuration file
├── tsconfig.json       # TypeScript configuration file
└── README.md           # Project documentation
```

## Installation

To install the dependencies, run:

```bash
npm install
```

## Usage

To start the application, run:

```bash
npm start
```

## Docker Compose (Python + Express)

To build and run all services (Python, Express, Redis, Kafka, etc.):

```powershell
docker compose up --build
```

- Python API: <http://localhost:5000>
- Express (TypeScript): <http://localhost:3000>

## Docker Deployment

### Build and Run All Services

```powershell
# Build all images
$ docker-compose build

# Start all services
$ docker-compose up
```

- Python API: <http://localhost:5000>
- Express (TypeScript): <http://localhost:3000>
- Grafana: <http://localhost:3000> (if enabled)
- Prometheus: <http://localhost:9090>
- Redis: localhost:6379
- Postgres: localhost:5432

### Environment Variables

Copy `.env.example` to `.env` and fill in secrets before running Docker Compose.

### Health Check

You can check if the backend is running:

```powershell
curl http://localhost:5000/health
```

### Troubleshooting

- Ensure Docker Desktop is running.
- If you see permission or path errors, try running as administrator.
- For logs: `docker-compose logs -f <service>`

---

For more details, see the monorepo documentation.

## Python Security & Fault-Tolerance (ZoL0)

This monorepo includes a Tier-0, fault-immune, ML-augmented crypto-trading platform in `ZoL0/`.

### Key Features

- Hardened `SecurityManager` (password, token, RBAC, audit-trail, rate limiting)
- Fault-immune `ComponentManager` (circuit-breaker, retry, audit-trail, async)
- 98%+ test coverage, property/fuzz/chaos tests
- Structured logging and observability for all critical flows

### Running Tests & Coverage

```powershell
# From repo root
pytest --maxfail=3 --disable-warnings --tb=short --cov=ZoL0/data/utils --cov-report=term ZoL0/data/tests/
```

### CI/CD & SBOM

- All tests and coverage are enforced in CI (see `.github/workflows/` if present)
- To generate a Software Bill of Materials (SBOM):

```powershell
pip install cyclonedx-bom
cyclonedx-py -o sbom.xml
```

### Coverage & Status

![Coverage](htmlcov/index.html)

All critical security and fault-tolerance logic is tested and observable.

## License

This project is licensed under the MIT License.

---

## Docker: Build and Run All Services

```powershell
# Build all images
docker-compose build

# Start all services
docker-compose up
```

- Python API: <http://localhost:5000>
- Express (TypeScript): <http://localhost:3000>
- Grafana: <http://localhost:3000>
- Prometheus: <http://localhost:9090>
- Postgres: localhost:5432
- Redis: localhost:6379

### Docker Environment Variables

Copy `.env.example` to `.env` and adjust as needed.

### Stopping Docker Services

```powershell
docker-compose down
```

### Docker Troubleshooting

- Check logs: `docker-compose logs <service>`
- Rebuild after code changes: `docker-compose build`
- Ensure ports 5000, 3000, 9090, 5432, 6379 are free.

---

For advanced usage, see the comments in `docker-compose.yml` and the Dockerfiles.
