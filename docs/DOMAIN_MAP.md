# Dependency DAG & Domain Map

## Dependency DAG (auto-generated placeholder)

- core: execution, risk, strategy
- adapters: exchanges, data
- infra: kafka, redis, db
- ops: observability, security
- cli: entrypoints

## Integration Coverage

- Risk management: tested (unit/integration)
- Strategy: tested (unit/integration)
- Exchange connectors: tested (integration)
- Order execution: tested (integration)
- Portfolio: tested (integration)
- AI models: tested (integration)
- Observability: config present (Grafana, Prometheus)
- Security: config present (OPA, JWT, Vault)

## Next Steps

- Refactor code into hexagonal structure (src/core, src/adapters, etc.)
- Ensure all modules are imported via adapters, not direct imports
- Add missing tests for IPC (Kafka/Redis), circuit-breaker, audit sink
- Add OpenTelemetry/Prometheus exporters to ops/
- Add OPA/Vault configs to ops/security/
- Add CI/CD pipeline and SBOM generation

---

*This file is updated as the repo is refactored and new domains are mapped.*
