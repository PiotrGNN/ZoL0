# Makefile for Dockerized Trading System

.PHONY: build up down logs shell lint test clean

build:
	docker-compose build

up:
	docker-compose up

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec backend /bin/bash

lint:
	docker-compose run --rm backend flake8 .
	docker-compose run --rm express-app npm run lint || true

test:
	docker-compose run --rm backend pytest
	docker-compose run --rm express-app npm test || true

clean:
	docker system prune -f
	rm -rf __pycache__ .pytest_cache htmlcov
