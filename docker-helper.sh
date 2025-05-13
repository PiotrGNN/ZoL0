#!/usr/bin/env bash
# Helper script for common Docker Compose commands

set -e

case $1 in
  build)
    docker-compose build
    ;;
  up)
    docker-compose up
    ;;
  down)
    docker-compose down
    ;;
  logs)
    docker-compose logs -f
    ;;
  shell)
    docker-compose exec trading-backend /bin/bash
    ;;
  lint)
    docker-compose run --rm trading-backend flake8 .
    docker-compose run --rm express-app npm run lint || true
    ;;
  test)
    docker-compose run --rm trading-backend pytest
    docker-compose run --rm express-app npm test || true
    ;;
  clean)
    docker system prune -f
    rm -rf __pycache__ .pytest_cache htmlcov
    ;;
  *)
    echo "Usage: $0 {build|up|down|logs|shell|lint|test|clean}"
    exit 1
    ;;
esac
