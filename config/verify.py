"""Configuration verification and validation utilities."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .manager import get_config, ConfigValidationError

logger = logging.getLogger(__name__)


def verify_directories() -> List[str]:
    """Verify required directories exist and are writable."""
    errors = []
    required_dirs = [
        "data",
        "data/cache",
        "logs",
        "saved_models",
        "config",
        "monitoring/dashboards",
        "monitoring/provisioning",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write permissions by creating a temp file
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"Directory {dir_path} error: {str(e)}")

    return errors


def verify_api_credentials() -> List[str]:
    """Verify API credentials are properly configured."""
    errors = []
    config = get_config()

    # Required API credentials
    if not os.getenv("BYBIT_API_KEY"):
        errors.append("Bybit API key not configured")
    if not os.getenv("BYBIT_API_SECRET"):
        errors.append("Bybit API secret not configured")

    # Check testnet settings consistency
    testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    if not testnet and config.get("environment.mode") != "production":
        errors.append("Production API can only be used in production mode")

    return errors


def verify_security_settings() -> List[str]:
    errors = []
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret or jwt_secret == "generate_a_secure_secret_key_here":
        errors.append("JWT secret not configured")

    # Check password salt
    password_salt = os.getenv("PASSWORD_SALT")
    if not password_salt or password_salt == "generate_a_secure_salt_here":
        errors.append("Password salt not configured")
    return errors


def verify_trading_settings() -> List[str]:
    """Verify trading-related configuration."""
    errors = []
    config = get_config()

    trading = config.get("trading", {})

    # Verify risk parameters
    if trading.get("max_risk", 0) > 0.1:
        errors.append("Maximum risk per trade cannot exceed 10%")
    if trading.get("max_position_size", 0) > 0.5:
        errors.append("Maximum position size cannot exceed 50% of portfolio")
    if trading.get("max_drawdown", 0) > 0.2:
        errors.append("Maximum drawdown cannot exceed 20%")

    # Verify strategy configuration
    strategies = config.get("strategies", {})
    if not any(s.get("enabled", False) for s in strategies.values()):
        errors.append("At least one trading strategy must be enabled")

    return errors


def verify_database_settings() -> List[str]:
    """Verify database configuration."""
    errors = []
    config = get_config()

    db_path = config.get("database.path")
    if not db_path:
        errors.append("Database path not configured")
    else:
        # Convert relative path to absolute
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)

        # Check directory exists and is writable
        db_dir = os.path.dirname(db_path)
        try:
            os.makedirs(db_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(db_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.unlink(test_file)
        except Exception as e:
            errors.append(f"Database directory error: {str(e)}")

    return errors


def verify_monitoring_settings() -> List[str]:
    """Verify monitoring configuration."""
    errors = []
    config = get_config()

    monitoring = config.get("monitoring", {})
    if monitoring.get("enabled"):
        # Verify ports don't conflict
        used_ports = set()
        for port_setting in ["metrics_port", "grafana_port"]:
            port = monitoring.get(port_setting)
            if port in used_ports:
                errors.append(f"Port conflict detected: {port} used multiple times")
            used_ports.add(port)

        # Verify Grafana admin credentials
        grafana_user = os.getenv("GRAFANA_ADMIN_USER")
        grafana_pass = os.getenv("GRAFANA_ADMIN_PASSWORD")
        if not grafana_user or grafana_user == "admin":
            errors.append("Default Grafana admin username should be changed")
        if not grafana_pass or grafana_pass == "admin":
            errors.append("Default Grafana admin password should be changed")

    return errors


def verify_all() -> Dict[str, List[str]]:
    """Run all verification checks and return any errors."""
    verifications = {
        "directories": verify_directories,
        "api_credentials": verify_api_credentials,
        "security": verify_security_settings,
        "trading": verify_trading_settings,
        "database": verify_database_settings,
        "monitoring": verify_monitoring_settings,
    }

    results = {}
    for name, verify_func in verifications.items():
        errors = verify_func()
        if errors:
            results[name] = errors
            for error in errors:
                logger.warning(f"{name} verification error: {error}")

    return results


def verify_config_file(config_path: str) -> List[str]:
    """Verify a specific configuration file."""
    errors = []

    try:
        config = get_config()
        config_instance = config.load_file(config_path)
        schema_path = os.path.join(os.path.dirname(__file__), "schema.yaml")

        validation_errors = config.validate_config_file(config_instance, schema_path)
        errors.extend(str(err) for err in validation_errors)

    except Exception as e:
        errors.append(f"Configuration file error: {str(e)}")

    return errors


def check_configuration(exit_on_error: bool = True) -> bool:
    """Verify entire configuration system.

    Args:
        exit_on_error: Exit program if verification fails

    Returns:
        True if verification passed
    """
    logger.info("Verifying configuration...")

    # Run all verifications
    verification_results = verify_all()

    if verification_results:
        logger.error("Configuration verification failed:")
        for category, errors in verification_results.items():
            for error in errors:
                logger.error(f"{category}: {error}")

        if exit_on_error:
            logger.critical("Exiting due to configuration errors")
            sys.exit(1)
        return False

    logger.info("Configuration verification passed")
    return True
