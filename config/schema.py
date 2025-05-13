"""Schema validation for configuration files."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import re
from pathlib import Path
import yaml


@dataclass
class ValidationError:
    """Configuration validation error."""

    path: str
    message: str
    value: Any


class SchemaValidator:
    """Validates configuration against schema definitions."""

    def __init__(self, schema_file: Optional[Union[str, Path]] = None):
        self.schema: Dict = {}
        if schema_file:
            self.load_schema(schema_file)

    def load_schema(self, schema_file: Union[str, Path]) -> None:
        """Load schema from file."""
        with open(schema_file) as f:
            self.schema = yaml.safe_load(f)

    def validate(self, config: Dict) -> List[ValidationError]:
        """Validate configuration against loaded schema."""
        errors: List[ValidationError] = []
        if not self.schema:
            return errors

        self._validate_dict(config, self.schema.get("required", {}), "", errors)
        return errors

    def _validate_dict(
        self, config: Dict, schema: Dict, path: str, errors: List[ValidationError]
    ) -> None:
        """Validate dictionary values recursively."""
        for key, value_schema in schema.items():
            new_path = f"{path}.{key}" if path else key

            # Check required fields
            if key not in config:
                errors.append(ValidationError(new_path, "Required field missing", None))
                continue

            value = config[key]
            self._validate_value(value, value_schema, new_path, errors)

    def _validate_value(
        self, value: Any, schema: Dict, path: str, errors: List[ValidationError]
    ) -> None:
        """Validate a single value against its schema."""
        try:
            if schema["type"] == "dict":
                if not isinstance(value, dict):
                    errors.append(ValidationError(path, "Must be a dictionary", value))
                    return
                self._validate_dict(value, schema.get("required", {}), path, errors)

            elif schema["type"] == "list":
                if not isinstance(value, list):
                    errors.append(ValidationError(path, "Must be a list", value))
                    return
                if "schema" in schema:
                    for i, item in enumerate(value):
                        self._validate_value(
                            item, schema["schema"], f"{path}[{i}]", errors
                        )

            elif schema["type"] == "int":
                if not isinstance(value, int):
                    errors.append(ValidationError(path, "Must be an integer", value))
                    return
                self._validate_number(value, schema, path, errors)

            elif schema["type"] == "float":
                if not isinstance(value, (int, float)):
                    errors.append(ValidationError(path, "Must be a number", value))
                    return
                self._validate_number(value, schema, path, errors)

            elif schema["type"] == "bool":
                if not isinstance(value, bool):
                    errors.append(ValidationError(path, "Must be a boolean", value))

            elif schema["type"] == "str":
                if not isinstance(value, str):
                    errors.append(ValidationError(path, "Must be a string", value))
                    return
                self._validate_string(value, schema, path, errors)

            # Additional type checks can be added here

        except Exception as e:
            errors.append(ValidationError(path, f"Validation error: {str(e)}", value))

    def _validate_number(
        self,
        value: Union[int, float],
        schema: Dict,
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate numeric constraints."""
        if "min" in schema and value < schema["min"]:
            errors.append(
                ValidationError(
                    path, f"Must be greater than or equal to {schema['min']}", value
                )
            )
        if "max" in schema and value > schema["max"]:
            errors.append(
                ValidationError(
                    path, f"Must be less than or equal to {schema['max']}", value
                )
            )

    def _validate_string(
        self, value: str, schema: Dict, path: str, errors: List[ValidationError]
    ) -> None:
        """Validate string constraints."""
        if "allowed" in schema and value not in schema["allowed"]:
            errors.append(
                ValidationError(
                    path, f"Must be one of: {', '.join(schema['allowed'])}", value
                )
            )
        if "pattern" in schema and not re.match(schema["pattern"], value):
            errors.append(
                ValidationError(path, f"Must match pattern: {schema['pattern']}", value)
            )
        if "min_length" in schema and len(value) < schema["min_length"]:
            errors.append(
                ValidationError(
                    path,
                    f"Must be at least {schema['min_length']} characters long",
                    value,
                )
            )
        if "max_length" in schema and len(value) > schema["max_length"]:
            errors.append(
                ValidationError(
                    path,
                    f"Must be at most {schema['max_length']} characters long",
                    value,
                )
            )


def validate_config(
    config: Dict, schema_file: Union[str, Path]
) -> List[ValidationError]:
    """Helper function to validate configuration using schema file."""
    validator = SchemaValidator(schema_file)
    return validator.validate(config)
