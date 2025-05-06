#!/usr/bin/env python3
"""
run_tests.py
-----------
Custom test runner with coverage enforcement and organized test execution.
"""

import os
import sys
import unittest
import coverage
import pytest
import logging
from typing import List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test_runs.log'),
        logging.StreamHandler()
    ]
)

class CoverageConfig:
    """Coverage configuration and requirements."""
    
    CRITICAL_PATHS = [
        'data/risk_management/*',
        'data/execution/*',
        'data/ai_models/*',
        'data/data_processing/*',
    ]
    
    COVERAGE_REQUIREMENTS = {
        'data/risk_management/*': 100,
        'data/execution/*': 100,
        'data/ai_models/*': 95,
        'data/data_processing/*': 95,
        'data/utils/*': 90,
    }
    
    EXCLUDE_PATTERNS = [
        '*/tests/*',
        '*/mock/*',
        '*/__pycache__/*',
    ]

class TestRunner:
    """Custom test runner with coverage enforcement."""

    def __init__(self):
        """Initialize test runner."""
        self.cov = coverage.Coverage(
            branch=True,
            source=['data'],
            omit=CoverageConfig.EXCLUDE_PATTERNS
        )
        self.test_results = {}
        self.coverage_results = {}

    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage tracking."""
        logging.info("Running unit tests...")
        self.cov.start()
        
        loader = unittest.TestLoader()
        start_dir = os.path.join(os.path.dirname(__file__))
        suite = loader.discover(start_dir, pattern="test_*.py")
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        self.cov.stop()
        self.cov.save()
        
        success = result.wasSuccessful()
        self.test_results['unit'] = {
            'total': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped)
        }
        
        return success

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        logging.info("Running integration tests...")
        result = pytest.main([
            '-v',
            '--integration-only',
            'data/tests/integration'
        ])
        return result == 0

    def check_coverage(self) -> Tuple[bool, dict]:
        """Check coverage against requirements."""
        logging.info("Checking coverage requirements...")
        self.cov.load()
        
        # Get coverage data
        total = 0
        covered = 0
        
        for filename in self.cov.get_data().measured_files():
            file_coverage = self.cov.analysis2(filename)
            
            # Match file against critical paths
            for pattern in CoverageConfig.CRITICAL_PATHS:
                if self._match_pattern(filename, pattern):
                    total += len(file_coverage[1])  # All lines
                    covered += len(file_coverage[0])  # Covered lines
                    
                    # Store individual file coverage
                    coverage_pct = (len(file_coverage[0]) / len(file_coverage[1])) * 100
                    self.coverage_results[filename] = coverage_pct
        
        # Check against requirements
        failed_requirements = []
        for pattern, required_coverage in CoverageConfig.COVERAGE_REQUIREMENTS.items():
            pattern_files = [
                f for f in self.coverage_results.keys()
                if self._match_pattern(f, pattern)
            ]
            if pattern_files:
                avg_coverage = sum(
                    self.coverage_results[f] for f in pattern_files
                ) / len(pattern_files)
                if avg_coverage < required_coverage:
                    failed_requirements.append({
                        'pattern': pattern,
                        'required': required_coverage,
                        'actual': avg_coverage
                    })
        
        success = len(failed_requirements) == 0
        return success, {
            'total_coverage': (covered / total * 100) if total > 0 else 0,
            'failed_requirements': failed_requirements,
            'file_coverage': self.coverage_results
        }

    def generate_report(self, coverage_data: dict) -> None:
        """Generate test and coverage report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'test_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=== Test Execution Report ===\n\n")
            
            # Test results
            f.write("Test Results:\n")
            f.write("--------------\n")
            for test_type, results in self.test_results.items():
                f.write(f"\n{test_type.title()} Tests:\n")
                f.write(f"Total: {results['total']}\n")
                f.write(f"Failures: {results['failures']}\n")
                f.write(f"Errors: {results['errors']}\n")
                f.write(f"Skipped: {results['skipped']}\n")
            
            # Coverage results
            f.write("\nCoverage Results:\n")
            f.write("-----------------\n")
            f.write(f"Total Coverage: {coverage_data['total_coverage']:.2f}%\n\n")
            
            if coverage_data['failed_requirements']:
                f.write("Failed Coverage Requirements:\n")
                for failure in coverage_data['failed_requirements']:
                    f.write(
                        f"- {failure['pattern']}: "
                        f"Required {failure['required']}%, "
                        f"Got {failure['actual']:.2f}%\n"
                    )
            
            # Individual file coverage
            f.write("\nIndividual File Coverage:\n")
            for filename, coverage_pct in sorted(
                coverage_data['file_coverage'].items(),
                key=lambda x: x[1]
            ):
                f.write(f"{filename}: {coverage_pct:.2f}%\n")
        
        logging.info(f"Report generated: {report_file}")

    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a glob pattern."""
        from fnmatch import fnmatch
        return fnmatch(filename, pattern)

def main():
    """Main test execution function."""
    runner = TestRunner()
    
    # Run tests
    unit_success = runner.run_unit_tests()
    integration_success = runner.run_integration_tests()
    
    # Check coverage
    coverage_success, coverage_data = runner.check_coverage()
    
    # Generate report
    runner.generate_report(coverage_data)
    
    # Determine overall success
    success = all([
        unit_success,
        integration_success,
        coverage_success
    ])
    
    if not success:
        logging.error("Tests failed or coverage requirements not met")
        sys.exit(1)
    
    logging.info("All tests passed and coverage requirements met")
    sys.exit(0)

if __name__ == '__main__':
    main()