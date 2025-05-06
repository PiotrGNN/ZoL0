"""
Enhanced test framework with better coverage and organization.
"""

import unittest
import asyncio
import logging
import json
import os
import sys
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import coverage
import pytest
from unittest.mock import MagicMock, patch

# Import system logger
from data.logging.system_logger import get_logger
logger = get_logger()

class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, success: bool, duration: float, error: Optional[str] = None):
        self.name = name
        self.success = success
        self.duration = duration
        self.error = error
        self.timestamp = datetime.now()

class TestSuite:
    """Test suite with enhanced features."""
    
    def __init__(self, name: str):
        self.name = name
        self.tests: Dict[str, Callable] = {}
        self.setup_functions: List[Callable] = []
        self.teardown_functions: List[Callable] = []
        self.results: List[TestResult] = []
        self.coverage = coverage.Coverage()

    def add_test(self, name: str, test_func: Callable) -> None:
        """Add test to suite."""
        self.tests[name] = test_func

    def add_setup(self, setup_func: Callable) -> None:
        """Add setup function."""
        self.setup_functions.append(setup_func)

    def add_teardown(self, teardown_func: Callable) -> None:
        """Add teardown function."""
        self.teardown_functions.append(teardown_func)

    async def run(self) -> Dict[str, Any]:
        """Run all tests in suite."""
        start_time = datetime.now()
        
        try:
            # Start coverage
            self.coverage.start()
            
            # Run setup functions
            for setup in self.setup_functions:
                await self._run_async_or_sync(setup)
            
            # Run tests
            for name, test in self.tests.items():
                try:
                    test_start = datetime.now()
                    await self._run_async_or_sync(test)
                    duration = (datetime.now() - test_start).total_seconds()
                    self.results.append(TestResult(name, True, duration))
                except Exception as e:
                    duration = (datetime.now() - test_start).total_seconds()
                    self.results.append(TestResult(name, False, duration, str(e)))
                    logger.log_error(f"Test {name} failed: {e}")
            
            # Run teardown functions
            for teardown in self.teardown_functions:
                await self._run_async_or_sync(teardown)
            
            # Stop coverage
            self.coverage.stop()
            self.coverage.save()
            
            return self._generate_report(start_time)
        except Exception as e:
            logger.log_error(f"Error running test suite {self.name}: {e}")
            return {
                'suite': self.name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _run_async_or_sync(self, func: Callable) -> Any:
        """Run function whether it's async or sync."""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            return await asyncio.get_event_loop().run_in_executor(None, func)

    def _generate_report(self, start_time: datetime) -> Dict[str, Any]:
        """Generate test report."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Get coverage report
        coverage_data = self.coverage.get_data()
        total_lines = 0
        covered_lines = 0
        
        for filename in coverage_data.measured_files():
            _, statements, missing, _ = self.coverage.analysis(filename)
            total_lines += len(statements)
            covered_lines += len(statements) - len(missing)
        
        coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            'suite': self.name,
            'timestamp': end_time.isoformat(),
            'duration': duration,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'coverage': coverage_percentage,
            'results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'duration': r.duration,
                    'error': r.error,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }

class TestFramework:
    """Enhanced test framework."""
    
    def __init__(self):
        self.suites: Dict[str, TestSuite] = {}
        self.mocks: Dict[str, MagicMock] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)

    def create_suite(self, name: str) -> TestSuite:
        """Create new test suite."""
        suite = TestSuite(name)
        self.suites[name] = suite
        return suite

    def create_mock(self, name: str) -> MagicMock:
        """Create and store mock object."""
        mock = MagicMock()
        self.mocks[name] = mock
        return mock

    async def run_all_suites(self) -> Dict[str, Any]:
        """Run all test suites."""
        start_time = datetime.now()
        results = []
        
        try:
            for suite in self.suites.values():
                results.append(await suite.run())
            
            return self._generate_summary(results, start_time)
        except Exception as e:
            logger.log_error(f"Error running test suites: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def run_suite(self, name: str) -> Optional[Dict[str, Any]]:
        """Run specific test suite."""
        suite = self.suites.get(name)
        if suite:
            return await suite.run()
        return None

    def _generate_summary(self, results: List[Dict[str, Any]], start_time: datetime) -> Dict[str, Any]:
        """Generate summary of all test results."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        total_tests = sum(r['total_tests'] for r in results)
        passed_tests = sum(r['passed_tests'] for r in results)
        failed_tests = sum(r['failed_tests'] for r in results)
        
        # Calculate average coverage
        total_coverage = sum(r['coverage'] for r in results)
        avg_coverage = total_coverage / len(results) if results else 0
        
        return {
            'timestamp': end_time.isoformat(),
            'duration': duration,
            'total_suites': len(results),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'average_coverage': avg_coverage,
            'suite_results': results
        }

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save test results to file."""
        try:
            os.makedirs('test_results', exist_ok=True)
            filepath = os.path.join('test_results', filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.log_info(f"Test results saved to {filepath}")
        except Exception as e:
            logger.log_error(f"Error saving test results: {e}")

# Create global framework instance
test_framework = TestFramework()

def get_test_framework() -> TestFramework:
    """Get global test framework instance."""
    return test_framework

# Example usage
if __name__ == "__main__":
    async def run_tests():
        framework = get_test_framework()
        
        # Create test suite
        suite = framework.create_suite("example_suite")
        
        # Add some test functions
        def test_success():
            assert 1 + 1 == 2
        
        def test_failure():
            assert 1 + 1 == 3
        
        suite.add_test("test_success", test_success)
        suite.add_test("test_failure", test_failure)
        
        # Run tests
        results = await framework.run_all_suites()
        
        # Save results
        framework.save_results(results, "test_results.json")
        
        print("\nTest Results:")
        print(json.dumps(results, indent=2))

    # Run the example
    asyncio.run(run_tests())