"""
Component management system with better initialization control and state management.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import system logger
from data.logging.system_logger import get_logger
logger = get_logger()

class ComponentState(Enum):
    """Component states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"

class Component:
    """Base component class."""
    
    def __init__(self, name: str, init_func: Callable, cleanup_func: Optional[Callable] = None):
        self.name = name
        self.init_func = init_func
        self.cleanup_func = cleanup_func
        self.state = ComponentState.UNINITIALIZED
        self.error = None
        self.last_state_change = datetime.now()
        self.retry_count = 0
        self.max_retries = 3
        self.dependencies: List[str] = []
        self._lock = threading.Lock()
        self.health_check_interval = 60  # seconds
        self._stop_health_check = threading.Event()
        self._health_check_thread = None

    def initialize(self) -> bool:
        """Initialize the component."""
        with self._lock:
            if self.state == ComponentState.RUNNING:
                return True

            try:
                self.state = ComponentState.INITIALIZING
                self.init_func()
                self.state = ComponentState.RUNNING
                self.error = None
                self.last_state_change = datetime.now()
                self._start_health_check()
                logger.log_info(f"Component {self.name} initialized successfully")
                return True
            except Exception as e:
                self.error = str(e)
                self.state = ComponentState.ERROR
                self.last_state_change = datetime.now()
                logger.log_error(f"Failed to initialize component {self.name}: {e}")
                return False

    def cleanup(self) -> bool:
        """Clean up the component."""
        with self._lock:
            try:
                if self.cleanup_func:
                    self.cleanup_func()
                self.state = ComponentState.STOPPED
                self.last_state_change = datetime.now()
                self._stop_health_check.set()
                if self._health_check_thread:
                    self._health_check_thread.join()
                logger.log_info(f"Component {self.name} cleaned up successfully")
                return True
            except Exception as e:
                self.error = str(e)
                self.state = ComponentState.ERROR
                self.last_state_change = datetime.now()
                logger.log_error(f"Failed to cleanup component {self.name}: {e}")
                return False

    def _health_check(self):
        """Perform health check."""
        while not self._stop_health_check.is_set():
            try:
                # Basic health check - override in subclasses for specific checks
                if self.state == ComponentState.RUNNING:
                    # Verify component is responsive
                    pass
            except Exception as e:
                with self._lock:
                    self.state = ComponentState.DEGRADED
                    self.error = str(e)
                    self.last_state_change = datetime.now()
                    logger.log_warning(f"Health check failed for component {self.name}: {e}")

            self._stop_health_check.wait(self.health_check_interval)

    def _start_health_check(self):
        """Start health check thread."""
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check,
            daemon=True
        )
        self._health_check_thread.start()

    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'error': self.error,
                'last_state_change': self.last_state_change.isoformat(),
                'retry_count': self.retry_count,
                'dependencies': self.dependencies
            }

class ComponentManager:
    """Component management system."""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self.initialization_order: List[str] = []

    def register_component(self, 
                         name: str, 
                         init_func: Callable,
                         cleanup_func: Optional[Callable] = None,
                         dependencies: List[str] = None) -> None:
        """Register a new component."""
        with self._lock:
            component = Component(name, init_func, cleanup_func)
            if dependencies:
                component.dependencies = dependencies
                # Verify dependencies exist
                for dep in dependencies:
                    if dep not in self.components:
                        logger.log_warning(f"Dependency {dep} for component {name} not registered")
            self.components[name] = component
            self._update_initialization_order()

    def _update_initialization_order(self) -> None:
        """Update component initialization order based on dependencies."""
        visited = set()
        temp = set()
        order = []

        def visit(name: str):
            if name in temp:
                raise ValueError(f"Circular dependency detected for {name}")
            if name in visited:
                return
            temp.add(name)
            component = self.components[name]
            for dep in component.dependencies:
                if dep in self.components:
                    visit(dep)
            temp.remove(name)
            visited.add(name)
            order.append(name)

        try:
            for name in self.components:
                if name not in visited:
                    visit(name)
            self.initialization_order = order
        except ValueError as e:
            logger.log_error(f"Dependency resolution error: {e}")
            # Fall back to random order
            self.initialization_order = list(self.components.keys())

    async def initialize_all(self) -> bool:
        """Initialize all components in correct order."""
        try:
            for name in self.initialization_order:
                component = self.components[name]
                # Wait for dependencies
                for dep in component.dependencies:
                    dep_component = self.components.get(dep)
                    if not dep_component or dep_component.state != ComponentState.RUNNING:
                        logger.log_error(f"Cannot initialize {name}: dependency {dep} not ready")
                        return False
                
                if not await asyncio.get_event_loop().run_in_executor(
                    self._executor, component.initialize
                ):
                    return False
            return True
        except Exception as e:
            logger.log_error(f"Error during component initialization: {e}")
            return False

    async def cleanup_all(self) -> bool:
        """Clean up all components in reverse order."""
        try:
            for name in reversed(self.initialization_order):
                component = self.components[name]
                if not await asyncio.get_event_loop().run_in_executor(
                    self._executor, component.cleanup
                ):
                    return False
            return True
        except Exception as e:
            logger.log_error(f"Error during component cleanup: {e}")
            return False

    def get_component_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific component."""
        component = self.components.get(name)
        if component:
            return component.get_status()
        return None

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            'components': {
                name: component.get_status()
                for name, component in self.components.items()
            },
            'initialization_order': self.initialization_order,
            'timestamp': datetime.now().isoformat()
        }

    def retry_failed_components(self) -> bool:
        """Retry initialization of failed components."""
        success = True
        with self._lock:
            for component in self.components.values():
                if component.state in (ComponentState.ERROR, ComponentState.DEGRADED):
                    if component.retry_count < component.max_retries:
                        component.retry_count += 1
                        if not component.initialize():
                            success = False
                    else:
                        logger.log_warning(
                            f"Component {component.name} exceeded max retries ({component.max_retries})"
                        )
                        success = False
        return success

# Create global component manager instance
component_manager = ComponentManager()

def get_component_manager() -> ComponentManager:
    """Get global component manager instance."""
    return component_manager

# Example usage
if __name__ == "__main__":
    async def test():
        cm = get_component_manager()
        
        # Register some test components
        def init_db():
            print("Initializing database...")
            time.sleep(1)
        
        def cleanup_db():
            print("Cleaning up database...")
        
        def init_api():
            print("Initializing API...")
            time.sleep(1)
        
        def init_models():
            print("Initializing models...")
            time.sleep(1)
            
        # Register components with dependencies
        cm.register_component("database", init_db, cleanup_db)
        cm.register_component("api", init_api, dependencies=["database"])
        cm.register_component("models", init_models, dependencies=["database"])
        
        # Initialize all components
        if await cm.initialize_all():
            print("All components initialized successfully")
            print("\nComponent status:")
            print(cm.get_all_status())
            
            # Clean up
            await cm.cleanup_all()
        else:
            print("Failed to initialize components")

    # Run the test
    asyncio.run(test())