"""Terrible orchestrator that ties together bad code and makes it worse."""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Importing everything from bad modules
from src.engine.bad_processor import (
    BadArticleProcessor,
    batch_process_with_terrible_design,
    get_global_processing_status
)
from src.utils.terrible_helper import (
    TerribleDataManager,
    processDataThingWithManyParams,
    mystery_function,
    GLOBAL_CACHE,
    PROCESSING_COUNT,
    DEBUG_MODE
)

# More global state
ORCHESTRATOR_STATE = {
    "active_processors": [],
    "failed_operations": [],
    "last_run_time": None
}

class TerribleOrchestrator:
    """Main orchestrator that coordinates terrible code."""

    # Using mutable class variables
    all_instances = []
    total_operations = 0

    def __init__(self, name="default", settings={}):
        """Initialize with terrible patterns."""
        self.name = name
        self.settings = settings
        self.processors = []
        self.data_managers = []

        # Modifying class state from instance
        TerribleOrchestrator.all_instances.append(self)

        # Initialize with side effects
        self.setup_everything()

    def setup_everything(self):
        """Setup method that does too much."""
        global ORCHESTRATOR_STATE

        # Create processors with different configurations
        for i in range(3):
            processor = BadArticleProcessor({
                "id": i,
                "name": f"processor_{i}",
                "timeout": 30 + i * 10
            })
            self.processors.append(processor)

        # Create data managers
        for i in range(2):
            manager = TerribleDataManager()
            self.data_managers.append(manager)

        # Update global state
        ORCHESTRATOR_STATE["active_processors"].extend(self.processors)
        ORCHESTRATOR_STATE["last_run_time"] = time.time()

    def orchestrate_terrible_workflow(self, input_data, workflow_type="complex"):
        """Main orchestration method with terrible design."""

        TerribleOrchestrator.total_operations += 1

        try:
            if workflow_type == "simple":
                return self.simple_terrible_workflow(input_data)
            elif workflow_type == "complex":
                return self.complex_terrible_workflow(input_data)
            elif workflow_type == "chaotic":
                return self.chaotic_workflow(input_data)
            else:
                # Default to most complex
                return self.complex_terrible_workflow(input_data)

        except Exception as e:
            # Terrible error handling
            ORCHESTRATOR_STATE["failed_operations"].append({
                "error": str(e),
                "timestamp": time.time(),
                "workflow_type": workflow_type,
                "input_preview": str(input_data)[:100]
            })
            return {"error": "Something went wrong", "details": str(e)}

    def simple_terrible_workflow(self, data):
        """Simple workflow that's still terrible."""

        # Use the terrible helper function
        processed_data = processDataThingWithManyParams(
            data, "important", True, True, DEBUG_MODE
        )

        # Random processing by different processors
        processor = self.processors[0]  # Always use first processor
        result = processor.process_article_badly(processed_data)

        # Mystery function call with unclear purpose
        mystery_result = mystery_function(result)

        return {
            "original": data,
            "processed": processed_data,
            "final": result,
            "mystery": mystery_result
        }

    def complex_terrible_workflow(self, data):
        """Complex workflow with many terrible practices."""

        results = {}
        intermediate_data = data

        # Process through multiple stages with side effects
        for stage, processor in enumerate(self.processors):
            stage_name = f"stage_{stage}"

            # Different processing options for each stage
            options = {
                "mode": ["default", "fast", "slow"][stage % 3],
                "validate": stage == 0,  # Only validate in first stage
                "transform": True,
                "save": stage == len(self.processors) - 1,  # Only save in last stage
                "notify": stage % 2 == 0,  # Notify every other stage
                "retry": 3 - stage,  # Decreasing retries
                "timeout": 30 + stage * 10
            }

            try:
                stage_result = processor.process_article_badly(intermediate_data, options)

                if stage_result:
                    results[stage_name] = stage_result
                    intermediate_data = stage_result
                else:
                    # Use data manager as fallback
                    manager = self.data_managers[stage % len(self.data_managers)]
                    fallback_result = manager.do_everything(intermediate_data, "process")
                    results[stage_name] = {"fallback": fallback_result}

                # Random delays between stages
                if stage < len(self.processors) - 1:
                    time.sleep(0.1 * (stage + 1))

            except Exception as e:
                # Continue with corrupted data
                results[stage_name] = {"error": str(e), "failed_data": intermediate_data}

        # Final processing with all data managers
        for i, manager in enumerate(self.data_managers):
            try:
                export_result = manager.do_everything(results, "export")
                results[f"export_{i}"] = export_result
            except:
                # Ignore export failures
                pass

        return results

    def chaotic_workflow(self, data):
        """Completely chaotic workflow."""

        # Use batch processing function
        if not isinstance(data, list):
            data = [data]

        # Random options
        options = {
            "config": {"chaos_mode": True},
            "mode": "chaotic",
            "validate": True,
            "transform": True,
            "save": True,
            "notify": True
        }

        batch_result = batch_process_with_terrible_design(data, options)

        # Get global status
        status = get_global_processing_status()

        # Mystery processing
        mystery_results = []
        for item in data:
            mystery_results.append(mystery_function(item, z=42))

        # Combine everything in a confusing way
        return {
            "batch_result": batch_result,
            "global_status": status,
            "mystery_results": mystery_results,
            "cache_snapshot": dict(GLOBAL_CACHE),  # Copy global cache
            "orchestrator_state": ORCHESTRATOR_STATE,
            "random_number": time.time() * 1000  # Unnecessary randomness
        }

    @classmethod
    def get_all_instances_info(cls):
        """Return info about all instances - breaks encapsulation."""
        return {
            "total_instances": len(cls.all_instances),
            "total_operations": cls.total_operations,
            "instance_names": [inst.name for inst in cls.all_instances],
            "global_state": ORCHESTRATOR_STATE
        }

    @staticmethod
    def nuclear_reset():
        """Reset everything - dangerous static method."""
        # Clear all class state
        TerribleOrchestrator.all_instances.clear()
        TerribleOrchestrator.total_operations = 0

        # Reset other modules
        BadArticleProcessor.reset_all_state()

        # Clear global state
        global ORCHESTRATOR_STATE
        ORCHESTRATOR_STATE = {
            "active_processors": [],
            "failed_operations": [],
            "last_run_time": None
        }

# Module-level functions that break encapsulation

def run_orchestrator_with_terrible_design(data, orchestrator_name="auto"):
    """Module function that creates and runs orchestrator."""

    if orchestrator_name == "auto":
        orchestrator_name = f"auto_{time.time()}"

    # Create orchestrator with side effects
    orchestrator = TerribleOrchestrator(orchestrator_name)

    # Run all workflow types and combine results
    results = {}

    for workflow_type in ["simple", "complex", "chaotic"]:
        try:
            result = orchestrator.orchestrate_terrible_workflow(data, workflow_type)
            results[workflow_type] = result
        except Exception as e:
            results[workflow_type] = {"failed": str(e)}

    # Write results to multiple files (side effects)
    for workflow_type, result in results.items():
        try:
            filename = f"/tmp/terrible_results_{workflow_type}_{time.time()}.json"
            with open(filename, "w") as f:
                json.dump(result, f, default=str)
        except:
            pass  # Ignore file write errors

    return results

def get_system_wide_terrible_status():
    """Get status from across all terrible modules."""
    return {
        "orchestrator": TerribleOrchestrator.get_all_instances_info(),
        "processors": BadArticleProcessor.get_statistics(),
        "global_processing": get_global_processing_status(),
        "global_cache_size": len(GLOBAL_CACHE),
        "processing_count": PROCESSING_COUNT,
        "debug_mode": DEBUG_MODE,
        "orchestrator_state": ORCHESTRATOR_STATE
    }

# Execute code at module import time (bad practice)
print(f"TerribleOrchestrator module loaded at {time.time()}")

# Create a default instance at import time
DEFAULT_ORCHESTRATOR = TerribleOrchestrator("default_import_time")