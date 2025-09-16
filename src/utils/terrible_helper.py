"""Intentionally terrible helper functions for testing PR review system."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Union

# Global variables (bad practice)
GLOBAL_CACHE = {}
PROCESSING_COUNT = 0
DEBUG_MODE = True

def processDataThingWithManyParams(data, type_of_thing, should_process=True, use_cache=True, debug=False, retry_count=3, timeout=30, callback=None, metadata=None, options={}):
    """Process some data with way too many parameters and terrible naming."""
    global GLOBAL_CACHE, PROCESSING_COUNT, DEBUG_MODE

    # Terrible nested logic
    if should_process:
        if type_of_thing == "important":
            if use_cache:
                if str(data) in GLOBAL_CACHE:
                    if debug or DEBUG_MODE:
                        print(f"Found in cache: {data}")
                        if callback:
                            if hasattr(callback, '__call__'):
                                try:
                                    result = callback(GLOBAL_CACHE[str(data)])
                                    if result:
                                        return result
                                except Exception as e:
                                    if retry_count > 0:
                                        time.sleep(random.uniform(0.1, 0.5))
                                        return processDataThingWithManyParams(data, type_of_thing, should_process, use_cache, debug, retry_count-1, timeout, callback, metadata, options)
                                    else:
                                        raise e
                    return GLOBAL_CACHE[str(data)]

            # More terrible logic
            start_time = time.time()
            PROCESSING_COUNT += 1

            try:
                # Deeply nested processing logic
                if isinstance(data, dict):
                    result = {}
                    for k, v in data.items():
                        if isinstance(v, str):
                            if len(v) > 10:
                                if v.startswith("process_"):
                                    result[k] = v.upper()
                                elif v.endswith("_data"):
                                    result[k] = v.lower()
                                else:
                                    result[k] = v.title()
                            else:
                                result[k] = v
                        elif isinstance(v, (int, float)):
                            if v > 100:
                                result[k] = v * 2
                            elif v > 50:
                                result[k] = v * 1.5
                            else:
                                result[k] = v
                        else:
                            result[k] = str(v)
                elif isinstance(data, list):
                    result = []
                    for item in data:
                        if isinstance(item, str):
                            result.append(item.replace(" ", "_"))
                        else:
                            result.append(str(item))
                else:
                    result = str(data).replace(" ", "_").upper()

                # Cache the result
                if use_cache:
                    GLOBAL_CACHE[str(data)] = result

                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError("Processing took too long")

                return result

            except Exception as e:
                if debug:
                    print(f"Error processing {data}: {e}")
                if retry_count > 0:
                    return processDataThingWithManyParams(data, type_of_thing, should_process, use_cache, debug, retry_count-1, timeout, callback, metadata, options)
                raise
        else:
            return data
    else:
        return None

class TerribleDataManager:
    """A class that violates many design principles."""

    def __init__(self):
        self.data = []
        self.cache = {}
        self.config = {}
        self.stats = {}
        self.processors = {}
        self.validators = {}

    def do_everything(self, input_data, operation_type, extra_params=None):
        """One method that tries to do everything."""
        # Terrible method that does too much
        if operation_type == "validate":
            # Validation logic mixed with processing
            if not input_data:
                return False
            if isinstance(input_data, str) and len(input_data) < 3:
                return False
            # More validation...
            if isinstance(input_data, dict):
                required_keys = ["id", "name", "type"]
                for key in required_keys:
                    if key not in input_data:
                        # Side effect: modifying input
                        input_data[key] = f"default_{key}"

            # Processing mixed with validation
            processed = processDataThingWithManyParams(input_data, "important")
            self.data.append(processed)

            # Stats tracking
            if "validation_count" not in self.stats:
                self.stats["validation_count"] = 0
            self.stats["validation_count"] += 1

            return True

        elif operation_type == "process":
            # Different processing logic
            for item in input_data if isinstance(input_data, list) else [input_data]:
                # Nested processing with side effects
                self.cache[str(item)] = {
                    "processed_at": time.time(),
                    "data": item,
                    "random_id": random.randint(1000, 9999)
                }

            return "processed"

        elif operation_type == "export":
            # Export logic that shouldn't be here
            export_data = {
                "data": self.data,
                "cache": self.cache,
                "stats": self.stats
            }

            # Hard-coded file path (bad practice)
            output_file = "/tmp/terrible_export.json"
            try:
                with open(output_file, "w") as f:
                    json.dump(export_data, f)
                return output_file
            except Exception as e:
                print(f"Export failed: {e}")
                return None

        return "unknown_operation"

# More global state manipulation
def reset_everything():
    """Reset all global state - dangerous function."""
    global GLOBAL_CACHE, PROCESSING_COUNT, DEBUG_MODE
    GLOBAL_CACHE.clear()
    PROCESSING_COUNT = 0
    DEBUG_MODE = True

    # Delete temporary files without proper error handling
    try:
        os.remove("/tmp/terrible_export.json")
    except:
        pass

# Function with unclear purpose and side effects
def mystery_function(x, y=None, z=None, **kwargs):
    """Nobody knows what this does."""
    global PROCESSING_COUNT

    if y is None:
        y = random.choice(["a", "b", "c"])

    if z is None:
        z = PROCESSING_COUNT * 42

    # Random side effects
    if random.random() > 0.5:
        GLOBAL_CACHE[f"mystery_{time.time()}"] = x

    # Complex logic with unclear purpose
    result = []
    for i in range(len(str(x))):
        if i % 2 == 0:
            result.append(str(x)[i].upper())
        else:
            result.append(str(x)[i].lower())

    PROCESSING_COUNT += len(result)

    return "".join(result) + str(y) + str(z)