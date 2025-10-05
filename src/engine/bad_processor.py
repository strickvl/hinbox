"""Intentionally bad processor with circular dependencies and poor design."""

import sys
import time
import random
from typing import Any, Dict, List, Optional

# Circular import (bad practice)
from src.utils.terrible_helper import TerribleDataManager, processDataThingWithManyParams, GLOBAL_CACHE

class BadArticleProcessor:
    """Processor that violates many software engineering principles."""

    # Class variables that should be instance variables
    processed_articles = []
    error_count = 0
    success_count = 0

    def __init__(self, config=None):
        # Initialize with mutable default argument (bad practice)
        if config is None:
            config = {}

        self.config = config
        self.terrible_manager = TerribleDataManager()

        # Modifying class variables from instance
        BadArticleProcessor.processed_articles = []
        BadArticleProcessor.error_count = 0

    def process_article_badly(self, article_data, processing_options={}):
        """Process article with terrible practices."""

        # Modify global state
        global GLOBAL_CACHE

        # Long parameter list and unclear logic
        try:
            # Terrible error handling - catching everything
            result = self.do_complex_processing(
                article_data,
                processing_options.get("mode", "default"),
                processing_options.get("validate", True),
                processing_options.get("transform", True),
                processing_options.get("save", False),
                processing_options.get("notify", False),
                processing_options.get("retry", 3),
                processing_options.get("timeout", 30)
            )

            # Side effects everywhere
            BadArticleProcessor.success_count += 1
            BadArticleProcessor.processed_articles.append(article_data.get("id", "unknown"))

            return result

        except Exception as e:
            # Bad error handling
            BadArticleProcessor.error_count += 1
            print(f"Something went wrong: {e}")
            return None

    def do_complex_processing(self, data, mode, validate, transform, save, notify, retry, timeout):
        """Method with too many parameters and complex logic."""

        # Terrible nested conditionals
        if mode == "default":
            if validate:
                validation_result = self.terrible_manager.do_everything(data, "validate")
                if not validation_result:
                    if retry > 0:
                        time.sleep(0.1)
                        return self.do_complex_processing(data, mode, validate, transform, save, notify, retry-1, timeout)
                    else:
                        raise ValueError("Validation failed after retries")

            if transform:
                # Using the terrible helper function
                transformed = processDataThingWithManyParams(
                    data, "important", True, True, True, retry, timeout
                )

                # More complex logic
                if isinstance(transformed, dict):
                    for key in list(transformed.keys()):
                        if key.startswith("temp_"):
                            del transformed[key]
                        elif isinstance(transformed[key], str) and len(transformed[key]) > 100:
                            transformed[key] = transformed[key][:100] + "..."

                data = transformed

            if save:
                # Terrible save logic with hardcoded paths
                self.terrible_manager.do_everything(data, "export")

            if notify:
                # Notification with side effects
                self.send_notification(data, mode)

        elif mode == "fast":
            # Duplicate code instead of reusing
            if validate:
                if not data:
                    return None
                if not isinstance(data, dict):
                    data = {"content": str(data)}

            # Fast processing (just random transformation)
            data["processed_fast"] = True
            data["timestamp"] = time.time()
            data["random_value"] = random.randint(1, 1000)

        elif mode == "slow":
            # Intentionally slow processing
            time.sleep(random.uniform(0.1, 0.5))

            # Complex transformation
            if isinstance(data, dict):
                new_data = {}
                for k, v in data.items():
                    new_key = k.replace("_", "-").upper()
                    if isinstance(v, str):
                        new_data[new_key] = v.strip().replace("\n", " ")
                    elif isinstance(v, (int, float)):
                        new_data[new_key] = v * 1.1
                    else:
                        new_data[new_key] = str(v)
                data = new_data

        return data

    def send_notification(self, data, mode):
        """Notification method with side effects."""
        # Modifying global state in notification
        GLOBAL_CACHE[f"notification_{time.time()}"] = {
            "data_id": data.get("id", "unknown"),
            "mode": mode,
            "sent_at": time.time()
        }

        # Print notification (should use proper logging)
        print(f"NOTIFICATION: Processed {data.get('id', 'unknown')} in {mode} mode")

    @classmethod
    def get_statistics(cls):
        """Class method that exposes internal state."""
        return {
            "processed_count": len(cls.processed_articles),
            "error_count": cls.error_count,
            "success_count": cls.success_count,
            "processed_ids": cls.processed_articles  # Exposing internal list
        }

    @staticmethod
    def reset_all_state():
        """Static method that modifies class state - confusing design."""
        BadArticleProcessor.processed_articles.clear()
        BadArticleProcessor.error_count = 0
        BadArticleProcessor.success_count = 0

        # Also reset global state from other module
        from src.utils.terrible_helper import reset_everything
        reset_everything()

# Module-level function that depends on class state
def get_global_processing_status():
    """Function that breaks encapsulation."""
    return {
        "total_processed": len(BadArticleProcessor.processed_articles),
        "error_rate": (BadArticleProcessor.error_count /
                      max(1, BadArticleProcessor.error_count + BadArticleProcessor.success_count)) * 100,
        "cache_size": len(GLOBAL_CACHE)
    }

# Another terrible function
def batch_process_with_terrible_design(articles_list, options_dict={}):
    """Batch processing with terrible error handling and design."""

    processor = BadArticleProcessor(options_dict.get("config", {}))
    results = []
    errors = []

    for i, article in enumerate(articles_list):
        try:
            # Terrible progress tracking
            if i % 10 == 0:
                print(f"Processing article {i}/{len(articles_list)}")

            # Process with random delays
            if random.random() > 0.9:
                time.sleep(random.uniform(0.1, 0.3))

            result = processor.process_article_badly(article, options_dict)

            if result:
                results.append(result)
            else:
                errors.append(f"Failed to process article {i}")

        except Exception as e:
            # Swallow exceptions and continue
            errors.append(f"Exception at article {i}: {str(e)}")
            continue

    # Return inconsistent data structure
    if len(errors) > len(results):
        return {"status": "mostly_failed", "errors": errors, "some_results": results}
    else:
        return {"results": results, "errors": errors, "status": "ok"}