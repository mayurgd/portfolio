"""
Load evaluation dataset to Langfuse using Python SDK.

Follows Langfuse dataset pattern:
https://langfuse.com/docs/evaluation/experiments/datasets

Usage:
    python -m evaluation.load_dataset --dataset-name "tool-call-eval"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from langfuse.client import Langfuse
from agent_eval.config import get_langfuse_config  # cross-package dependency (intentional)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset_to_langfuse(dataset_name: str, dataset_file: str = "dataset.json"):
    """Load dataset from JSON file to Langfuse.
    
    Args:
        dataset_name: Name of the dataset in Langfuse
        dataset_file: Path to JSON dataset file
    """
    config = get_langfuse_config()
    if not config.enabled:
        logger.error("Langfuse not configured")
        sys.exit(1)
    
    # Load dataset from JSON
    dataset_path = Path(__file__).parent / dataset_file
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path) as f:
        items = json.load(f)
    
    logger.info(f"Loaded {len(items)} items from {dataset_file}")
    
    # Initialize Langfuse
    try:
        import httpx
        if config.ssl_verify is False:
            import os
            import urllib3
            os.environ.setdefault("CURL_CA_BUNDLE", "")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        httpx_client = httpx.Client(verify=config.ssl_verify)
        client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
            httpx_client=httpx_client,
        )
        logger.info(f"Connected to Langfuse at {config.host}")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        sys.exit(1)
    
    # Create dataset first (if it doesn't exist)
    try:
        client.create_dataset(name=dataset_name)
        logger.info(f"Created dataset: {dataset_name}")
    except Exception as e:
        # Dataset might already exist, that's okay
        logger.info(f"Dataset exists or creation skipped: {e}")
    
    # Upload dataset items
    uploaded = 0
    for item in items:
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                id=item["id"],
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item.get("metadata", {}),
            )
            uploaded += 1
            logger.info(f"  ✓ {item['id']}: {item['metadata'].get('description', '')}")
        except Exception as e:
            logger.error(f"  ✗ Failed to upload {item['id']}: {e}")
    
    client.flush()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Uploaded {uploaded}/{len(items)} items to dataset '{dataset_name}'")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Load dataset to Langfuse")
    parser.add_argument("--dataset-name", required=True, help="Dataset name in Langfuse")
    parser.add_argument("--dataset-file", default="dataset.json", help="JSON dataset file")
    
    args = parser.parse_args()
    load_dataset_to_langfuse(args.dataset_name, args.dataset_file)


if __name__ == "__main__":
    main()


