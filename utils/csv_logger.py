import csv
from typing import List, Dict, Any

def save_metrics_to_csv(filename: str, metrics: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Save a list of metric dictionaries to a CSV file."""
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
            