"""Baseline implementations for HALLMARK.

All baselines are registered via the registry module. Use the registry
to discover, check availability, and run baselines:

    from hallmark.baselines.registry import list_baselines, run_baseline
    available = list_baselines(free_only=True)
    predictions = run_baseline("doi_only", entries)
"""
