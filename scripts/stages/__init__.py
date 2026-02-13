"""HALLMARK dataset build pipeline stages.

Each stage is a pure function that takes in-memory data and returns in-memory data.
No stage reads or writes files directly — the orchestrator handles I/O.
"""
