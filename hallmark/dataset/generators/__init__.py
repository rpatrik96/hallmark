from __future__ import annotations

from ._helpers import _clone_entry, is_preprint_source
from ._pools import FAKE_DOI_PREFIXES
from .batch import (
    TIER1_GENERATORS,
    generate_tier1_batch,
    generate_tier2_batch,
    generate_tier3_batch,
)
from .tier1 import (
    generate_fabricated_doi,
    generate_future_date,
    generate_nonexistent_venue,
    generate_placeholder_authors,
)
from .tier2 import (
    generate_chimeric_title,
    generate_hybrid_fabrication,
    generate_merged_citation,
    generate_partial_author_list,
    generate_preprint_as_published,
    generate_swapped_authors,
    generate_wrong_venue,
)
from .tier3 import (
    generate_arxiv_version_mismatch,
    generate_near_miss_title,
    generate_plausible_fabrication,
)

__all__ = [
    "FAKE_DOI_PREFIXES",
    "TIER1_GENERATORS",
    "_clone_entry",
    "generate_arxiv_version_mismatch",
    "generate_chimeric_title",
    "generate_fabricated_doi",
    "generate_future_date",
    "generate_hybrid_fabrication",
    "generate_merged_citation",
    "generate_near_miss_title",
    "generate_nonexistent_venue",
    "generate_partial_author_list",
    "generate_placeholder_authors",
    "generate_plausible_fabrication",
    "generate_preprint_as_published",
    "generate_swapped_authors",
    "generate_tier1_batch",
    "generate_tier2_batch",
    "generate_tier3_batch",
    "generate_wrong_venue",
    "is_preprint_source",
]
