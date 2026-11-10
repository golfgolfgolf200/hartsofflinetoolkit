import pytest
from src.config import settings, apply_hardware_governor
from src.engine import SearchEngine

def test_hardware_governor():
    apply_hardware_governor()
    import torch
    assert torch.get_num_threads() == settings.worker_threads

def test_mrl_truncation_and_search():
    engine = SearchEngine()
    test_docs = [
        {
            "doc_id": "tax_invoice_1099",
            "source": "invoice_1099.txt",
            "page": 1,
            "content": "Invoice #1099: Total $4,500. Tax EIN: 88-1928371. Payable within 30 days."
        },
        {
            "doc_id": "sla_policy",
            "source": "sla_policy.txt",
            "page": 1,
            "content": "Service level agreement requires 99.99% uptime. System crashes result in 10% penalty fee."
        }
    ]
    indexed = engine.index_documents(test_docs)
    assert indexed == 2

    # Exact Keyword Retrieval
    kw_results = engine.hybrid_search("88-1928371", top_n=1)
    assert len(kw_results) > 0
    assert kw_results[0]["doc_id"] == "tax_invoice_1099"

    # Semantic Retrieval
    sem_results = engine.hybrid_search("What financial penalty happens if servers crash?", top_n=1)
    assert len(sem_results) > 0
    assert sem_results[0]["doc_id"] == "sla_policy"
