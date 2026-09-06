import pandas as pd

from fedqtrust.reporting.plots import save_smoke_plot
from fedqtrust.reporting.tables import save_table


def test_plot_and_table_creation(tmp_path):
    _, pdf, png = save_smoke_plot(tmp_path)
    paths = save_table(pd.DataFrame({"a": [1]}), tmp_path, "table")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0
    assert paths["csv"].stat().st_size > 0
    assert paths["tex"].stat().st_size > 0
    assert paths["md"].stat().st_size > 0


def test_table_markdown_fallback_without_tabulate(tmp_path, monkeypatch):
    def raise_import_error(self, *args, **kwargs):
        raise ImportError("Missing optional dependency 'tabulate'")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", raise_import_error)
    paths = save_table(pd.DataFrame({"check": ["plot"], "status": ["PASS"]}), tmp_path, "fallback")
    text = paths["md"].read_text(encoding="utf-8")
    assert "check" in text
    assert "PASS" in text
