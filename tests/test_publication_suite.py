from fedqtrust.publication import REQUIRED_FIGURES, REQUIRED_TABLES, audit_publication_outputs
from fedqtrust.publication_suite import PublicationSuiteConfig, run_publication_suite


def test_publication_suite_generates_required_artifacts(tmp_path):
    out = run_publication_suite(
        PublicationSuiteConfig(
            output_dir=str(tmp_path),
            mode="test",
            device="cpu",
            rounds=2,
            seeds=(42,),
        )
    )

    for stem in REQUIRED_FIGURES:
        assert (out / "paper_figures" / f"{stem}.pdf").stat().st_size > 0
        assert (out / "paper_figures" / f"{stem}.png").stat().st_size > 0

    for stem in REQUIRED_TABLES:
        assert (out / "paper_tables" / f"{stem}.csv").stat().st_size > 0
        assert (out / "paper_tables" / f"{stem}.md").stat().st_size > 0
        assert (out / "paper_tables" / f"{stem}.tex").stat().st_size > 0

    publishable, issues = audit_publication_outputs(out, strict_infra=False)
    assert publishable, [issue.message for issue in issues]
