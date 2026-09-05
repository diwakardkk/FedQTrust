from fedqtrust.publication import audit_publication_outputs, write_publication_audit


def test_publication_audit_reports_missing_artifacts(tmp_path):
    report = write_publication_audit(tmp_path, strict_infra=False)
    publishable, issues = audit_publication_outputs(tmp_path, strict_infra=False)
    assert report.exists()
    assert not publishable
    assert issues

