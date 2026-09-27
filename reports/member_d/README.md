# Member D reports

Generated reports are runtime artifacts and should not be committed unless the team explicitly changes policy.

Expected runtime layout:

- `reports/member_d/<version>/evaluation.json`
- `reports/member_d/<version>/evaluation_summary.tsv`
- `reports/member_d/<version>/ownership_audit.json`
- `reports/member_d/<version>/comparison.json`
- `reports/member_d/<version>/comparison.tsv`
- `reports/member_d/<version>/runtime_budget.json`

Use `src.member_d_report.write_comparison_report()` for comparison tables. Missing measurements must be written as `N/A`, never fabricated as zero. Only rows with `feature_contract_valid=true` are eligible for final model/decoder selection.
