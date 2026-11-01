# Nsight raw CSV export compatibility

Nsight Compute2025.3.1 rejects the obsolete `--units` option. It also rejects
`--print-metric-name` for the raw page; that option is details-only. The corrected
export command is:

```bash
ncu --import profile.ncu-rep --page raw --csv --print-units base
```

The parser accepts long metric-row CSV and wide raw CSV with a units row.
Matching completed captures are re-exported without repeating GPU collection.
The streaming preflight checks export before expensive model preparation.
All24 unit tests pass, including option spelling, wide units and failed-export
recovery. Final A/B datasets contain112 valid profiles with exported CSVs; the
earlier failed smoke export is not counted as a completed performance result.
