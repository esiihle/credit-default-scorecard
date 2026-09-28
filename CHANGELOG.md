# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0]

### Added
- **Value-labelled output columns.** The output-name template now supports a
  `{label}` placeholder (slugified answer label), so indicators can be named
  `Q3_krunch` instead of `Q3_4`. A `--labels` CLI flag switches the default
  template to `{qid}_{label}`. Existing codeframes are unaffected.
- **Exclusive-code integrity check.** A multi-response question can declare
  `exclusive: [<code>]` (e.g. a "None of these" option). The validator flags
  any respondent who selects an exclusive code alongside other codes.
- **Penetration/frequency stats.** New `stats` subcommand and `penetration()`
  API compute per-code selection counts and percentages, weighted when a weight
  column is supplied, with off-base (NaN) respondents excluded from the base.
- **Parquet & TSV I/O.** `convert`, `validate`, and `stats` now read and write
  `.parquet` and `.tsv` in addition to `.csv`, chosen by file extension.
  Parquet support requires the `parquet` extra (`pip install '.[parquet]'`).
- **Logging with `--verbose` / `--quiet`** on every subcommand; status messages
  go to stderr so data output on stdout stays clean and pipeable.

### Changed
- Indicator-column naming is now centralised in a single `format_indicator`
  helper shared by the converter and validator.
- The example codeframe gained a realistic exclusive "None of these" code (9)
  on Q3; the synthetic generator emits it for respondents who bought nothing
  and can inject a co-occurrence violation with `--dirty`.

## [0.1.0]

### Added
- Initial release: codeframe-driven compact→binary conversion for multi-column
  and delimited multi-response storage, passthrough of all other columns,
  per-question missing-data policy (`zero` / `nan`), and an independent
  validator (unknown codes, unparseable tokens, round-trip count check, binary
  domain check). CLI, synthetic data generator, tests, and CI included.
