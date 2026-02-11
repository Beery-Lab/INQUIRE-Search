## INQUIRE-Search Case Studies: Data Format

This repository stores the code and data used for the case studies in the INQUIRE-Search paper. The CSV files exported for each case study follow a simple tabular format. This document explains the expected columns, common conventions, and recommended usage when filtering or publishing results.

### Typical CSV header

The exported CSV files use the following columns (CSV header example):

```
photo_id,species,marked,inat_url,latitude,longitude,month
```

### Column descriptions

- `photo_id` (string)
  - Unique identifier for the image/photo record. This is typically a numeric or alphanumeric id assigned by the source (for example, iNaturalist photo id or an internal id). Treat this field as the primary key for rows in the CSV.

- `species` (string)
  - Taxonomic identification associated with the photo. This may be a scientific name (preferred) or a common name depending on how the export was produced. If a case study uses a particular taxonomic column (e.g., `taxon_name`), the exported `species` column will contain the value used in the analysis.

- `marked` (boolean / string)
  - Indicates whether the photo/image was marked or labeled for some special status in the study (for example, manually verified, part of a training set, or identified as a target). Values are typically `TRUE`/`FALSE`, `1`/`0`, or `yes`/`no`. Check the specific case study scripts if you need a strict boolean conversion.

- `inat_url` (string / URL)
  - When available, the source URL for the record (for example an iNaturalist observation/photo page). This is useful for provenance and human inspection.

- `latitude`, `longitude` (decimal degrees, numeric)
  - Geographic coordinates in WGS84 (EPSG:4326) decimal degrees. These columns are only present in exports when spatial/metadata filtering was used in the search (for example when results were limited by bounding box, taxa range, or other geospatial constraints). Missing or unknown coordinates should be left empty or set to `NA`.

- `month` (integer 1–12)
  - The month (1–12) associated with the photo's observation date. This column is included in exports when temporal metadata filters are used in the search. Use integer month values (1 for January, 12 for December). If the original data does not include a month, this column may be empty or `NA`.

### Missing values and types
- Empty cells and `NA` are used for missing values. When ingesting these CSVs into analysis scripts, ensure the parser treats the empty fields as NA/null.
- Coordinates should parse as floating point numbers. Non-numeric strings in `latitude`/`longitude` should be treated as missing.

### Files and naming conventions
- CSV files for each case study are typically stored in `case_studies/<case_name>/data/` and follow names like `*_search_results_*.csv` or `Processed_*.csv`.
- Large GIS data are stored in `case_studies/<case_name>/gis/` (these are often large binary files and may be stored outside Git or tracked with Git LFS).
- Plots and derived outputs are stored in `case_studies/<case_name>/outputs/`.