import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)


def analyze_column(df, col_name):
    """Analyze statistics for a single column."""
    print(f"\n{'='*60}")
    print(f"Field: {col_name}")
    print(f"{'='*60}")

    col_data = df[col_name]

    # Basic info
    print(f"Data type: {col_data.dtype}")
    print(f"Total rows: {len(col_data):,}")
    print(f"Non-null count: {col_data.notna().sum():,}")
    print(f"Null count: {col_data.isna().sum():,} ({col_data.isna().sum()/len(col_data)*100:.2f}%)")

    non_null_data = col_data.dropna()

    if len(non_null_data) == 0:
        print("All values are null")
        return

    unique_count = non_null_data.nunique()
    print(f"Unique values: {unique_count:,}")

    # Numeric statistics
    if pd.api.types.is_numeric_dtype(col_data):
        print(f"\nNumeric statistics:")
        print(f"  Min: {non_null_data.min()}")
        print(f"  Max: {non_null_data.max()}")
        print(f"  Mean: {non_null_data.mean():.2f}")
        print(f"  Median: {non_null_data.median():.2f}")
        print(f"  Std: {non_null_data.std():.2f}")

        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = non_null_data.quantile(p / 100)
            print(f"  {p}%: {val:.2f}")

    # Value distribution (top 20)
    print(f"\nValue distribution (top 20):")
    value_counts = non_null_data.value_counts()

    display_count = min(len(value_counts), 20)
    for val, count in value_counts.head(display_count).items():
        percentage = count / len(non_null_data) * 100
        print(f"  {val}: {count:,} ({percentage:.2f}%)")

    if len(value_counts) > 20:
        print(f"  ... ({len(value_counts) - 20} more values)")


def print_null_summary(df, columns):
    """Print null value summary for selected columns."""
    print(f"\nNull value summary:")
    for col in columns:
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        print(f"  {col}: {null_count:,} nulls ({null_pct:.2f}%)")


def print_dataset_overview(df):
    """Print basic dataset information."""
    print(f"\n{'='*60}")
    print("Dataset Overview")
    print(f"{'='*60}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Shape: {df.shape}")

    print(f"\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nData type distribution:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} column(s)")

    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory usage: {memory_mb:.2f} MB")


def print_data_quality_report(df):
    """Print data quality report."""
    print(f"\n{'='*60}")
    print("Data Quality Report")
    print(f"{'='*60}")

    total_cells = len(df) * len(df.columns)
    total_missing = df.isnull().sum().sum()
    print(f"Total cells: {total_cells:,}")
    print(f"Total missing: {total_missing:,}")
    print(f"Overall missing rate: {total_missing/total_cells*100:.2f}%")

    missing_stats = [
        (col, df[col].isna().sum(), df[col].isna().sum() / len(df) * 100)
        for col in df.columns if df[col].isna().sum() > 0
    ]

    if missing_stats:
        print(f"\nMissing values by field:")
        missing_stats.sort(key=lambda x: x[1], reverse=True)
        for col, count, pct in missing_stats:
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print(f"\nNo missing values in any field")

    complete_rows = df.notna().all(axis=1).sum()
    incomplete_rows = len(df) - complete_rows
    print(f"\nComplete rows (no nulls): {complete_rows:,} ({complete_rows/len(df)*100:.2f}%)")
    print(f"Incomplete rows (>=1 null): {incomplete_rows:,} ({incomplete_rows/len(df)*100:.2f}%)")


def print_column_variance_report(df):
    """Print constant, low-variance, and high-cardinality column report."""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\nConstant columns (1 unique value) ({len(constant_cols)}):")
        for col in constant_cols:
            unique_val = df[col].iloc[0] if len(df) > 0 else None
            print(f"  - {col}: {unique_val}")
    else:
        print(f"\nAll fields have more than one unique value")

    low_var_cols = [(col, df[col].nunique()) for col in df.columns if df[col].nunique() <= 5]
    if low_var_cols:
        print(f"\nLow-variance fields (unique values <= 5) ({len(low_var_cols)}):")
        for col, n in low_var_cols:
            print(f"  - {col}: {n} unique value(s)")

    threshold = len(df) * 0.9
    high_card_cols = [(col, df[col].nunique()) for col in df.columns if df[col].nunique() > threshold]
    if high_card_cols:
        print(f"\nHigh-cardinality fields (unique values > 90% of rows) ({len(high_card_cols)}):")
        for col, n in high_card_cols:
            print(f"  - {col}: {n} unique values ({n/len(df)*100:.1f}%)")


def build_summary_table(df):
    """Build and return a summary DataFrame for all columns."""
    rows = []
    for col in df.columns:
        col_data = df[col]
        rows.append({
            'field': col,
            'dtype': str(col_data.dtype),
            'non_null': col_data.notna().sum(),
            'null': col_data.isna().sum(),
            'null_pct': f"{col_data.isna().sum()/len(col_data)*100:.2f}",
            'unique': col_data.nunique(),
        })
    return pd.DataFrame(rows)


def analyze_epistart_dates(df):
    """Analyze the epistart date field and produce yearly breakdowns."""
    print(f"\n{'='*60}")
    print("epistart Date Analysis")
    print(f"{'='*60}")

    if 'epistart' not in df.columns:
        print("\nField 'epistart' not found in data")
        return

    epistart_data = df['epistart'].dropna()
    if len(epistart_data) == 0:
        print("\nepistart is entirely null")
        return

    print(f"\nepistart non-null count: {len(epistart_data):,}")

    try:
        epistart_dates = pd.to_datetime(epistart_data, errors='coerce')
        valid_dates = epistart_dates.dropna()

        print(f"Successfully parsed dates: {len(valid_dates):,}")
        print(f"Failed to parse: {len(epistart_dates) - len(valid_dates):,}")

        if len(valid_dates) == 0:
            print("\nNo dates could be parsed")
            return

        years = valid_dates.dt.year

        print(f"\nYear range:")
        print(f"  Earliest: {years.min()}")
        print(f"  Latest: {years.max()}")
        print(f"  Span: {years.max() - years.min()} years")

        print(f"\nYear distribution:")
        for year, count in years.value_counts().sort_index().items():
            pct = count / len(years) * 100
            print(f"  {int(year)}: {count:,} records ({pct:.2f}%)")

        # Attach parsed year to a working copy
        df_with_year = df.copy()
        df_with_year['epistart_parsed'] = pd.to_datetime(df['epistart'], errors='coerce')
        df_with_year['year'] = df_with_year['epistart_parsed'].dt.year

        _analyze_dsource_by_year(df_with_year)
        _extract_year_subset(df, df_with_year, 2021)
        _extract_year_subset(df, df_with_year, 2022)

    except Exception as e:
        print(f"\nDate parsing error: {e}")
        print("Sample epistart values:")
        for val in epistart_data.head(10):
            print(f"  {val}")


def _analyze_dsource_by_year(df_with_year):
    """Print dsource distribution per year and cross-tabulations."""
    if 'dsource' not in df_with_year.columns:
        print("\nField 'dsource' not found in data")
        return

    valid_data = df_with_year[df_with_year['year'].notna() & df_with_year['dsource'].notna()]
    if len(valid_data) == 0:
        print("\nNo records with both valid year and dsource")
        return

    print(f"\n{'='*60}")
    print("dsource Distribution by Year")
    print(f"{'='*60}")

    for year in sorted(valid_data['year'].unique()):
        year_data = valid_data[valid_data['year'] == year]
        print(f"\n{int(year)} ({len(year_data):,} records):")
        for dsource, count in year_data['dsource'].value_counts().items():
            pct = count / len(year_data) * 100
            print(f"  dsource={dsource}: {count:,} ({pct:.2f}%)")

    # Cross-tabulation (counts)
    print(f"\n{'='*60}")
    print("Year-dsource Cross-tabulation")
    print(f"{'='*60}")

    crosstab = pd.crosstab(
        valid_data['year'].astype(int),
        valid_data['dsource'],
        margins=True,
        margins_name='Total',
    )
    print("\n" + crosstab.to_string())

    crosstab_file = '/data/xiejingyi/WAGLE/year_dsource_crosstab.csv'
    crosstab.to_csv(crosstab_file)
    print(f"\nCross-tabulation saved to: {crosstab_file}")

    # Cross-tabulation (row percentages)
    print(f"\n{'='*60}")
    print("Year-dsource Cross-tabulation (row %)")
    print(f"{'='*60}")

    crosstab_pct = pd.crosstab(
        valid_data['year'].astype(int),
        valid_data['dsource'],
        normalize='index',
    ) * 100
    print("\n" + crosstab_pct.round(2).to_string())

    crosstab_pct_file = '/data/xiejingyi/WAGLE/year_dsource_crosstab_pct.csv'
    crosstab_pct.to_csv(crosstab_pct_file)
    print(f"\nPercentage cross-tabulation saved to: {crosstab_pct_file}")


def _extract_year_subset(df_original, df_with_year, year):
    """Extract and save data for a specific year."""
    print(f"\n{'='*60}")
    print(f"Extracting {year} Data")
    print(f"{'='*60}")

    df_year = df_with_year[df_with_year['year'] == year].copy()

    if len(df_year) == 0:
        print(f"\nNo records found for {year}")
        return

    df_year_output = df_year.drop(columns=['epistart_parsed', 'year'], errors='ignore')

    print(f"\n{year} data summary:")
    print(f"  Total records: {len(df_year):,}")
    print(f"  Proportion of full dataset: {len(df_year)/len(df_original)*100:.2f}%")

    if 'dsource' in df_year_output.columns:
        print(f"\n{year} dsource distribution:")
        for dsource, count in df_year_output['dsource'].value_counts().sort_index().items():
            pct = count / len(df_year) * 100
            print(f"  dsource={dsource}: {count:,} ({pct:.2f}%)")

    output_file = f'/data/xiejingyi/dataset/hesin_{year}.csv'
    df_year_output.to_csv(output_file, index=False)
    print(f"\n{year} data saved to: {output_file}")


def main():
    file_path = '/data/xiejingyi/dataset/hesin.csv'

    print("=" * 60)
    print("UK Biobank hesin.csv Dataset Analysis")
    print("=" * 60)

    selected_columns = [
        'admimeth_uni', 'admisorc_uni', 'classpat_uni', 'intmanag_uni',
        'source', 'mainspef_uni', 'tretspef_uni', 'operstat',
        'dsource', 'epidur', 'epistart', 'epiend', 'epitype', 'elecdur',
    ]

    print(f"\nLoading data: {file_path}")
    print(f"Selected fields: {', '.join(selected_columns)}")

    try:
        df_full = pd.read_csv(file_path)
        print(f"Data loaded successfully. Original shape: {df_full.shape}")

        missing_cols = [c for c in selected_columns if c not in df_full.columns]
        if missing_cols:
            print(f"\nWarning: fields not found in data: {missing_cols}")
            print(f"Available fields: {list(df_full.columns)}")
            return

        df = df_full[selected_columns].copy()
        print(f"Shape after selection: {df.shape}")

        print_null_summary(df, selected_columns)
        print(f"\nNote: all rows retained (nulls not dropped)")
        print(f"Row count: {len(df):,}")

    except FileNotFoundError:
        print(f"Error: file not found {file_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Overview
    print_dataset_overview(df)

    # Per-column analysis
    print(f"\n{'='*60}")
    print("Detailed Field Analysis")
    print(f"{'='*60}")

    for col in df.columns:
        analyze_column(df, col)

    # Summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")

    summary_df = build_summary_table(df)
    print("\n" + summary_df.to_string(index=False))

    summary_file = '/data/xiejingyi/WAGLE/hesin_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

    # Save cleaned subset
    cleaned_file = '/data/xiejingyi/dataset/hesin_cleaned.csv'
    df.to_csv(cleaned_file, index=False)
    print(f"Cleaned data saved to: {cleaned_file}")

    # Data quality
    print_data_quality_report(df)

    # Date analysis
    analyze_epistart_dates(df)

    # Column variance
    print_column_variance_report(df)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()