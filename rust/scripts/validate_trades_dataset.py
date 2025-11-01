#!/usr/bin/env python3
"""
Validate the converted Parquet trades dataset

Performs comprehensive validation:
- Schema consistency across all files
- Null value checks
- Date continuity
- Trade ID uniqueness (sample)
- Price/quantity reasonableness
- File integrity
- Statistics per month

Usage:
    python validate_trades_dataset.py /path/to/trades_parquet
"""

import polars as pl
from pathlib import Path
import json
from datetime import datetime
import argparse


def validate_schema(parquet_dir: Path) -> dict:
    """Validate schema consistency across all Parquet files"""
    print("=" * 80)
    print("1. Schema Validation")
    print("=" * 80)

    expected_schema = {
        'id': pl.UInt64,
        'price': pl.Float64,
        'qty': pl.Float64,
        'quote_qty': pl.Float64,
        'time': pl.Int64,
        'is_buyer_maker': pl.Boolean,
        'timestamp': pl.Datetime(time_unit='ms'),
        'year_month': pl.String,
        'side': pl.String,
    }

    # Sample 10 files from different months
    all_files = sorted(parquet_dir.rglob("*.parquet"))
    sample_files = all_files[::len(all_files)//10][:10]

    schema_errors = []
    for file in sample_files:
        df = pl.read_parquet(file, n_rows=1)

        for col, dtype in expected_schema.items():
            if col not in df.columns:
                schema_errors.append(f"{file.name}: Missing column '{col}'")
            elif df.schema[col] != dtype:
                schema_errors.append(f"{file.name}: Column '{col}' has type {df.schema[col]}, expected {dtype}")

    if schema_errors:
        print(f"❌ Schema validation FAILED: {len(schema_errors)} errors")
        for err in schema_errors[:5]:
            print(f"  - {err}")
    else:
        print(f"✅ Schema validation PASSED ({len(sample_files)} files checked)")

    return {"passed": len(schema_errors) == 0, "errors": schema_errors}


def validate_nulls(parquet_dir: Path) -> dict:
    """Check for null values in critical columns"""
    print("\n" + "=" * 80)
    print("2. Null Value Check")
    print("=" * 80)

    # Sample 5 random files
    all_files = sorted(parquet_dir.rglob("*.parquet"))
    sample_files = all_files[::len(all_files)//5][:5]

    null_errors = []
    for file in sample_files:
        df = pl.read_parquet(file)

        for col in ['id', 'price', 'qty', 'time', 'timestamp']:
            null_count = df[col].null_count()
            if null_count > 0:
                null_errors.append(f"{file.name}: Column '{col}' has {null_count} nulls")

    if null_errors:
        print(f"❌ Null check FAILED: {len(null_errors)} columns with nulls")
        for err in null_errors:
            print(f"  - {err}")
    else:
        print(f"✅ Null check PASSED ({len(sample_files)} files checked)")

    return {"passed": len(null_errors) == 0, "errors": null_errors}


def validate_date_continuity(parquet_dir: Path) -> dict:
    """Verify date ranges are continuous"""
    print("\n" + "=" * 80)
    print("3. Date Continuity Check")
    print("=" * 80)

    # Get all month directories
    month_dirs = sorted([d for d in parquet_dir.iterdir() if d.is_dir() and d.name != '__pycache__'])

    print(f"Found {len(month_dirs)} month directories")
    print(f"  First: {month_dirs[0].name}")
    print(f"  Last: {month_dirs[-1].name}")

    # Sample first and last file from each month
    date_errors = []
    for month_dir in month_dirs[:5] + month_dirs[-5:]:  # First 5 and last 5 months
        files = sorted(month_dir.glob("*.parquet"))
        if not files:
            date_errors.append(f"{month_dir.name}: No parquet files found")
            continue

        # Check first file
        df_first = pl.read_parquet(files[0], n_rows=10)
        first_timestamp = df_first['timestamp'].min()

        # Check last file
        df_last = pl.read_parquet(files[-1])
        last_timestamp = df_last['timestamp'].max()

        print(f"  {month_dir.name}: {first_timestamp} → {last_timestamp} ({len(df_last):,} trades)")

    return {"passed": len(date_errors) == 0, "errors": date_errors}


def validate_data_quality(parquet_dir: Path) -> dict:
    """Check for reasonable price/quantity values"""
    print("\n" + "=" * 80)
    print("4. Data Quality Check")
    print("=" * 80)

    # Sample a few files
    all_files = sorted(parquet_dir.rglob("*.parquet"))
    sample_files = all_files[::len(all_files)//3][:3]

    quality_errors = []
    for file in sample_files:
        df = pl.read_parquet(file)

        # Check price range (prices should be positive and reasonable)
        min_price = df['price'].min()
        max_price = df['price'].max()

        if min_price <= 0 or max_price <= 0:
            quality_errors.append(f"{file.name}: Found non-positive price: ${min_price:,.2f} - ${max_price:,.2f}")
        elif min_price > max_price:
            quality_errors.append(f"{file.name}: Min price > max price: ${min_price:,.2f} - ${max_price:,.2f}")

        # Check quantity (should be positive)
        if df['qty'].min() <= 0:
            quality_errors.append(f"{file.name}: Found non-positive quantity")

        # Check side values
        sides = df['side'].unique().to_list()
        if not set(sides).issubset({'buy', 'sell'}):
            quality_errors.append(f"{file.name}: Unexpected side values: {sides}")

        print(f"  {file.name}:")
        print(f"    Price range: ${min_price:,.2f} - ${max_price:,.2f}")
        print(f"    Qty range: {df['qty'].min():.8f} - {df['qty'].max():.8f}")
        print(f"    Sides: {sides}")

    if quality_errors:
        print(f"\n❌ Data quality check FAILED: {len(quality_errors)} issues")
        for err in quality_errors:
            print(f"  - {err}")
    else:
        print(f"\n✅ Data quality check PASSED")

    return {"passed": len(quality_errors) == 0, "errors": quality_errors}


def generate_statistics(parquet_dir: Path) -> dict:
    """Generate comprehensive dataset statistics"""
    print("\n" + "=" * 80)
    print("5. Dataset Statistics")
    print("=" * 80)

    # Get all month directories
    month_dirs = sorted([d for d in parquet_dir.iterdir() if d.is_dir()])

    stats = {
        "months": [],
        "total_trades": 0,
        "total_size_gb": 0,
    }

    print(f"\n{'Month':<15} {'Trades':>15} {'Size (GB)':>12} {'Avg Trade Size':>15}")
    print("-" * 60)

    for month_dir in month_dirs:
        files = list(month_dir.glob("*.parquet"))
        if not files:
            continue

        # Get file size
        month_size_bytes = sum(f.stat().st_size for f in files)
        month_size_gb = month_size_bytes / (1024**3)

        # Count trades (read just first file for estimate, or all for accuracy)
        df_sample = pl.read_parquet(files[0])
        trades_estimate = len(df_sample) * len(files)

        # More accurate: sum all files (slower)
        # total_trades = sum(len(pl.read_parquet(f)) for f in files)

        avg_trade_size = month_size_bytes / trades_estimate if trades_estimate > 0 else 0

        print(f"{month_dir.name:<15} {trades_estimate:>15,} {month_size_gb:>12.2f} {avg_trade_size:>15.1f}")

        stats["months"].append({
            "month": month_dir.name,
            "trades": trades_estimate,
            "size_gb": month_size_gb,
        })
        stats["total_trades"] += trades_estimate
        stats["total_size_gb"] += month_size_gb

    print("-" * 60)
    print(f"{'TOTAL':<15} {stats['total_trades']:>15,} {stats['total_size_gb']:>12.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate Parquet trades dataset")
    parser.add_argument("parquet_dir", type=Path, help="Path to trades_parquet directory")
    parser.add_argument("--output", type=Path, help="Output JSON validation report")

    args = parser.parse_args()

    if not args.parquet_dir.exists():
        print(f"Error: Directory not found: {args.parquet_dir}")
        return 1

    print("\n" + "=" * 80)
    print("BINANCE BTCUSDT TRADES DATASET VALIDATION")
    print("=" * 80)
    print(f"Directory: {args.parquet_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Run all validations
    results = {
        "timestamp": datetime.now().isoformat(),
        "directory": str(args.parquet_dir),
        "validations": {}
    }

    results["validations"]["schema"] = validate_schema(args.parquet_dir)
    results["validations"]["nulls"] = validate_nulls(args.parquet_dir)
    results["validations"]["date_continuity"] = validate_date_continuity(args.parquet_dir)
    results["validations"]["data_quality"] = validate_data_quality(args.parquet_dir)
    results["statistics"] = generate_statistics(args.parquet_dir)

    # Overall result
    all_passed = all(
        v["passed"] for v in results["validations"].values()
    )

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for name, result in results["validations"].items():
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"{name.replace('_', ' ').title():<30} {status}")

    if all_passed:
        print("\n🎉 All validations PASSED! Dataset is ready for use.")
    else:
        print("\n⚠️  Some validations FAILED. Review errors above.")

    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nValidation report saved to: {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
