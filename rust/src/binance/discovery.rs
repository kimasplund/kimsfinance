//! File discovery utilities for Binance trade data
//!
//! Provides pattern-based file discovery for locating Binance trade data files
//! across multiple months or days.
//!
//! # Features
//! - Simple glob pattern matching (supports `*` wildcard)
//! - Date range-based file discovery
//! - Sorted output for deterministic processing
//! - Handles both monthly and daily file patterns
//!
//! # Example
//! ```no_run
//! use kimsfinance_core::binance::{BinanceDataFinder, DateRange};
//!
//! let finder = BinanceDataFinder::new("/data/binance");
//! let range = DateRange::parse("2021-01-01", "2021-03-31")?;
//!
//! let files = finder.find_by_date_range(&range)?;
//! println!("Found {} files", files.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::date_utils::DateRange;
use std::fs;
use std::path::{Path, PathBuf};

/// File finder for Binance trade data
///
/// Discovers Binance trade data files by pattern matching or date range.
/// Designed for processing large datasets spanning multiple months.
pub struct BinanceDataFinder {
    base_path: PathBuf,
}

impl BinanceDataFinder {
    /// Create a new file finder for the given base directory
    ///
    /// # Arguments
    /// * `base_path` - Directory containing Binance trade data files
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::binance::BinanceDataFinder;
    /// let finder = BinanceDataFinder::new("/data/binance");
    /// ```
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Find files matching a glob pattern
    ///
    /// Supports simple glob patterns with `*` wildcard matching any characters.
    /// Files are returned in sorted order for deterministic processing.
    ///
    /// # Arguments
    /// * `pattern` - Glob pattern (e.g., "BTCUSDT-trades-*.zip")
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::binance::BinanceDataFinder;
    /// let finder = BinanceDataFinder::new("/data/binance");
    /// let files = finder.find_files("BTCUSDT-trades-2021-*.zip")?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn find_files(&self, pattern: &str) -> std::io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            // Edition 2024: Let chains for cleaner nested conditions
            if path.is_file()
                && let Some(filename) = path.file_name()
                && let Some(filename_str) = filename.to_str()
                && Self::matches_pattern(filename_str, pattern)
            {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    /// Find files for a date range
    ///
    /// Discovers files matching the months in the given date range.
    /// Tries multiple patterns to find both monthly and daily files.
    ///
    /// # Arguments
    /// * `range` - Date range to search
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::binance::{BinanceDataFinder, DateRange};
    /// let finder = BinanceDataFinder::new("/data/binance");
    /// let range = DateRange::parse("2021-01-01", "2021-03-31")?;
    ///
    /// let files = finder.find_by_date_range(&range)?;
    /// // Returns: BTCUSDT-trades-2021-01.zip, BTCUSDT-trades-2021-02.zip, ...
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn find_by_date_range(&self, range: &DateRange) -> std::io::Result<Vec<PathBuf>> {
        let months = range.months();
        let mut files = Vec::new();

        for month in months {
            // Try multiple patterns to find both monthly and daily files
            let patterns = vec![
                format!("*trades*{}*.zip", month),
                format!("*{}*.zip", month),
                format!("*trades*{}*.csv", month),
                format!("*{}*.csv", month),
            ];

            for pattern in patterns {
                let found = self.find_files(&pattern)?;
                files.extend(found);
            }
        }

        // Remove duplicates and sort
        files.sort();
        files.dedup();

        Ok(files)
    }

    /// Find files for a specific symbol and date range
    ///
    /// Filters results to only include files for the given trading pair symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `range` - Date range to search
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::binance::{BinanceDataFinder, DateRange};
    /// let finder = BinanceDataFinder::new("/data/binance");
    /// let range = DateRange::parse("2021-01-01", "2021-03-31")?;
    ///
    /// let files = finder.find_by_symbol_and_range("BTCUSDT", &range)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn find_by_symbol_and_range(
        &self,
        symbol: &str,
        range: &DateRange,
    ) -> std::io::Result<Vec<PathBuf>> {
        let all_files = self.find_by_date_range(range)?;

        // Filter by symbol
        let symbol_lower = symbol.to_lowercase();
        let filtered: Vec<PathBuf> = all_files
            .into_iter()
            .filter(|path| {
                // Edition 2024: Let chains for cleaner filtering
                if let Some(filename) = path.file_name()
                    && let Some(filename_str) = filename.to_str()
                {
                    filename_str.to_lowercase().contains(&symbol_lower)
                } else {
                    false
                }
            })
            .collect();

        Ok(filtered)
    }

    /// Check if a filename matches a glob pattern
    ///
    /// Simple glob matching supporting `*` wildcard.
    ///
    /// # Arguments
    /// * `filename` - Filename to check
    /// * `pattern` - Glob pattern with `*` wildcards
    fn matches_pattern(filename: &str, pattern: &str) -> bool {
        // Match everything
        if pattern == "*" {
            return true;
        }

        let parts: Vec<&str> = pattern.split('*').collect();

        // No wildcards - exact match
        if parts.len() == 1 {
            return filename == pattern;
        }

        let mut pos = 0;

        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }

            if i == 0 {
                // First part must match start
                if !filename.starts_with(part) {
                    return false;
                }
                pos = part.len();
            } else if i == parts.len() - 1 {
                // Last part must match end
                if !filename.ends_with(part) {
                    return false;
                }
            } else {
                // Middle parts must exist
                if let Some(idx) = filename[pos..].find(part) {
                    pos += idx + part.len();
                } else {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        // Exact match
        assert!(BinanceDataFinder::matches_pattern("file.txt", "file.txt"));
        assert!(!BinanceDataFinder::matches_pattern("file.csv", "file.txt"));

        // Wildcard at end
        assert!(BinanceDataFinder::matches_pattern("file.txt", "*.txt"));
        assert!(BinanceDataFinder::matches_pattern("test.txt", "*.txt"));
        assert!(!BinanceDataFinder::matches_pattern("file.csv", "*.txt"));

        // Wildcard at start
        assert!(BinanceDataFinder::matches_pattern("file.txt", "file.*"));
        assert!(BinanceDataFinder::matches_pattern("file.csv", "file.*"));
        assert!(!BinanceDataFinder::matches_pattern("test.txt", "file.*"));

        // Wildcard in middle
        assert!(BinanceDataFinder::matches_pattern(
            "BTCUSDT-trades-2021-01.zip",
            "*2021-01*.zip"
        ));
        assert!(BinanceDataFinder::matches_pattern(
            "BTCUSDT-trades-2021-01-full.zip",
            "*2021-01*.zip"
        ));
        assert!(!BinanceDataFinder::matches_pattern(
            "BTCUSDT-trades-2021-02.zip",
            "*2021-01*.zip"
        ));

        // Multiple wildcards
        assert!(BinanceDataFinder::matches_pattern(
            "BTCUSDT-trades-2021-01.zip",
            "BTCUSDT-*-2021-*.zip"
        ));
        assert!(BinanceDataFinder::matches_pattern(
            "ETHUSDT-trades-2021-12.zip",
            "*USDT-*-2021-*.zip"
        ));

        // Match all
        assert!(BinanceDataFinder::matches_pattern("anything.txt", "*"));
    }

    #[test]
    fn test_pattern_edge_cases() {
        // Empty pattern parts (consecutive wildcards)
        assert!(BinanceDataFinder::matches_pattern("file.txt", "**"));
        assert!(BinanceDataFinder::matches_pattern("file.txt", "file**txt"));

        // Pattern starts/ends with wildcard
        assert!(BinanceDataFinder::matches_pattern(
            "prefix-test-suffix",
            "*test*"
        ));
        assert!(BinanceDataFinder::matches_pattern("test-suffix", "*suffix"));
        assert!(BinanceDataFinder::matches_pattern("prefix-test", "prefix*"));
    }

    #[test]
    fn test_new_finder() {
        let finder = BinanceDataFinder::new("/data/binance");
        assert_eq!(finder.base_path, PathBuf::from("/data/binance"));
    }

    // Integration tests require actual filesystem
    #[test]
    fn test_find_files_empty_directory() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let finder = BinanceDataFinder::new(temp_dir.path());

        let files = finder.find_files("*.zip").unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_find_files_with_matches() {
        use std::fs::File;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let finder = BinanceDataFinder::new(temp_dir.path());

        // Create test files
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-01.zip")).unwrap();
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-02.zip")).unwrap();
        File::create(temp_dir.path().join("ETHUSDT-trades-2021-01.zip")).unwrap();
        File::create(temp_dir.path().join("unrelated.txt")).unwrap();

        // Find all zip files
        let files = finder.find_files("*.zip").unwrap();
        assert_eq!(files.len(), 3);

        // Find BTCUSDT files
        let files = finder.find_files("BTCUSDT-*.zip").unwrap();
        assert_eq!(files.len(), 2);

        // Find January files
        let files = finder.find_files("*2021-01.zip").unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_by_date_range() {
        use std::fs::File;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let finder = BinanceDataFinder::new(temp_dir.path());

        // Create test files
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-01.zip")).unwrap();
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-02.zip")).unwrap();
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-03.zip")).unwrap();
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-04.zip")).unwrap();

        let range = DateRange::parse("2021-01-01", "2021-03-31").unwrap();
        let files = finder.find_by_date_range(&range).unwrap();

        // Should find Jan, Feb, Mar (not Apr)
        assert_eq!(files.len(), 3);
        assert!(files[0].to_str().unwrap().contains("2021-01"));
        assert!(files[2].to_str().unwrap().contains("2021-03"));
    }

    #[test]
    fn test_find_by_symbol_and_range() {
        use std::fs::File;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let finder = BinanceDataFinder::new(temp_dir.path());

        // Create test files
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-01.zip")).unwrap();
        File::create(temp_dir.path().join("ETHUSDT-trades-2021-01.zip")).unwrap();
        File::create(temp_dir.path().join("BTCUSDT-trades-2021-02.zip")).unwrap();

        let range = DateRange::parse("2021-01-01", "2021-02-28").unwrap();
        let files = finder.find_by_symbol_and_range("BTCUSDT", &range).unwrap();

        // Should find only BTCUSDT files
        assert_eq!(files.len(), 2);
        assert!(files[0].to_str().unwrap().contains("BTCUSDT"));
        assert!(files[1].to_str().unwrap().contains("BTCUSDT"));
    }

    #[test]
    fn test_sorted_results() {
        use std::fs::File;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let finder = BinanceDataFinder::new(temp_dir.path());

        // Create files in random order
        File::create(temp_dir.path().join("file-2021-03.zip")).unwrap();
        File::create(temp_dir.path().join("file-2021-01.zip")).unwrap();
        File::create(temp_dir.path().join("file-2021-02.zip")).unwrap();

        let files = finder.find_files("file-*.zip").unwrap();

        // Should be sorted
        assert_eq!(files.len(), 3);
        assert!(files[0].to_str().unwrap().contains("2021-01"));
        assert!(files[1].to_str().unwrap().contains("2021-02"));
        assert!(files[2].to_str().unwrap().contains("2021-03"));
    }
}
