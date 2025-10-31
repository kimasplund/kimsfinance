//! File integrity verification using SHA-256 checksums
//!
//! Provides functions to calculate and verify file checksums, useful for:
//! - Detecting corrupted downloads
//! - Verifying data integrity
//! - Comparing file versions

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Calculate SHA-256 checksum of a file
///
/// Reads file in 8KB chunks to minimize memory usage for large files.
///
/// # Arguments
/// * `path` - Path to file
///
/// # Returns
/// Hexadecimal checksum string (64 characters)
///
/// # Errors
/// Returns `io::Error` if file cannot be read
///
/// # Performance
/// - Typical throughput: 200-500 MB/s
/// - Memory usage: Fixed 8KB buffer
/// - Suitable for multi-GB files
///
/// # Example
/// ```no_run
/// # use kimsfinance_core::validation::calculate_checksum;
/// let checksum = calculate_checksum("data.csv")?;
/// println!("SHA-256: {}", checksum);
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn calculate_checksum<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify file checksum matches expected value
///
/// Compares calculated checksum against expected value (case-insensitive).
///
/// # Arguments
/// * `file_path` - Path to file
/// * `expected` - Expected checksum (hexadecimal string)
///
/// # Returns
/// `true` if checksums match, `false` otherwise
///
/// # Errors
/// Returns `io::Error` if file cannot be read
///
/// # Example
/// ```no_run
/// # use kimsfinance_core::validation::verify_checksum;
/// let expected = "abc123..."; // Known good checksum
/// if verify_checksum("data.csv", expected)? {
///     println!("File verified!");
/// } else {
///     eprintln!("Checksum mismatch - file may be corrupted");
/// }
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn verify_checksum<P: AsRef<Path>>(file_path: P, expected: &str) -> io::Result<bool> {
    let actual = calculate_checksum(file_path)?;
    Ok(actual.eq_ignore_ascii_case(expected))
}

/// Calculate checksum from in-memory data
///
/// Useful for verifying data that's already loaded.
///
/// # Arguments
/// * `data` - Byte slice to hash
///
/// # Returns
/// Hexadecimal checksum string (64 characters)
///
/// # Example
/// ```
/// # use kimsfinance_core::validation::checksum_bytes;
/// let data = b"test data";
/// let checksum = checksum_bytes(data);
/// assert_eq!(checksum.len(), 64); // SHA-256 = 64 hex chars
/// ```
pub fn checksum_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Verify checksums for multiple files
///
/// Batch verification for datasets with manifest files.
///
/// # Arguments
/// * `files` - Slice of (file_path, expected_checksum) tuples
///
/// # Returns
/// Vector of (file_path, verification_result) tuples
///
/// # Example
/// ```no_run
/// # use kimsfinance_core::validation::verify_checksums;
/// let files = vec![
///     ("file1.csv", "abc123..."),
///     ("file2.csv", "def456..."),
/// ];
/// let results = verify_checksums(&files);
/// for (path, ok) in results {
///     if let Ok(verified) = ok {
///         println!("{}: {}", path, if verified { "OK" } else { "FAILED" });
///     }
/// }
/// ```
pub fn verify_checksums<P: AsRef<Path> + Clone>(files: &[(P, &str)]) -> Vec<(P, io::Result<bool>)> {
    files
        .iter()
        .map(|(path, expected)| {
            let result = verify_checksum(path.as_ref(), expected);
            (path.clone(), result)
        })
        .collect()
}

/// Checksum verification report
///
/// Aggregates batch verification results.
#[derive(Debug, Clone)]
pub struct ChecksumReport {
    pub total_files: usize,
    pub verified: usize,
    pub failed: usize,
    pub errors: usize,
}

impl ChecksumReport {
    /// Generate report from batch verification results
    ///
    /// # Arguments
    /// * `results` - Results from `verify_checksums`
    ///
    /// # Returns
    /// Aggregated report
    pub fn from_results<P>(results: &[(P, io::Result<bool>)]) -> Self {
        let mut verified = 0;
        let mut failed = 0;
        let mut errors = 0;

        for (_, result) in results {
            match result {
                Ok(true) => verified += 1,
                Ok(false) => failed += 1,
                Err(_) => errors += 1,
            }
        }

        Self {
            total_files: results.len(),
            verified,
            failed,
            errors,
        }
    }

    /// Check if all files passed verification
    pub fn all_verified(&self) -> bool {
        self.verified == self.total_files && self.failed == 0 && self.errors == 0
    }

    /// Print human-readable summary
    pub fn print_summary(&self) {
        println!("=== Checksum Verification Report ===");
        println!("Total files: {}", self.total_files);
        println!("Verified: {}", self.verified);
        println!("Failed: {}", self.failed);
        println!("Errors: {}", self.errors);
        if self.all_verified() {
            println!("Status: ALL VERIFIED ✓");
        } else {
            println!("Status: VERIFICATION FAILED ✗");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_checksum_calculation() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "test data").unwrap();

        let checksum = calculate_checksum(file.path()).unwrap();
        assert_eq!(checksum.len(), 64); // SHA256 = 64 hex chars
    }

    #[test]
    fn test_checksum_verification_match() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "test").unwrap();

        let checksum = calculate_checksum(file.path()).unwrap();
        assert!(verify_checksum(file.path(), &checksum).unwrap());
    }

    #[test]
    fn test_checksum_verification_mismatch() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "test").unwrap();

        let wrong_checksum = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(!verify_checksum(file.path(), wrong_checksum).unwrap());
    }

    #[test]
    fn test_checksum_case_insensitive() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "test").unwrap();

        let checksum_lower = calculate_checksum(file.path()).unwrap();
        let checksum_upper = checksum_lower.to_uppercase();

        assert!(verify_checksum(file.path(), &checksum_upper).unwrap());
    }

    #[test]
    fn test_checksum_bytes() {
        let data1 = b"test";
        let data2 = b"test";
        let data3 = b"different";

        let checksum1 = checksum_bytes(data1);
        let checksum2 = checksum_bytes(data2);
        let checksum3 = checksum_bytes(data3);

        assert_eq!(checksum1, checksum2);
        assert_ne!(checksum1, checksum3);
        assert_eq!(checksum1.len(), 64);
    }

    #[test]
    fn test_checksum_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let checksum = calculate_checksum(file.path()).unwrap();
        assert_eq!(checksum.len(), 64);
        // SHA-256 of empty string is known value
        assert_eq!(
            checksum,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_batch_verification() {
        let mut file1 = NamedTempFile::new().unwrap();
        let mut file2 = NamedTempFile::new().unwrap();
        write!(file1, "data1").unwrap();
        write!(file2, "data2").unwrap();

        let checksum1 = calculate_checksum(file1.path()).unwrap();
        let checksum2 = calculate_checksum(file2.path()).unwrap();

        let files = vec![
            (file1.path().to_str().unwrap(), checksum1.as_str()),
            (file2.path().to_str().unwrap(), checksum2.as_str()),
        ];

        let results = verify_checksums(&files);
        assert_eq!(results.len(), 2);
        assert!(results[0].1.as_ref().unwrap());
        assert!(results[1].1.as_ref().unwrap());
    }

    #[test]
    fn test_checksum_report() {
        let mut file1 = NamedTempFile::new().unwrap();
        let mut file2 = NamedTempFile::new().unwrap();
        write!(file1, "data1").unwrap();
        write!(file2, "data2").unwrap();

        let checksum1 = calculate_checksum(file1.path()).unwrap();
        let wrong_checksum = "0000000000000000000000000000000000000000000000000000000000000000";

        let files = vec![
            (file1.path().to_str().unwrap(), checksum1.as_str()),
            (file2.path().to_str().unwrap(), wrong_checksum),
        ];

        let results = verify_checksums(&files);
        let report = ChecksumReport::from_results(&results);

        assert_eq!(report.total_files, 2);
        assert_eq!(report.verified, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.errors, 0);
        assert!(!report.all_verified());
    }

    #[test]
    fn test_large_file_checksum() {
        // Test with 1MB file to verify chunked reading
        let mut file = NamedTempFile::new().unwrap();
        let data = vec![0u8; 1024 * 1024]; // 1MB of zeros
        file.write_all(&data).unwrap();

        let checksum = calculate_checksum(file.path()).unwrap();
        assert_eq!(checksum.len(), 64);
        // Should produce consistent checksum
        let checksum2 = calculate_checksum(file.path()).unwrap();
        assert_eq!(checksum, checksum2);
    }
}
