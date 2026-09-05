//! Common types and traits for data downloaders

use std::path::PathBuf;
use thiserror::Error;

/// Download configuration
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Base directory for all data (e.g., "data/")
    pub base_path: PathBuf,
    /// Number of parallel downloads
    pub parallel_downloads: usize,
    /// Verify checksums after download
    pub verify_checksums: bool,
    /// Resume interrupted downloads
    pub resume: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("data"),
            parallel_downloads: 4,
            verify_checksums: true,
            resume: true,
        }
    }
}

/// Download progress tracking
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Total bytes to download
    pub total_bytes: u64,
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Download speed (bytes/sec)
    pub speed_bps: f64,
    /// Estimated time remaining (seconds)
    pub eta_seconds: f64,
    /// Current file being downloaded
    pub current_file: String,
}

impl DownloadProgress {
    pub fn percent_complete(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.downloaded_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// Download errors
#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Rate limit exceeded: retry after {retry_after_secs}s")]
    RateLimit { retry_after_secs: u64 },

    #[error("API error: {0}")]
    ApiError(String),
}

/// Generic downloader trait
#[async_trait::async_trait]
pub trait Downloader {
    /// Download historical data for a symbol
    async fn download(
        &self,
        symbol: &str,
        start_date: chrono::NaiveDate,
        end_date: Option<chrono::NaiveDate>,
    ) -> Result<PathBuf, DownloadError>;

    /// Get download progress
    fn progress(&self) -> Option<DownloadProgress>;

    /// Cancel ongoing download
    async fn cancel(&self);
}
