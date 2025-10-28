use kimsfinance_core::gpu::{GpuDevice, RocBatch, execute_batch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing persistent kernel with minimal example...");

    let device = GpuDevice::new()?;

    // Single task test using RocBatch type alias
    let mut batch = RocBatch::new();
    let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
    batch.add_task(data, 14);

    println!("Executing ROC batch with 1 task...");
    let results = execute_batch(&device, &batch)?;

    println!("Success! Got {} results", results.len());
    println!(
        "First 5 results: {:?}",
        &results[0][..5.min(results[0].len())]
    );

    Ok(())
}
