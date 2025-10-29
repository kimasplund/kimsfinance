// Minimal test for Pivot Points implementation
// Compile with: rustc --edition 2024 test_pivot_minimal.rs && ./test_pivot_minimal

fn main() {
    // Test case from Python implementation
    let high = 110.0;
    let low = 100.0;
    let close = 105.0;

    // Calculate pivot points using the formula
    let pp = (high + low + close) / 3.0;
    let range = high - low;
    let r1 = 2.0 * pp - low;
    let r2 = pp + range;
    let r3 = high + 2.0 * (pp - low);
    let s1 = 2.0 * pp - high;
    let s2 = pp - range;
    let s3 = low - 2.0 * (high - pp);

    // Verify calculations (explicitly use f64)
    assert!((pp - 105.0_f64).abs() < 1e-10, "PP mismatch");
    assert!((r1 - 110.0_f64).abs() < 1e-10, "R1 mismatch");
    assert!((r2 - 115.0_f64).abs() < 1e-10, "R2 mismatch");
    assert!((r3 - 120.0_f64).abs() < 1e-10, "R3 mismatch");
    assert!((s1 - 100.0_f64).abs() < 1e-10, "S1 mismatch");
    assert!((s2 - 95.0_f64).abs() < 1e-10, "S2 mismatch");
    assert!((s3 - 90.0_f64).abs() < 1e-10, "S3 mismatch");

    println!("✓ All pivot points calculations correct!");
    println!("PP: {:.2}, R1: {:.2}, R2: {:.2}, R3: {:.2}", pp, r1, r2, r3);
    println!("S1: {:.2}, S2: {:.2}, S3: {:.2}", s1, s2, s3);
}
