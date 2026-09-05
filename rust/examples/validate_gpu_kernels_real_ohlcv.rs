#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "This example requires --features gpu\n\
         Run:\n\
         cargo run --release --example validate_gpu_kernels_real_ohlcv --features gpu"
    );
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kimsfinance_core::gpu::scan::{ScanOp, inclusive_scan_f32, wilder_smooth_f32};
    use kimsfinance_core::gpu::{
        GpuDevice, adx_gpu, dema_gpu, hma_gpu, kama_gpu, mfi_gpu, obv_gpu, tema_gpu,
        vwap_anchored_gpu,
    };
    use ndarray::Array1;
    use std::fs;
    use std::time::Instant;

    fn load_ohlcv_csv(
        path: &str,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>
    {
        let content = fs::read_to_string(path)?;

        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        for (idx, line) in content.lines().enumerate() {
            if idx == 0 {
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }

            let mut parts = line.split(',');
            let o = parts.next().ok_or("missing Open")?.parse::<f64>()?;
            let h = parts.next().ok_or("missing High")?.parse::<f64>()?;
            let l = parts.next().ok_or("missing Low")?.parse::<f64>()?;
            let c = parts.next().ok_or("missing Close")?.parse::<f64>()?;
            let v = parts.next().ok_or("missing Volume")?.parse::<f64>()?;

            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v);
        }

        Ok((open, high, low, close, volume))
    }

    fn obv_cpu(close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut out = vec![0.0; n];
        for i in 1..n {
            out[i] = if close[i] > close[i - 1] {
                out[i - 1] + volume[i]
            } else if close[i] < close[i - 1] {
                out[i - 1] - volume[i]
            } else {
                out[i - 1]
            };
        }
        out
    }

    fn scan_sum_cpu_f32(x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        let mut acc = 0.0f32;
        for &v in x {
            acc += v;
            out.push(acc);
        }
        out
    }

    fn wilder_cpu_f32(x: &[f32], period: usize) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![f32::NAN; n];
        if n < period || period == 0 {
            return out;
        }

        let mut seed_sum = 0.0f32;
        for &v in &x[..period] {
            seed_sum += v;
        }
        out[period - 1] = seed_sum / period as f32;

        let alpha = 1.0f32 / period as f32;
        for i in period..n {
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
        }
        out
    }

    fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .filter_map(|(x, y)| {
                if x.is_nan() || y.is_nan() {
                    None
                } else {
                    Some((x - y).abs())
                }
            })
            .fold(0.0f64, f64::max)
    }

    fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .filter_map(|(x, y)| {
                if x.is_nan() || y.is_nan() {
                    None
                } else {
                    Some((x - y).abs())
                }
            })
            .fold(0.0f32, f32::max)
    }

    fn finite_count_f64(v: &Array1<f64>) -> usize {
        v.iter().filter(|x| x.is_finite()).count()
    }

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../data/binance/BTCUSDT_5m_1y_ohlcv.csv".to_string());

    println!("=== Real-Data GPU Kernel Validation (OHLCV CSV) ===");
    println!("Input: {path}");

    let t0 = Instant::now();
    let (_open, high, low, close, volume) = load_ohlcv_csv(&path)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if close.len() < 100 {
        return Err("Need at least 100 rows in OHLCV CSV".into());
    }

    println!("Rows: {}", close.len());
    println!("Load time: {:.2} ms", load_ms);

    let device = GpuDevice::new()?;
    println!("GPU initialized\n");

    let close_arr = Array1::from_vec(close.clone());
    let _high_arr = Array1::from_vec(high.clone());
    let _low_arr = Array1::from_vec(low.clone());
    let volume_arr = Array1::from_vec(volume.clone());

    let period = 20usize;

    let t = Instant::now();
    let dema = dema_gpu(&device, &close_arr, period, None)?;
    println!(
        "DEMA kernel: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&dema)
    );

    let t = Instant::now();
    let tema = tema_gpu(&device, &close_arr, period, None)?;
    println!(
        "TEMA kernel: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&tema)
    );

    let t = Instant::now();
    let hma = hma_gpu(&device, &close_arr, period, None)?;
    println!(
        "HMA kernel: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&hma)
    );

    let t = Instant::now();
    let kama = kama_gpu(&device, &close_arr, 10, 2, 30, None)?;
    println!(
        "KAMA kernel: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&kama)
    );

    let t = Instant::now();
    let obv = obv_gpu(&device, &close_arr, &volume_arr, None)?;
    let obv_ms = t.elapsed().as_secs_f64() * 1000.0;
    let obv_ref = obv_cpu(&close, &volume);
    let obv_diff = max_abs_diff_f64(obv.as_slice().unwrap_or(&[]), &obv_ref);
    println!(
        "OBV kernels: OK | time={:.2} ms | max_abs_diff_vs_cpu={:.6}",
        obv_ms, obv_diff
    );

    let high_arr = Array1::from_vec(high.clone());
    let low_arr = Array1::from_vec(low.clone());

    let t = Instant::now();
    let adx = adx_gpu(&device, &high_arr, &low_arr, &close_arr, 14, None)?;
    println!(
        "ADX kernels: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&adx)
    );

    let t = Instant::now();
    let mfi = mfi_gpu(
        &device,
        &high_arr,
        &low_arr,
        &close_arr,
        &volume_arr,
        14,
        None,
    )?;
    println!(
        "MFI kernels: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&mfi)
    );

    let t = Instant::now();
    let avwap = vwap_anchored_gpu(
        &device,
        &high_arr,
        &low_arr,
        &close_arr,
        &volume_arr,
        0,
        None,
    )?;
    println!(
        "Anchored VWAP kernels: OK | time={:.2} ms | finite={}",
        t.elapsed().as_secs_f64() * 1000.0,
        finite_count_f64(&avwap)
    );

    let volume_f32: Vec<f32> = volume.iter().map(|&x| x as f32).collect();
    let t = Instant::now();
    let d_scan_in = device.copy_to_device_f32(&volume_f32)?;
    let mut d_scan_out = device.allocate_device_buffer::<f32>(volume_f32.len())?;
    inclusive_scan_f32(&device, None, &d_scan_in, &mut d_scan_out, ScanOp::Sum)?;
    let scan_out = device.copy_to_host_f32(&d_scan_out)?;
    let scan_ms = t.elapsed().as_secs_f64() * 1000.0;
    let scan_ref = scan_sum_cpu_f32(&volume_f32);
    let scan_diff = max_abs_diff_f32(&scan_out, &scan_ref);
    println!(
        "scan.cu sum kernels: OK | time={:.2} ms | max_abs_diff_vs_cpu={:.6}",
        scan_ms, scan_diff
    );

    let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
    let t = Instant::now();
    let d_w_in = device.copy_to_device_f32(&close_f32)?;
    let mut d_w_out = device.allocate_device_buffer::<f32>(close_f32.len())?;
    wilder_smooth_f32(&device, None, &d_w_in, 14, &mut d_w_out)?;
    let w_out = device.copy_to_host_f32(&d_w_out)?;
    let w_ms = t.elapsed().as_secs_f64() * 1000.0;
    let w_ref = wilder_cpu_f32(&close_f32, 14);
    let w_diff = max_abs_diff_f32(&w_out, &w_ref);
    println!(
        "scan.cu affine (Wilder) kernels: OK | time={:.2} ms | max_abs_diff_vs_cpu={:.6}",
        w_ms, w_diff
    );

    println!("\nValidation complete.");
    println!(
        "Note: tick/trade aggregation kernels need tick-level trade data; this dataset is OHLCV candles."
    );

    Ok(())
}
