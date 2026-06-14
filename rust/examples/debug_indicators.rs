use kimsfinance_core::backtest::{BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, Signal, Strategy};
use ndarray::Array1;

struct DebugStrategy { count: usize, signals: usize }

impl Strategy for DebugStrategy {
    fn on_data(&mut self, bar: &OHLCVBar, ind: &IndicatorValues) -> Signal {
        self.count += 1;
        if self.count <= 35 || self.count % 5000 == 0 {
            let keys: Vec<_> = ind.keys().collect();
            let mv = ind.get("macd_8_17_9_macd").copied().unwrap_or(f64::NAN);
            let sv = ind.get("macd_8_17_9_signal").copied().unwrap_or(f64::NAN);
            let rv = ind.get("rsi_14").copied().unwrap_or(f64::NAN);
            if self.count <= 35 {
                println!("  Bar {:>5}: MACD={:>8.4}  Signal={:>8.4}  RSI={:>6.2}  keys={}",
                         self.count, mv, sv, rv, keys.len());
            }
        }
        Signal::Hold
    }
    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::MACD { fast: 8, slow: 17, signal: 9 },
            IndicatorConfig::RSI  { period: 14 },
        ]
    }
    fn initial_capital(&self) -> f64 { 10_000.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate 200 synthetic prices: 100 + random walk
    let n = 200;
    let mut close = Vec::with_capacity(n);
    let mut p = 100.0f64;
    for i in 0..n {
        p += (i as f64 * 0.1).sin() * 2.0;
        close.push(p);
    }
    let ts: Vec<i64>  = (0..n as i64).map(|i| i * 300_000).collect();
    let open_arr   = Array1::from_vec(close.clone());
    let high_arr   = Array1::from_vec(close.iter().map(|x| x + 0.5).collect());
    let low_arr    = Array1::from_vec(close.iter().map(|x| x - 0.5).collect());
    let close_arr  = Array1::from_vec(close);
    let volume_arr = Array1::ones(n);

    let cfg = BacktestConfig { initial_capital: 10_000.0, trading_fee: 0.0, slippage: 0.0,
                               execution_latency_ms: 0, use_gpu: false, force_cpu: true };
    let engine = BacktestEngine::with_config(cfg);
    let mut strat = DebugStrategy { count: 0, signals: 0 };

    println!("Running debug indicator check on {} bars...", n);
    println!("First 35 bars (expecting NaN until bar ~27):\n");
    engine.run(&mut strat, &ts, &open_arr, &high_arr, &low_arr, &close_arr, &volume_arr)?;
    println!("\nProcessed {} bars total", strat.count);
    Ok(())
}
