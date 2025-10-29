# Heston Calibrator Implementation Plan

**Target**: GPU-accelerated Heston model calibration for options pricing across crypto and equity markets

**Data Sources**:
- **Crypto Options**: Deribit, Binance Options
- **Equity Options**: Interactive Brokers (IBKR) API
- **Futures**: Binance (existing integration)

**Performance Target**: 100-500x faster than CPU for calibration (50-200ms per calibration)

---

## Phase 1: Core Infrastructure (Week 1-2)

### 1.1 Heston Model Core

**File**: `src/quantitative/heston/model.rs`

```rust
/// Heston stochastic volatility model
///
/// dS_t = μS_t dt + √v_t S_t dW_t^S  (asset price)
/// dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v  (variance process)
/// Corr(dW_t^S, dW_t^v) = ρ dt
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HestonParams {
    /// Mean reversion speed (typical: 0.5 - 5.0)
    pub kappa: f64,

    /// Long-term variance (typical: 0.01 - 0.1)
    pub theta: f64,

    /// Volatility of volatility (typical: 0.1 - 1.0)
    pub sigma: f64,

    /// Correlation between asset and variance (-1.0 to +1.0)
    /// Negative = leverage effect (vol increases when price drops)
    pub rho: f64,

    /// Initial variance (current market vol²)
    pub v0: f64,
}

impl HestonParams {
    /// Validate parameters satisfy Feller condition: 2κθ > σ²
    /// Ensures variance stays positive
    pub fn validate(&self) -> Result<(), ValidationError> {
        if 2.0 * self.kappa * self.theta <= self.sigma.powi(2) {
            return Err(ValidationError::FellerCondition);
        }
        // Additional checks...
        Ok(())
    }

    /// Forecast variance at time t
    pub fn forecast_variance(&self, t: f64) -> f64 {
        // E[v_t] = v₀e^(-κt) + θ(1 - e^(-κt))
        let exp_term = (-self.kappa * t).exp();
        self.v0 * exp_term + self.theta * (1.0 - exp_term)
    }

    /// Long-term volatility (annualized)
    pub fn long_term_vol(&self) -> f64 {
        self.theta.sqrt()
    }
}
```

### 1.2 Market Data Types

**File**: `src/quantitative/options/types.rs`

```rust
/// Option quote from market
#[derive(Debug, Clone)]
pub struct OptionQuote {
    pub symbol: String,
    pub underlying: String,
    pub strike: f64,
    pub expiry: DateTime<Utc>,
    pub option_type: OptionType,
    pub bid: f64,
    pub ask: f64,
    pub mid_price: f64,
    pub implied_vol: Option<f64>,  // If available from exchange
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub volume: f64,
    pub open_interest: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum OptionType {
    Call,
    Put,
}

/// Option chain for calibration
#[derive(Debug, Clone)]
pub struct OptionChain {
    pub underlying_price: f64,
    pub timestamp: DateTime<Utc>,
    pub options: Vec<OptionQuote>,
    pub risk_free_rate: f64,  // From IBKR or yield curve
    pub dividend_yield: f64,   // For stocks
}

impl OptionChain {
    /// Filter to liquid options for calibration
    /// Use ATM ± 2 strikes, volume > threshold
    pub fn filter_liquid(&self, min_volume: f64) -> Self {
        // Keep options with sufficient liquidity
        // Focus on ATM ± 20% strikes
    }
}
```

---

## Phase 2: GPU Characteristic Function (Week 2-3)

### 2.1 CUDA Kernel

**File**: `src/gpu/heston/characteristic_function.cu`

```cuda
/// Heston characteristic function for option pricing
///
/// φ(u; τ, v₀) = exp(C(u,τ) + D(u,τ)v₀ + iu·ln(S₀))
///
/// Used for fast Fourier transform (FFT) option pricing
extern "C" __global__ void heston_characteristic_function(
    const double* u,           // Frequency points [n_points]
    double* phi_real,          // Output: Re[φ(u)] [n_points]
    double* phi_imag,          // Output: Im[φ(u)] [n_points]
    double S0,                 // Spot price
    double K,                  // Strike price
    double r,                  // Risk-free rate
    double q,                  // Dividend yield
    double tau,                // Time to maturity (years)
    double v0,                 // Initial variance
    double kappa,              // Mean reversion speed
    double theta,              // Long-term variance
    double sigma,              // Vol of vol
    double rho,                // Correlation
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    double u_val = u[idx];

    // Compute auxiliary variables
    double xi = kappa - rho * sigma * u_val * I;  // I = sqrt(-1)
    double d = sqrt(xi * xi + sigma * sigma * (u_val * u_val + u_val * I));
    double g = (xi - d) / (xi + d);

    // Time-dependent coefficients
    double exp_d_tau = exp(-d * tau);
    double D_real = ((xi - d) / (sigma * sigma))
                    * ((1.0 - exp_d_tau) / (1.0 - g * exp_d_tau));

    double C_real = r * u_val * I * tau
                    + (kappa * theta / (sigma * sigma))
                    * ((xi - d) * tau - 2.0 * log((1.0 - g * exp_d_tau) / (1.0 - g)));

    // Characteristic function: φ = exp(C + D·v₀ + iu·ln(S₀))
    double exponent_real = C_real + D_real * v0;
    double exponent_imag = u_val * log(S0);

    // exp(a + ib) = exp(a)(cos(b) + i·sin(b))
    double exp_real = exp(exponent_real);
    phi_real[idx] = exp_real * cos(exponent_imag);
    phi_imag[idx] = exp_real * sin(exponent_imag);
}
```

### 2.2 Rust Wrapper

**File**: `src/gpu/heston/pricing.rs`

```rust
use super::GpuDevice;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

pub struct HestonGpuPricer {
    device: GpuDevice,
    cf_kernel: cudarc::driver::CudaFunction,
}

impl HestonGpuPricer {
    pub fn new() -> Result<Self, GpuError> {
        let device = GpuDevice::new()?;
        let ptx = compile_heston_kernels()?;
        let module = device.context().load_module(ptx)?;
        let cf_kernel = module.load_function("heston_characteristic_function")?;

        Ok(Self { device, cf_kernel })
    }

    /// Price European option using FFT + characteristic function
    pub fn price_european(
        &self,
        params: &HestonParams,
        spot: f64,
        strike: f64,
        expiry: f64,  // Years
        risk_free_rate: f64,
        option_type: OptionType,
    ) -> Result<f64, PricingError> {
        // 1. Set up frequency grid for FFT
        let n_points = 4096;  // Power of 2 for FFT
        let du = 0.01;
        let u: Vec<f64> = (0..n_points).map(|i| i as f64 * du).collect();

        // 2. Compute characteristic function on GPU
        let d_u = self.device.copy_to_device(&u)?;
        let mut d_phi_real = self.device.alloc_buffer(n_points)?;
        let mut d_phi_imag = self.device.alloc_buffer(n_points)?;

        let config = LaunchConfig::for_num_elems(n_points as u32);
        let mut builder = self.device.stream.launch_builder(&self.cf_kernel);

        builder.arg(&d_u);
        builder.arg(&mut d_phi_real);
        builder.arg(&mut d_phi_imag);
        builder.arg(&spot);
        builder.arg(&strike);
        builder.arg(&risk_free_rate);
        builder.arg(&0.0);  // Dividend yield
        builder.arg(&expiry);
        builder.arg(&params.v0);
        builder.arg(&params.kappa);
        builder.arg(&params.theta);
        builder.arg(&params.sigma);
        builder.arg(&params.rho);
        builder.arg(&(n_points as i32));

        unsafe {
            builder.launch(config)?;
        }

        // 3. Inverse FFT to get option price
        let phi_real = self.device.copy_to_host(&d_phi_real)?;
        let phi_imag = self.device.copy_to_host(&d_phi_imag)?;

        // 4. Apply Carr-Madan formula
        let price = self.carr_madan_integration(&phi_real, &phi_imag, strike, du)?;

        Ok(price)
    }
}
```

---

## Phase 3: Calibration Engine (Week 3-4)

### 3.1 Optimization Target

**File**: `src/quantitative/heston/calibration.rs`

```rust
/// Calibration objective: minimize pricing error
pub struct CalibrationObjective {
    pub option_chain: OptionChain,
    pub pricer: HestonGpuPricer,
}

impl CalibrationObjective {
    /// Compute sum of squared pricing errors (MSE)
    pub fn objective_function(&self, params: &HestonParams) -> f64 {
        let mut total_error = 0.0;

        for option in &self.option_chain.options {
            // Model price
            let model_price = self.pricer.price_european(
                params,
                self.option_chain.underlying_price,
                option.strike,
                time_to_expiry(option.expiry),
                self.option_chain.risk_free_rate,
                option.option_type,
            ).unwrap_or(0.0);

            // Market price (bid-ask mid)
            let market_price = option.mid_price;

            // Weighted error (weight by vega or volume)
            let weight = option.volume.sqrt();  // More liquid = more weight
            let error = weight * (model_price - market_price).powi(2);
            total_error += error;
        }

        total_error / self.option_chain.options.len() as f64
    }

    /// Compute gradient for optimization (finite differences)
    pub fn gradient(&self, params: &HestonParams) -> [f64; 5] {
        let eps = 1e-4;
        let base_obj = self.objective_function(params);

        let mut grad = [0.0; 5];

        // ∂f/∂κ
        let mut p = *params;
        p.kappa += eps;
        grad[0] = (self.objective_function(&p) - base_obj) / eps;

        // ... repeat for θ, σ, ρ, v₀

        grad
    }
}
```

### 3.2 GPU-Accelerated Optimizer

```rust
/// L-BFGS-B optimizer with GPU-accelerated objective
pub struct HestonCalibrator {
    pricer: HestonGpuPricer,
    max_iterations: usize,
    tolerance: f64,
}

impl HestonCalibrator {
    /// Calibrate Heston parameters to option chain
    pub fn calibrate(
        &self,
        option_chain: &OptionChain,
        initial_guess: Option<HestonParams>,
    ) -> Result<CalibratedHeston, CalibrationError> {
        // 1. Set up objective
        let objective = CalibrationObjective {
            option_chain: option_chain.clone(),
            pricer: self.pricer.clone(),
        };

        // 2. Initial guess (ATM vol if not provided)
        let x0 = initial_guess.unwrap_or_else(|| {
            HestonParams::from_atm_vol(option_chain.atm_implied_vol())
        });

        // 3. Run L-BFGS-B with box constraints
        let bounds = HestonBounds::default();
        let optimizer = LBfgsB::new()
            .with_max_iterations(self.max_iterations)
            .with_tolerance(self.tolerance);

        let result = optimizer.minimize(
            |params| objective.objective_function(params),
            |params| objective.gradient(params),
            x0,
            bounds,
        )?;

        // 4. Validate and return
        result.params.validate()?;

        Ok(CalibratedHeston {
            params: result.params,
            rmse: result.final_objective.sqrt(),
            iterations: result.iterations,
            calibration_time: result.elapsed,
        })
    }
}
```

---

## Phase 4: Data Integration (Week 4-5)

### 4.1 IBKR Integration

**File**: `src/data_sources/ibkr/options.rs`

```rust
use ibapi::Client as IbkrClient;

pub struct IbkrOptionsProvider {
    client: IbkrClient,
}

impl IbkrOptionsProvider {
    /// Fetch option chain for symbol
    pub async fn fetch_option_chain(
        &self,
        symbol: &str,
        expiration: Option<DateTime<Utc>>,
    ) -> Result<OptionChain, DataError> {
        // 1. Request contract details
        let contract = self.client.req_contract_details(symbol).await?;

        // 2. Get underlying price
        let underlying_price = self.client.req_mkt_data(&contract).await?.last_price;

        // 3. Fetch option chain
        let options = self.client.req_sec_def_opt_params(symbol).await?;

        // 4. Get quotes for each option
        let mut quotes = Vec::new();
        for option in options.iter() {
            let quote = self.client.req_mkt_data(&option.contract).await?;
            quotes.push(OptionQuote::from_ibkr(quote));
        }

        // 5. Get risk-free rate from short-term treasuries
        let risk_free_rate = self.fetch_treasury_yield("^IRX").await?;

        Ok(OptionChain {
            underlying_price,
            timestamp: Utc::now(),
            options: quotes,
            risk_free_rate,
            dividend_yield: contract.div_yield,
        })
    }
}
```

### 4.2 Deribit Integration

**File**: `src/data_sources/deribit/options.rs`

```rust
pub struct DeribitOptionsProvider {
    client: reqwest::Client,
    base_url: String,
}

impl DeribitOptionsProvider {
    /// Fetch BTC or ETH option chain
    pub async fn fetch_option_chain(
        &self,
        currency: &str,  // "BTC" or "ETH"
    ) -> Result<OptionChain, DataError> {
        // 1. Get instruments
        let url = format!("{}/public/get_instruments", self.base_url);
        let params = [
            ("currency", currency),
            ("kind", "option"),
            ("expired", "false"),
        ];

        let instruments: Vec<DeribitInstrument> = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

        // 2. Get orderbook for each option
        let mut quotes = Vec::new();
        for inst in instruments {
            let book = self.get_orderbook(&inst.instrument_name).await?;
            quotes.push(OptionQuote::from_deribit(inst, book));
        }

        // 3. Get index price (underlying)
        let index = self.get_index_price(currency).await?;

        Ok(OptionChain {
            underlying_price: index,
            timestamp: Utc::now(),
            options: quotes,
            risk_free_rate: 0.0,  // Crypto has no risk-free rate
            dividend_yield: 0.0,
        })
    }
}
```

---

## Phase 5: Trading Strategies (Week 5-6)

### 5.1 Vol Arbitrage

```rust
/// Identify mispriced options using calibrated Heston
pub struct VolatilityArbitrage {
    calibrator: HestonCalibrator,
    pricer: HestonGpuPricer,
}

impl VolatilityArbitrage {
    /// Find mispriced options
    pub fn find_opportunities(
        &self,
        option_chain: &OptionChain,
    ) -> Vec<ArbitrageOpportunity> {
        // 1. Calibrate Heston to entire chain
        let calibrated = self.calibrator.calibrate(option_chain, None).unwrap();

        // 2. Price each option with calibrated params
        let mut opportunities = Vec::new();

        for option in &option_chain.options {
            let model_price = self.pricer.price_european(
                &calibrated.params,
                option_chain.underlying_price,
                option.strike,
                time_to_expiry(option.expiry),
                option_chain.risk_free_rate,
                option.option_type,
            ).unwrap();

            let market_price = option.mid_price;
            let mispricing = (model_price - market_price) / model_price;

            // Flag if mispricing > 5%
            if mispricing.abs() > 0.05 {
                opportunities.push(ArbitrageOpportunity {
                    option: option.clone(),
                    model_price,
                    market_price,
                    mispricing_pct: mispricing * 100.0,
                    signal: if mispricing > 0.0 {
                        Signal::Buy  // Model thinks it's cheap
                    } else {
                        Signal::Sell // Model thinks it's expensive
                    },
                });
            }
        }

        opportunities.sort_by(|a, b| {
            b.mispricing_pct.abs().partial_cmp(&a.mispricing_pct.abs()).unwrap()
        });

        opportunities
    }
}
```

### 5.2 Adaptive Futures Strategy

```rust
/// Use Heston vol regime detection for futures trading
pub struct AdaptiveFuturesStrategy {
    heston_calibrator: HestonCalibrator,
    mean_reversion_strategy: Box<dyn TradingStrategy>,
    momentum_strategy: Box<dyn TradingStrategy>,
    breakout_strategy: Box<dyn TradingStrategy>,
}

impl AdaptiveFuturesStrategy {
    /// Select strategy based on vol regime
    pub fn select_strategy(
        &mut self,
        futures_data: &[Candle],
    ) -> &dyn TradingStrategy {
        // 1. Calibrate Heston from realized vol
        let returns = compute_returns(futures_data);
        let realized_vol = realized_volatility(&returns);

        // Simplified calibration from realized vol time series
        let params = self.heston_calibrator
            .calibrate_from_realized(&realized_vol)
            .unwrap();

        // 2. Classify regime
        match self.classify_regime(&params) {
            VolRegime::HighVolOfVol => &*self.mean_reversion_strategy,
            VolRegime::FastMeanReversion => &*self.breakout_strategy,
            VolRegime::SlowMeanReversion => &*self.momentum_strategy,
        }
    }

    fn classify_regime(&self, params: &HestonParams) -> VolRegime {
        // High σ (vol-of-vol) → Volatile, use mean-reversion
        if params.sigma > 0.5 {
            return VolRegime::HighVolOfVol;
        }

        // High κ (mean reversion) → Ranging, use breakout
        if params.kappa > 2.0 {
            return VolRegime::FastMeanReversion;
        }

        // Low κ → Trending, use momentum
        VolRegime::SlowMeanReversion
    }
}
```

---

## Performance Targets

| Operation | CPU Baseline | GPU Target | Speedup |
|-----------|--------------|------------|---------|
| **Characteristic Function** (4096 pts) | 15ms | 0.1ms | 150x |
| **Single Option Price** | 20ms | 0.2ms | 100x |
| **Calibration** (50 options, 100 iter) | 30s | 50-100ms | 300-600x |
| **Greeks Calculation** (100 options) | 2s | 10ms | 200x |

---

## Testing & Validation

### Unit Tests
- Heston model equations vs analytical solutions
- Characteristic function vs Black-Scholes limit
- Feller condition validation

### Integration Tests
- IBKR data fetch and parsing
- Deribit data fetch and parsing
- End-to-end calibration pipeline

### Validation
- Compare vs QuantLib (C++ reference)
- Validate Greeks vs finite differences
- Test on known market data (2020 vol spike, etc.)

---

## Deliverables

1. **Core Library** (`src/quantitative/heston/`)
   - Model, calibration, pricing, Greeks

2. **GPU Kernels** (`src/gpu/heston/`)
   - Characteristic function, FFT, optimization

3. **Data Connectors** (`src/data_sources/`)
   - IBKR, Deribit, Binance options

4. **Trading Strategies** (`src/strategies/`)
   - Vol arbitrage, regime detection, adaptive futures

5. **Documentation**
   - API reference, user guide, examples

6. **Benchmarks**
   - Performance comparison vs CPU, QuantLib

---

## Dependencies

```toml
[dependencies]
# Existing
cudarc = "=0.17.3"
ndarray = "0.16"
chrono = "0.4"

# New for options
ibapi = "1.0"              # Interactive Brokers API
reqwest = { version = "0.11", features = ["json"] }
rustfft = "6.0"            # FFT for option pricing
argmin = "0.9"             # L-BFGS-B optimizer
statrs = "0.16"            # Statistical distributions
```

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1** | 1-2 weeks | Core Heston model, types, validation |
| **Phase 2** | 1-2 weeks | GPU characteristic function, FFT pricing |
| **Phase 3** | 1 week | Calibration engine, optimizer |
| **Phase 4** | 1-2 weeks | IBKR + Deribit data integration |
| **Phase 5** | 1-2 weeks | Trading strategies, Greeks |
| **Testing** | 1 week | Validation, benchmarks, docs |
| **Total** | **6-8 weeks** | Production-ready Heston system |

---

**Status**: Ready to implement
**Next Step**: Begin Phase 1 - Core Infrastructure
