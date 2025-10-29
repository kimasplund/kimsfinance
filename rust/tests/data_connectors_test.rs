//! Integration tests for options data connectors
//!
//! These tests require external services to be available:
//! - Deribit: Public API (no authentication required)
//! - IBKR: TWS/Gateway running locally with paper trading account
//!
//! Run tests with specific features:
//! ```bash
//! cargo test --features data-deribit -- deribit
//! cargo test --features data-ibkr -- ibkr
//! cargo test --features data-all
//! ```

#[cfg(feature = "data-deribit")]
mod deribit_tests {
    use kimsfinance_core::data::OptionsDataProvider;
    use kimsfinance_core::data::deribit::DeribitConnector;

    #[tokio::test]
    async fn test_deribit_connection() {
        let result = DeribitConnector::connect().await;
        assert!(
            result.is_ok(),
            "Failed to connect to Deribit: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_deribit_btc_options_chain() {
        let connector = DeribitConnector::connect()
            .await
            .expect("Connection failed");
        let options = connector
            .fetch_options_chain("BTC")
            .await
            .expect("Failed to fetch BTC options");

        assert!(!options.is_empty(), "No BTC options found");

        // Validate first option
        let first = &options[0];
        assert_eq!(first.underlying, "BTC");
        assert!(first.strike > 0.0);
        assert!(first.expiration > 0);
        assert!(first.spot_price > 0.0);

        // Check implied volatility
        assert!(
            first.implied_vol.is_some(),
            "Implied volatility should be present"
        );
        let iv = first.implied_vol.unwrap();
        assert!(iv > 0.0 && iv < 5.0, "IV should be reasonable: {}", iv);

        println!("✅ Found {} BTC options", options.len());
        println!(
            "Sample: Strike {} {} @ IV {:.2}%",
            first.strike,
            match first.option_type {
                kimsfinance_core::quantitative::heston::OptionType::Call => "Call",
                kimsfinance_core::quantitative::heston::OptionType::Put => "Put",
            },
            iv * 100.0
        );
    }

    #[tokio::test]
    async fn test_deribit_eth_options_chain() {
        let connector = DeribitConnector::connect()
            .await
            .expect("Connection failed");
        let options = connector
            .fetch_options_chain("ETH")
            .await
            .expect("Failed to fetch ETH options");

        assert!(!options.is_empty(), "No ETH options found");
        println!("✅ Found {} ETH options", options.len());
    }

    #[tokio::test]
    async fn test_deribit_greeks() {
        let connector = DeribitConnector::connect()
            .await
            .expect("Connection failed");
        let options = connector
            .fetch_options_chain("BTC")
            .await
            .expect("Failed to fetch options");

        let option_with_greeks = options
            .iter()
            .find(|opt| opt.greeks.is_some())
            .expect("No options with Greeks found");

        let greeks = option_with_greeks.greeks.as_ref().unwrap();
        assert!(greeks.delta.is_some(), "Delta should be present in Greeks");
        assert!(greeks.gamma.is_some(), "Gamma should be present in Greeks");
        assert!(greeks.vega.is_some(), "Vega should be present in Greeks");
        assert!(greeks.theta.is_some(), "Theta should be present in Greeks");

        println!("✅ Greeks validation passed");
        println!("  Delta: {:?}", greeks.delta);
        println!("  Gamma: {:?}", greeks.gamma);
        println!("  Vega: {:?}", greeks.vega);
        println!("  Theta: {:?}", greeks.theta);
    }

    #[tokio::test]
    async fn test_deribit_historical_volatility() {
        let connector = DeribitConnector::connect()
            .await
            .expect("Connection failed");
        let hist_vol = connector
            .fetch_historical_volatility("BTC", 30)
            .await
            .expect("Failed to fetch historical volatility");

        assert!(!hist_vol.is_empty(), "No historical volatility data found");

        // Validate data format
        for (timestamp, vol) in hist_vol.iter().take(5) {
            assert!(*timestamp > 0, "Invalid timestamp");
            assert!(*vol > 0.0 && *vol < 10.0, "Invalid volatility: {}", vol);
        }

        println!(
            "✅ Found {} historical volatility data points",
            hist_vol.len()
        );
        println!("  Latest: {:?}", hist_vol.last());
    }

    #[tokio::test]
    async fn test_deribit_data_quality() {
        let connector = DeribitConnector::connect()
            .await
            .expect("Connection failed");
        let options = connector
            .fetch_options_chain("BTC")
            .await
            .expect("Failed to fetch options");

        let mut liquid_options = 0;
        let mut with_tight_spreads = 0;

        for option in options.iter() {
            // Check liquidity (open interest > 0)
            if option.open_interest > 0.0 {
                liquid_options += 1;
            }

            // Check bid-ask spread
            if let (Some(bid), Some(ask)) = (option.bid, option.ask) {
                let mid = (bid + ask) / 2.0;
                let spread_pct = if mid > 0.0 {
                    ((ask - bid) / mid) * 100.0
                } else {
                    0.0
                };

                if spread_pct < 10.0 {
                    with_tight_spreads += 1;
                }
            }
        }

        println!("✅ Data quality metrics:");
        println!(
            "  Liquid options (OI > 0): {}/{}",
            liquid_options,
            options.len()
        );
        println!(
            "  Tight spreads (<10%): {}/{}",
            with_tight_spreads,
            options.len()
        );

        assert!(liquid_options > 0, "Expected at least some liquid options");
    }
}

#[cfg(feature = "data-ibkr")]
mod ibkr_tests {
    use kimsfinance_core::data::OptionsDataProvider;
    use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrConnector};

    #[tokio::test]
    #[ignore] // Requires TWS/Gateway running
    async fn test_ibkr_connection() {
        let config = IbkrConfig::default(); // Paper trading
        let result = IbkrConnector::connect(config).await;

        assert!(
            result.is_ok(),
            "Failed to connect to IBKR. Is TWS/Gateway running? Error: {:?}",
            result.err()
        );

        println!("✅ Connected to IBKR TWS/Gateway");
    }

    #[tokio::test]
    #[ignore] // Requires TWS/Gateway running
    async fn test_ibkr_aapl_options_chain() {
        let config = IbkrConfig::default();
        let connector = IbkrConnector::connect(config)
            .await
            .expect("Connection failed");

        let options = connector
            .fetch_options_chain("AAPL")
            .await
            .expect("Failed to fetch AAPL options");

        assert!(!options.is_empty(), "No AAPL options found");

        // Validate data
        let first = &options[0];
        assert_eq!(first.underlying, "AAPL");
        assert!(first.strike > 0.0);
        assert!(first.expiration > 0);

        println!("✅ Found {} AAPL options", options.len());
        println!(
            "Sample: Strike {} {} @ IV {:?}",
            first.strike,
            match first.option_type {
                kimsfinance_core::quantitative::heston::OptionType::Call => "Call",
                kimsfinance_core::quantitative::heston::OptionType::Put => "Put",
            },
            first.implied_vol
        );
    }

    #[tokio::test]
    #[ignore] // Requires TWS/Gateway running
    async fn test_ibkr_spx_options_chain() {
        let config = IbkrConfig::default();
        let connector = IbkrConnector::connect(config)
            .await
            .expect("Connection failed");

        let options = connector
            .fetch_options_chain("SPX")
            .await
            .expect("Failed to fetch SPX options");

        assert!(!options.is_empty(), "No SPX options found");
        println!("✅ Found {} SPX options", options.len());
    }
}

#[cfg(all(feature = "data-ibkr", feature = "data-deribit"))]
mod unified_tests {
    use kimsfinance_core::data::OptionsDataProvider;
    use kimsfinance_core::data::deribit::DeribitConnector;
    use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrConnector};

    #[tokio::test]
    async fn test_unified_trait_deribit() {
        // Test that DeribitConnector implements OptionsDataProvider
        let connector: Box<dyn OptionsDataProvider> = Box::new(
            DeribitConnector::connect()
                .await
                .expect("Connection failed"),
        );

        let options = connector
            .fetch_options_chain("BTC")
            .await
            .expect("Failed to fetch options");

        assert!(!options.is_empty());
        println!("✅ Deribit via trait object: {} options", options.len());
    }

    #[tokio::test]
    #[ignore] // Requires TWS/Gateway
    async fn test_unified_trait_ibkr() {
        // Test that IbkrConnector implements OptionsDataProvider
        let connector: Box<dyn OptionsDataProvider> = Box::new(
            IbkrConnector::connect(IbkrConfig::default())
                .await
                .expect("Connection failed"),
        );

        let options = connector
            .fetch_options_chain("AAPL")
            .await
            .expect("Failed to fetch options");

        assert!(!options.is_empty());
        println!("✅ IBKR via trait object: {} options", options.len());
    }
}
