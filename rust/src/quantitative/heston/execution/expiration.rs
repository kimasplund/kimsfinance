//! Expiration Handler
//!
//! Handles option expiration events including auto-exercise and assignment.

use super::position_manager::OptionPosition;
use crate::quantitative::heston::OptionType;

/// Expiration event types
#[derive(Debug, Clone)]
pub enum ExpirationEvent {
    /// Option auto-exercised (ITM position)
    AutoExercise {
        position_id: String,
        intrinsic_value: f64,
        settlement_amount: f64,
    },

    /// Option expired worthless (OTM position)
    Expire {
        position_id: String,
        option_type: OptionType,
        strike: f64,
    },

    /// Short option assigned (counterparty exercised)
    Assignment {
        position_id: String,
        strike: f64,
        settlement_amount: f64,
    },
}

impl ExpirationEvent {
    /// Get position ID for this event
    pub fn position_id(&self) -> &str {
        match self {
            ExpirationEvent::AutoExercise { position_id, .. } => position_id,
            ExpirationEvent::Expire { position_id, .. } => position_id,
            ExpirationEvent::Assignment { position_id, .. } => position_id,
        }
    }

    /// Get settlement amount (0 for expired positions)
    pub fn settlement_amount(&self) -> f64 {
        match self {
            ExpirationEvent::AutoExercise {
                settlement_amount, ..
            } => *settlement_amount,
            ExpirationEvent::Expire { .. } => 0.0,
            ExpirationEvent::Assignment {
                settlement_amount, ..
            } => *settlement_amount,
        }
    }

    /// Check if this is a profitable event
    pub fn is_profitable(&self) -> bool {
        self.settlement_amount() > 0.0
    }
}

/// Expiration Handler
///
/// Processes option expirations and determines settlement amounts.
pub struct ExpirationHandler;

impl ExpirationHandler {
    /// Check and process expirations for all positions
    ///
    /// # Arguments
    ///
    /// * `positions` - Slice of active positions
    /// * `current_time` - Current timestamp
    /// * `underlying_price` - Current underlying price
    ///
    /// # Returns
    ///
    /// Vector of expiration events for positions that expired
    pub fn check_expirations(
        positions: &[OptionPosition],
        current_time: i64,
        underlying_price: f64,
    ) -> Vec<ExpirationEvent> {
        positions
            .iter()
            .filter(|pos| pos.expiration <= current_time)
            .map(|pos| Self::process_expiration(pos, underlying_price))
            .collect()
    }

    /// Process single position expiration
    ///
    /// Determines whether position should be exercised or expires worthless.
    ///
    /// # Arguments
    ///
    /// * `position` - Position that has expired
    /// * `underlying_price` - Current underlying price
    ///
    /// # Returns
    ///
    /// Appropriate expiration event
    pub fn process_expiration(
        position: &OptionPosition,
        underlying_price: f64,
    ) -> ExpirationEvent {
        let intrinsic = Self::calculate_intrinsic_value(
            position.option_type,
            position.strike,
            underlying_price,
        );

        if intrinsic > 0.0 {
            // Option is ITM
            let settlement = Self::calculate_settlement(position, underlying_price);

            if position.is_long() {
                // Long position: auto-exercise
                ExpirationEvent::AutoExercise {
                    position_id: position.position_id.clone(),
                    intrinsic_value: intrinsic,
                    settlement_amount: settlement,
                }
            } else {
                // Short position: assignment
                ExpirationEvent::Assignment {
                    position_id: position.position_id.clone(),
                    strike: position.strike,
                    settlement_amount: settlement,
                }
            }
        } else {
            // Option is OTM: expires worthless
            ExpirationEvent::Expire {
                position_id: position.position_id.clone(),
                option_type: position.option_type,
                strike: position.strike,
            }
        }
    }

    /// Calculate intrinsic value of option
    ///
    /// # Arguments
    ///
    /// * `option_type` - Call or Put
    /// * `strike` - Strike price
    /// * `underlying_price` - Current underlying price
    ///
    /// # Returns
    ///
    /// Intrinsic value per share (always >= 0)
    pub fn calculate_intrinsic_value(
        option_type: OptionType,
        strike: f64,
        underlying_price: f64,
    ) -> f64 {
        match option_type {
            OptionType::Call => (underlying_price - strike).max(0.0),
            OptionType::Put => (strike - underlying_price).max(0.0),
        }
    }

    /// Calculate settlement amount for position
    ///
    /// Takes into account position size (quantity) and contract multiplier (100).
    ///
    /// # Arguments
    ///
    /// * `position` - Expiring position
    /// * `underlying_price` - Current underlying price
    ///
    /// # Returns
    ///
    /// Total settlement amount (positive = cash in, negative = cash out)
    pub fn calculate_settlement(position: &OptionPosition, underlying_price: f64) -> f64 {
        let intrinsic = Self::calculate_intrinsic_value(
            position.option_type,
            position.strike,
            underlying_price,
        );

        // Long positions receive intrinsic value
        // Short positions pay intrinsic value
        let multiplier = 100.0; // Standard option contract multiplier
        intrinsic * (position.quantity as f64) * multiplier
    }

    /// Check if option is in-the-money
    pub fn is_itm(option_type: OptionType, strike: f64, underlying_price: f64) -> bool {
        match option_type {
            OptionType::Call => underlying_price > strike,
            OptionType::Put => underlying_price < strike,
        }
    }

    /// Check if option is at-the-money (within threshold)
    pub fn is_atm(strike: f64, underlying_price: f64, threshold: f64) -> bool {
        (underlying_price - strike).abs() / strike <= threshold
    }

    /// Check if option is out-of-the-money
    pub fn is_otm(option_type: OptionType, strike: f64, underlying_price: f64) -> bool {
        !Self::is_itm(option_type, strike, underlying_price)
    }

    /// Calculate moneyness ratio (S/K for calls, K/S for puts)
    pub fn moneyness(option_type: OptionType, strike: f64, underlying_price: f64) -> f64 {
        match option_type {
            OptionType::Call => underlying_price / strike,
            OptionType::Put => strike / underlying_price,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_position(
        option_type: OptionType,
        strike: f64,
        quantity: i32,
        expiration: i64,
    ) -> OptionPosition {
        OptionPosition::new(
            "test_pos".to_string(),
            option_type,
            strike,
            expiration,
            quantity,
            5.0,
            1735000000,
        )
    }

    #[test]
    fn test_intrinsic_value_call() {
        // ITM call
        let intrinsic =
            ExpirationHandler::calculate_intrinsic_value(OptionType::Call, 100.0, 110.0);
        assert_eq!(intrinsic, 10.0);

        // OTM call
        let intrinsic =
            ExpirationHandler::calculate_intrinsic_value(OptionType::Call, 100.0, 95.0);
        assert_eq!(intrinsic, 0.0);

        // ATM call
        let intrinsic =
            ExpirationHandler::calculate_intrinsic_value(OptionType::Call, 100.0, 100.0);
        assert_eq!(intrinsic, 0.0);
    }

    #[test]
    fn test_intrinsic_value_put() {
        // ITM put
        let intrinsic =
            ExpirationHandler::calculate_intrinsic_value(OptionType::Put, 100.0, 90.0);
        assert_eq!(intrinsic, 10.0);

        // OTM put
        let intrinsic =
            ExpirationHandler::calculate_intrinsic_value(OptionType::Put, 100.0, 105.0);
        assert_eq!(intrinsic, 0.0);

        // ATM put
        let intrinsic =
            ExpirationHandler::calculate_intrinsic_value(OptionType::Put, 100.0, 100.0);
        assert_eq!(intrinsic, 0.0);
    }

    #[test]
    fn test_long_call_itm_expiration() {
        let position = create_test_position(OptionType::Call, 100.0, 1, 1735000000);
        let event = ExpirationHandler::process_expiration(&position, 110.0);

        match event {
            ExpirationEvent::AutoExercise {
                intrinsic_value,
                settlement_amount,
                ..
            } => {
                assert_eq!(intrinsic_value, 10.0);
                assert_eq!(settlement_amount, 1000.0); // 10 * 1 * 100
            }
            _ => panic!("Expected AutoExercise event"),
        }
    }

    #[test]
    fn test_long_call_otm_expiration() {
        let position = create_test_position(OptionType::Call, 100.0, 1, 1735000000);
        let event = ExpirationHandler::process_expiration(&position, 95.0);

        match event {
            ExpirationEvent::Expire {
                option_type,
                strike,
                ..
            } => {
                assert_eq!(option_type, OptionType::Call);
                assert_eq!(strike, 100.0);
            }
            _ => panic!("Expected Expire event"),
        }
    }

    #[test]
    fn test_short_put_itm_expiration() {
        let position = create_test_position(OptionType::Put, 100.0, -1, 1735000000);
        let event = ExpirationHandler::process_expiration(&position, 90.0);

        match event {
            ExpirationEvent::Assignment {
                strike,
                settlement_amount,
                ..
            } => {
                assert_eq!(strike, 100.0);
                assert_eq!(settlement_amount, -1000.0); // (90-100) * -1 * 100 = -1000
            }
            _ => panic!("Expected Assignment event"),
        }
    }

    #[test]
    fn test_short_call_otm_expiration() {
        let position = create_test_position(OptionType::Call, 100.0, -1, 1735000000);
        let event = ExpirationHandler::process_expiration(&position, 95.0);

        match event {
            ExpirationEvent::Expire { .. } => {
                assert_eq!(event.settlement_amount(), 0.0);
            }
            _ => panic!("Expected Expire event"),
        }
    }

    #[test]
    fn test_multiple_contracts_settlement() {
        let position = create_test_position(OptionType::Call, 100.0, 5, 1735000000);
        let settlement = ExpirationHandler::calculate_settlement(&position, 110.0);

        assert_eq!(settlement, 5000.0); // 10 * 5 * 100
    }

    #[test]
    fn test_is_itm() {
        assert!(ExpirationHandler::is_itm(OptionType::Call, 100.0, 105.0));
        assert!(!ExpirationHandler::is_itm(OptionType::Call, 100.0, 95.0));

        assert!(ExpirationHandler::is_itm(OptionType::Put, 100.0, 95.0));
        assert!(!ExpirationHandler::is_itm(OptionType::Put, 100.0, 105.0));
    }

    #[test]
    fn test_is_atm() {
        assert!(ExpirationHandler::is_atm(100.0, 100.0, 0.01));
        assert!(ExpirationHandler::is_atm(100.0, 100.5, 0.01));
        assert!(!ExpirationHandler::is_atm(100.0, 102.0, 0.01));
    }

    #[test]
    fn test_moneyness() {
        let moneyness = ExpirationHandler::moneyness(OptionType::Call, 100.0, 110.0);
        assert!((moneyness - 1.1).abs() < 0.01);

        let moneyness = ExpirationHandler::moneyness(OptionType::Put, 100.0, 90.0);
        assert!((moneyness - 1.111).abs() < 0.01);
    }

    #[test]
    fn test_check_expirations_batch() {
        let positions = vec![
            create_test_position(OptionType::Call, 100.0, 1, 1735000000),
            create_test_position(OptionType::Put, 100.0, 1, 1735000000),
            create_test_position(OptionType::Call, 100.0, 1, 1736000000), // Not expired
        ];

        let events = ExpirationHandler::check_expirations(&positions, 1735100000, 110.0);

        assert_eq!(events.len(), 2); // First two expired
        assert!(matches!(events[0], ExpirationEvent::AutoExercise { .. }));
        assert!(matches!(events[1], ExpirationEvent::Expire { .. }));
    }
}
