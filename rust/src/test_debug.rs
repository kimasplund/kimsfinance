#[cfg(test)]
mod debug_tests {
    use crate::indicators::{Indicator, DEMA, TEMA};
    use ndarray::arr1;

    #[test]
    fn debug_dema() {
        let prices = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dema = DEMA::new(3).unwrap();
        let result = dema.calculate(prices.view()).unwrap();
        
        eprintln!("DEMA results:");
        for (i, &val) in result.iter().enumerate() {
            eprintln!("  [{}] = {} (finite: {})", i, val, val.is_finite());
        }
    }

    #[test]
    fn debug_tema() {
        let prices = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let tema = TEMA::new(3).unwrap();
        let result = tema.calculate(prices.view()).unwrap();
        
        eprintln!("\nTEMA results:");
        for (i, &val) in result.iter().enumerate() {
            eprintln!("  [{}] = {} (finite: {})", i, val, val.is_finite());
        }
    }
}
