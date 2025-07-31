/// A simple linear regression model implementation
/// y = a + b*x where 'a' is the intercept and 'b' is the slope
#[derive(Debug, Clone)]
pub struct LinearRegression {
    intercept: f64,
    slope: f64,
    is_fitted: bool,
}

impl LinearRegression {
    /// Create a new, unfitted linear regression model
    pub fn new() -> Self {
        LinearRegression {
            intercept: 0.0,
            slope: 0.0,
            is_fitted: false,
        }
    }

    /// Fit the model using ordinary least squares
    /// 
    /// # Arguments
    /// * `x` - Input features (independent variable)
    /// * `y` - Target values (dependent variable)
    /// 
    /// # Returns
    /// * `Result<(), &'static str>` - Ok if successful, Err with message otherwise
    pub fn fit(&mut self, x: &[f64], y: &[f64]) -> Result<(), &'static str> {
        if x.len() != y.len() {
            return Err("Input arrays must have the same length");
        }
        
        if x.is_empty() {
            return Err("Input arrays must not be empty");
        }
        
        let n = x.len() as f64;
        
        // Calculate means
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;
        
        // Calculate slope (b) using the formula:
        // b = Σ((xi - x_mean) * (yi - y_mean)) / Σ((xi - x_mean)²)
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..x.len() {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }
        
        if denominator == 0.0 {
            return Err("All x values are identical, cannot fit a line");
        }
        
        self.slope = numerator / denominator;
        
        // Calculate intercept (a) using: a = y_mean - b * x_mean
        self.intercept = y_mean - self.slope * x_mean;
        
        self.is_fitted = true;
        Ok(())
    }
    
    /// Predict y values for given x values
    /// 
    /// # Arguments
    /// * `x` - Input values to predict for
    /// 
    /// # Returns
    /// * `Result<Vec<f64>, &'static str>` - Predicted values or error
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>, &'static str> {
        if !self.is_fitted {
            return Err("Model must be fitted before making predictions");
        }
        
        Ok(x.iter()
            .map(|&xi| self.intercept + self.slope * xi)
            .collect())
    }
    
    /// Predict a single y value for a given x value
    pub fn predict_single(&self, x: f64) -> Result<f64, &'static str> {
        if !self.is_fitted {
            return Err("Model must be fitted before making predictions");
        }
        
        Ok(self.intercept + self.slope * x)
    }
    
    /// Calculate R-squared (coefficient of determination)
    /// 
    /// # Arguments
    /// * `x` - Input features
    /// * `y` - True target values
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - R-squared value between 0 and 1
    pub fn r_squared(&self, x: &[f64], y: &[f64]) -> Result<f64, &'static str> {
        if !self.is_fitted {
            return Err("Model must be fitted before calculating R-squared");
        }
        
        if x.len() != y.len() || x.is_empty() {
            return Err("Invalid input arrays");
        }
        
        let predictions = self.predict(x)?;
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        
        // Total sum of squares
        let ss_tot: f64 = y.iter()
            .map(|&yi| (yi - y_mean).powi(2))
            .sum();
        
        // Residual sum of squares
        let ss_res: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(&yi, &yi_pred)| (yi - yi_pred).powi(2))
            .sum();
        
        if ss_tot == 0.0 {
            return Ok(1.0); // Perfect fit if all y values are the same
        }
        
        Ok(1.0 - (ss_res / ss_tot))
    }
    
    /// Get the model parameters
    pub fn get_params(&self) -> Option<(f64, f64)> {
        if self.is_fitted {
            Some((self.intercept, self.slope))
        } else {
            None
        }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perfect_fit() {
        let mut model = LinearRegression::new();
        
        // y = 2 + 3*x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 8.0, 11.0, 14.0, 17.0];
        
        assert!(model.fit(&x, &y).is_ok());
        
        let (intercept, slope) = model.get_params().unwrap();
        assert!((intercept - 2.0).abs() < 1e-10);
        assert!((slope - 3.0).abs() < 1e-10);
        
        let r2 = model.r_squared(&x, &y).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_prediction() {
        let mut model = LinearRegression::new();
        
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        
        model.fit(&x, &y).unwrap();
        
        let pred = model.predict_single(5.0).unwrap();
        assert!((pred - 10.0).abs() < 1e-10);
        
        let preds = model.predict(&vec![0.0, 5.0, 10.0]).unwrap();
        assert_eq!(preds.len(), 3);
        assert!((preds[0] - 0.0).abs() < 1e-10);
        assert!((preds[1] - 10.0).abs() < 1e-10);
        assert!((preds[2] - 20.0).abs() < 1e-10);
    }
}

// Example usage
fn main() {
    // Create sample data
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![2.5, 5.1, 7.9, 10.2, 12.8, 15.3, 17.7, 20.1, 22.6, 25.0];
    
    // Create and fit the model
    let mut model = LinearRegression::new();
    
    match model.fit(&x, &y) {
        Ok(_) => println!("Model fitted successfully!"),
        Err(e) => {
            eprintln!("Error fitting model: {}", e);
            return;
        }
    }
    
    // Get model parameters
    if let Some((intercept, slope)) = model.get_params() {
        println!("Linear equation: y = {:.4} + {:.4}*x", intercept, slope);
    }
    
    // Calculate R-squared
    match model.r_squared(&x, &y) {
        Ok(r2) => println!("R-squared: {:.4}", r2),
        Err(e) => eprintln!("Error calculating R-squared: {}", e),
    }
    
    // Make predictions
    let test_values = vec![11.0, 12.0, 13.0];
    match model.predict(&test_values) {
        Ok(predictions) => {
            println!("\nPredictions:");
            for (x_val, y_pred) in test_values.iter().zip(predictions.iter()) {
                println!("x = {:.1}, predicted y = {:.4}", x_val, y_pred);
            }
        }
        Err(e) => eprintln!("Error making predictions: {}", e),
    }
}
