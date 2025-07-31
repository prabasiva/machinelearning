use std::f64::consts::E;

/// A logistic regression model for binary classification
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
    is_fitted: bool,
}

impl LogisticRegression {
    /// Create a new logistic regression model
    /// 
    /// # Arguments
    /// * `learning_rate` - Step size for gradient descent (default: 0.01)
    /// * `max_iterations` - Maximum number of iterations (default: 1000)
    /// * `tolerance` - Convergence tolerance (default: 1e-6)
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        LogisticRegression {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            max_iterations,
            tolerance,
            is_fitted: false,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(0.01, 1000, 1e-6)
    }

    /// Sigmoid activation function
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + E.powf(-z))
    }

    /// Compute the linear combination of features and weights
    fn linear_combination(&self, features: &[f64]) -> f64 {
        self.weights.iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>() + self.bias
    }

    /// Compute the cost (negative log-likelihood)
    fn compute_cost(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mut cost = 0.0;

        for (features, &label) in x.iter().zip(y.iter()) {
            let z = self.linear_combination(features);
            let prediction = Self::sigmoid(z);
            
            // Avoid log(0) by clamping predictions
            let pred_clamped = prediction.max(1e-15).min(1.0 - 1e-15);
            
            cost += -label * pred_clamped.ln() - (1.0 - label) * (1.0 - pred_clamped).ln();
        }

        cost / n
    }

    /// Fit the model using gradient descent
    /// 
    /// # Arguments
    /// * `x` - Training features (each inner Vec is one sample)
    /// * `y` - Training labels (0 or 1)
    /// 
    /// # Returns
    /// * `Result<Vec<f64>, &'static str>` - Cost history or error
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, &'static str> {
        // Validation
        if x.is_empty() || y.is_empty() {
            return Err("Input data cannot be empty");
        }

        if x.len() != y.len() {
            return Err("Features and labels must have the same number of samples");
        }

        let n_features = x[0].len();
        if n_features == 0 {
            return Err("Feature vectors cannot be empty");
        }

        // Check all samples have the same number of features
        if !x.iter().all(|sample| sample.len() == n_features) {
            return Err("All samples must have the same number of features");
        }

        // Check labels are binary
        if !y.iter().all(|&label| label == 0.0 || label == 1.0) {
            return Err("Labels must be 0 or 1");
        }

        // Initialize weights
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let n_samples = x.len() as f64;
        let mut cost_history = Vec::new();
        let mut prev_cost = f64::INFINITY;

        // Gradient descent
        for iteration in 0..self.max_iterations {
            // Compute gradients
            let mut weight_gradients = vec![0.0; n_features];
            let mut bias_gradient = 0.0;

            for (features, &label) in x.iter().zip(y.iter()) {
                let z = self.linear_combination(features);
                let prediction = Self::sigmoid(z);
                let error = prediction - label;

                // Update gradients
                for (j, &feature) in features.iter().enumerate() {
                    weight_gradients[j] += error * feature;
                }
                bias_gradient += error;
            }

            // Average gradients
            for grad in &mut weight_gradients {
                *grad /= n_samples;
            }
            bias_gradient /= n_samples;

            // Update parameters
            for (w, &grad) in self.weights.iter_mut().zip(weight_gradients.iter()) {
                *w -= self.learning_rate * grad;
            }
            self.bias -= self.learning_rate * bias_gradient;

            // Calculate cost
            let current_cost = self.compute_cost(x, y);
            cost_history.push(current_cost);

            // Check convergence
            if (prev_cost - current_cost).abs() < self.tolerance {
                println!("Converged at iteration {}", iteration);
                break;
            }

            prev_cost = current_cost;
        }

        self.is_fitted = true;
        Ok(cost_history)
    }

    /// Predict probabilities for given features
    /// 
    /// # Arguments
    /// * `x` - Features to predict for
    /// 
    /// # Returns
    /// * `Result<Vec<f64>, &'static str>` - Predicted probabilities
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, &'static str> {
        if !self.is_fitted {
            return Err("Model must be fitted before making predictions");
        }

        if x.is_empty() {
            return Ok(Vec::new());
        }

        if x[0].len() != self.weights.len() {
            return Err("Feature dimension mismatch");
        }

        Ok(x.iter()
            .map(|features| {
                let z = self.linear_combination(features);
                Self::sigmoid(z)
            })
            .collect())
    }

    /// Predict class labels (0 or 1) for given features
    /// 
    /// # Arguments
    /// * `x` - Features to predict for
    /// * `threshold` - Decision threshold (default: 0.5)
    /// 
    /// # Returns
    /// * `Result<Vec<f64>, &'static str>` - Predicted class labels
    pub fn predict(&self, x: &[Vec<f64>], threshold: f64) -> Result<Vec<f64>, &'static str> {
        let probabilities = self.predict_proba(x)?;
        
        Ok(probabilities.iter()
            .map(|&prob| if prob >= threshold { 1.0 } else { 0.0 })
            .collect())
    }

    /// Calculate accuracy on test data
    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> Result<f64, &'static str> {
        if x.len() != y.len() {
            return Err("Features and labels must have the same length");
        }

        if x.is_empty() {
            return Err("Cannot calculate accuracy on empty data");
        }

        let predictions = self.predict(x, 0.5)?;
        let correct = predictions.iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();

        Ok(correct as f64 / x.len() as f64)
    }

    /// Get model parameters
    pub fn get_params(&self) -> Option<(&[f64], f64)> {
        if self.is_fitted {
            Some((&self.weights, self.bias))
        } else {
            None
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Set maximum iterations
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(LogisticRegression::sigmoid(10.0) > 0.99);
        assert!(LogisticRegression::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_linearly_separable_data() {
        let mut model = LogisticRegression::new(0.1, 100, 1e-4);
        
        // Create linearly separable data
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
            vec![6.0, 6.0],
        ];
        
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        
        model.fit(&x, &y).unwrap();
        
        let accuracy = model.accuracy(&x, &y).unwrap();
        assert!(accuracy >= 0.8); // Should achieve high accuracy
    }

    #[test]
    fn test_predict_proba() {
        let mut model = LogisticRegression::default();
        
        let x_train = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        
        let y_train = vec![0.0, 0.0, 0.0, 1.0];
        
        model.fit(&x_train, &y_train).unwrap();
        
        let probabilities = model.predict_proba(&x_train).unwrap();
        assert_eq!(probabilities.len(), 4);
        
        // All probabilities should be between 0 and 1
        for prob in probabilities {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }
}

// Example usage
fn main() {
    // Generate sample data for binary classification
    let x_train = vec![
        vec![2.8, 1.5],
        vec![3.2, 1.3],
        vec![3.5, 1.7],
        vec![4.0, 2.0],
        vec![4.5, 2.2],
        vec![5.0, 2.8],
        vec![5.5, 3.0],
        vec![6.0, 3.5],
        vec![6.5, 4.0],
        vec![7.0, 4.5],
    ];
    
    let y_train = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    
    // Create and train the model
    let mut model = LogisticRegression::new(0.5, 1000, 1e-6);
    
    println!("Training logistic regression model...");
    match model.fit(&x_train, &y_train) {
        Ok(cost_history) => {
            println!("Model trained successfully!");
            println!("Final cost: {:.6}", cost_history.last().unwrap());
            println!("Number of iterations: {}", cost_history.len());
        }
        Err(e) => {
            eprintln!("Error training model: {}", e);
            return;
        }
    }
    
    // Get model parameters
    if let Some((weights, bias)) = model.get_params() {
        println!("\nModel parameters:");
        println!("Weights: {:?}", weights);
        println!("Bias: {:.4}", bias);
    }
    
    // Calculate training accuracy
    match model.accuracy(&x_train, &y_train) {
        Ok(acc) => println!("\nTraining accuracy: {:.2}%", acc * 100.0),
        Err(e) => eprintln!("Error calculating accuracy: {}", e),
    }
    
    // Make predictions on new data
    let x_test = vec![
        vec![3.0, 1.5],  // Should be class 0
        vec![6.0, 3.8],  // Should be class 1
        vec![4.5, 2.5],  // Near decision boundary
    ];
    
    println!("\nPredictions on test data:");
    match model.predict_proba(&x_test) {
        Ok(probabilities) => {
            for (i, prob) in probabilities.iter().enumerate() {
                println!("Sample {}: P(y=1) = {:.4}", i + 1, prob);
            }
        }
        Err(e) => eprintln!("Error making predictions: {}", e),
    }
    
    // Class predictions
    match model.predict(&x_test, 0.5) {
        Ok(predictions) => {
            println!("\nClass predictions (threshold=0.5):");
            for (i, pred) in predictions.iter().enumerate() {
                println!("Sample {}: class {}", i + 1, *pred as i32);
            }
        }
        Err(e) => eprintln!("Error making predictions: {}", e),
    }
}
