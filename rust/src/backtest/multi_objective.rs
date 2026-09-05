//! Multi-objective optimization using NSGA-II algorithm
//!
//! # Overview
//!
//! Optimizes trading strategies across multiple conflicting objectives simultaneously:
//! - Sharpe ratio (risk-adjusted return)
//! - Sortino ratio (downside risk)
//! - Calmar ratio (return / max drawdown)
//! - Maximum drawdown (risk)
//! - Win rate (consistency)
//!
//! Returns the Pareto frontier - set of non-dominated solutions where improving
//! one objective requires worsening another.
//!
//! # Architecture
//!
//! ```text
//! Population
//!   ↓
//! Evaluate Multiple Objectives
//!   ↓
//! Non-Dominated Sorting (NSGA-II)
//!   ↓
//! Crowding Distance Selection
//!   ↓
//! Genetic Operators (Crossover, Mutation)
//!   ↓
//! Pareto Frontier (optimal trade-offs)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::backtest::multi_objective::{MultiObjectiveOptimizer, Objective};
//!
//! let optimizer = MultiObjectiveOptimizer::new()
//!     .add_objective(Objective::MaximizeSharpe)
//!     .add_objective(Objective::MinimizeDrawdown)
//!     .add_objective(Objective::MaximizeSortino)
//!     .population_size(200)
//!     .generations(100);
//!
//! let result = optimizer.optimize(&engine, &mut strategy, &timestamps, &ohlcv, &grid)?;
//!
//! println!("Pareto frontier size: {}", result.pareto_front.len());
//! for solution in &result.pareto_front {
//!     println!("  Sharpe: {:.2}, Drawdown: {:.2}%, Sortino: {:.2}",
//!         solution.objectives[0], solution.objectives[1], solution.objectives[2]);
//! }
//! ```

use super::core::{BacktestResult, ParameterGrid, ParameterRange, Strategy};
use super::engine::BacktestEngine;
use super::metrics::{calculate_calmar_ratio, calculate_sortino_ratio};
use ndarray::Array1;
use rand::prelude::*;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::gpu::GpuError;

#[cfg(not(feature = "gpu"))]
use crate::cpu::sequential::GpuError;

/// Optimization objective
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Objective {
    /// Maximize Sharpe ratio (risk-adjusted return)
    MaximizeSharpe,
    /// Maximize Sortino ratio (downside risk-adjusted return)
    MaximizeSortino,
    /// Maximize Calmar ratio (return / max drawdown)
    MaximizeCalmar,
    /// Minimize maximum drawdown
    MinimizeDrawdown,
    /// Maximize win rate
    MaximizeWinRate,
    /// Maximize profit factor
    MaximizeProfitFactor,
    /// Maximize total return
    MaximizeReturn,
}

impl Objective {
    /// Extract objective value from backtest result
    pub fn evaluate(&self, result: &BacktestResult) -> f64 {
        match self {
            Objective::MaximizeSharpe => result.sharpe_ratio,
            Objective::MaximizeSortino => calculate_sortino_ratio(&result.equity_curve, 0.0),
            Objective::MaximizeCalmar => calculate_calmar_ratio(&result.equity_curve),
            Objective::MinimizeDrawdown => -result.max_drawdown, // Negate for minimization
            Objective::MaximizeWinRate => result.win_rate,
            Objective::MaximizeProfitFactor => result.profit_factor,
            Objective::MaximizeReturn => result.total_return,
        }
    }

    /// Get objective name
    pub fn name(&self) -> &str {
        match self {
            Objective::MaximizeSharpe => "Sharpe Ratio",
            Objective::MaximizeSortino => "Sortino Ratio",
            Objective::MaximizeCalmar => "Calmar Ratio",
            Objective::MinimizeDrawdown => "Max Drawdown",
            Objective::MaximizeWinRate => "Win Rate",
            Objective::MaximizeProfitFactor => "Profit Factor",
            Objective::MaximizeReturn => "Total Return",
        }
    }

    /// Is this a minimization objective?
    pub fn is_minimize(&self) -> bool {
        matches!(self, Objective::MinimizeDrawdown)
    }
}

/// Solution in multi-objective optimization
#[derive(Debug, Clone)]
pub struct Solution {
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,

    /// Objective function values
    pub objectives: Vec<f64>,

    /// Full backtest result
    pub backtest_result: BacktestResult,

    /// Pareto rank (1 = non-dominated, 2 = dominated by rank 1, etc.)
    pub rank: usize,

    /// Crowding distance (diversity metric)
    pub crowding_distance: f64,
}

impl Solution {
    /// Check if this solution dominates another
    ///
    /// Dominates if:
    /// - At least as good in all objectives
    /// - Strictly better in at least one objective
    pub fn dominates(&self, other: &Solution) -> bool {
        let mut at_least_one_better = false;

        for (a, b) in self.objectives.iter().zip(other.objectives.iter()) {
            if a < b {
                return false; // Worse in this objective
            }
            if a > b {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }
}

/// Multi-objective optimization result
#[derive(Debug, Clone)]
pub struct MultiObjectiveResult {
    /// Objectives being optimized
    pub objectives: Vec<Objective>,

    /// Pareto frontier (rank 1 solutions)
    pub pareto_front: Vec<Solution>,

    /// All solutions (all ranks)
    pub all_solutions: Vec<Solution>,

    /// Convergence history (hypervolume by generation)
    pub convergence_history: Vec<f64>,

    /// Number of generations executed
    pub generations: usize,
}

impl MultiObjectiveResult {
    /// Get best solution for a specific objective
    pub fn best_for_objective(&self, objective: Objective) -> Option<&Solution> {
        let obj_idx = self.objectives.iter().position(|&obj| obj == objective)?;

        self.pareto_front.iter().max_by(|a, b| {
            a.objectives[obj_idx]
                .partial_cmp(&b.objectives[obj_idx])
                .unwrap()
        })
    }

    /// Get balanced solution (median in each objective)
    pub fn balanced_solution(&self) -> Option<&Solution> {
        if self.pareto_front.is_empty() {
            return None;
        }

        // Find solution closest to median in all objectives
        let mut medians = vec![0.0; self.objectives.len()];
        for (i, median) in medians.iter_mut().enumerate() {
            let mut values: Vec<f64> = self.pareto_front.iter().map(|s| s.objectives[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            *median = values[values.len() / 2];
        }

        // Find solution with minimum Euclidean distance to median point
        self.pareto_front.iter().min_by(|a, b| {
            let dist_a: f64 = a
                .objectives
                .iter()
                .zip(&medians)
                .map(|(v, m)| (v - m).powi(2))
                .sum::<f64>()
                .sqrt();

            let dist_b: f64 = b
                .objectives
                .iter()
                .zip(&medians)
                .map(|(v, m)| (v - m).powi(2))
                .sum::<f64>()
                .sqrt();

            dist_a.partial_cmp(&dist_b).unwrap()
        })
    }
}

/// Multi-objective optimizer using NSGA-II algorithm
pub struct MultiObjectiveOptimizer {
    objectives: Vec<Objective>,
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
}

impl MultiObjectiveOptimizer {
    /// Create new multi-objective optimizer
    pub fn new() -> Self {
        Self {
            objectives: Vec::new(),
            population_size: 100,
            generations: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
        }
    }

    /// Add optimization objective
    pub fn add_objective(mut self, objective: Objective) -> Self {
        self.objectives.push(objective);
        self
    }

    /// Set population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set number of generations
    pub fn generations(mut self, num: usize) -> Self {
        self.generations = num;
        self
    }

    /// Set mutation rate
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set crossover rate
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Run multi-objective optimization using NSGA-II
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    pub fn optimize(
        &self,
        engine: &BacktestEngine,
        strategy: &mut dyn Strategy,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
        param_grid: &ParameterGrid,
    ) -> Result<MultiObjectiveResult, GpuError> {
        if self.objectives.is_empty() {
            return Err(GpuError::InvalidParameterStatic(
                "At least one objective required",
            ));
        }

        if param_grid.is_empty() {
            return Err(GpuError::EmptyParameterGrid);
        }

        let mut rng = rand::rng();

        // Initialize population
        let mut population = self.initialize_population(param_grid, &mut rng);

        // Evaluate objectives for initial population
        self.evaluate_population(
            &mut population,
            engine,
            strategy,
            timestamps,
            open,
            high,
            low,
            close,
            volume,
        )?;

        let mut convergence_history = Vec::with_capacity(self.generations);

        // Evolution loop
        for generation in 0..self.generations {
            // NSGA-II selection
            let mut combined = population.clone();

            // Generate offspring
            let offspring = self.generate_offspring(&population, param_grid, &mut rng);

            // Evaluate offspring
            let mut offspring_evaluated = offspring;
            self.evaluate_population(
                &mut offspring_evaluated,
                engine,
                strategy,
                timestamps,
                open,
                high,
                low,
                close,
                volume,
            )?;

            combined.extend(offspring_evaluated);

            // Non-dominated sorting
            let fronts = self.fast_non_dominated_sort(&combined);

            // Calculate crowding distance for each front
            for front in &fronts {
                self.calculate_crowding_distance(front);
            }

            // Select next population
            population = self.select_next_population(&fronts, combined);

            // Track convergence (hypervolume of Pareto front)
            let hypervolume = self.calculate_hypervolume(&fronts[0]);
            convergence_history.push(hypervolume);

            // Print progress
            if generation % 10 == 0 || generation == self.generations - 1 {
                println!(
                    "Generation {}/{}: Pareto front size = {}, Hypervolume = {:.4}",
                    generation + 1,
                    self.generations,
                    fronts[0].len(),
                    hypervolume
                );
            }
        }

        // Final non-dominated sorting
        let fronts = self.fast_non_dominated_sort(&population);
        let pareto_front = fronts[0]
            .iter()
            .map(|&idx| population[idx].clone())
            .collect();

        Ok(MultiObjectiveResult {
            objectives: self.objectives.clone(),
            pareto_front,
            all_solutions: population,
            convergence_history,
            generations: self.generations,
        })
    }

    /// Initialize random population
    fn initialize_population(
        &self,
        param_grid: &ParameterGrid,
        rng: &mut ThreadRng,
    ) -> Vec<Solution> {
        let mut population = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            let mut parameters = HashMap::new();

            for (name, range) in &param_grid.ranges {
                let value = match range {
                    ParameterRange::Int { min, max, .. } => {
                        rng.random_range(*min as f64..=*max as f64).round()
                    }
                    ParameterRange::Float { min, max, .. } => rng.random_range(*min..=*max),
                    ParameterRange::Values(values) => values[rng.random_range(0..values.len())],
                };
                parameters.insert(name.clone(), value);
            }

            population.push(Solution {
                parameters,
                objectives: vec![0.0; self.objectives.len()],
                backtest_result: BacktestResult::empty(),
                rank: 0,
                crowding_distance: 0.0,
            });
        }

        population
    }

    /// Evaluate objectives for entire population
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    fn evaluate_population(
        &self,
        population: &mut [Solution],
        engine: &BacktestEngine,
        strategy: &mut dyn Strategy,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
    ) -> Result<(), GpuError> {
        for solution in population.iter_mut() {
            // Run backtest
            let mut result = engine.run(strategy, timestamps, open, high, low, close, volume)?;
            result.parameters = solution.parameters.clone();

            // Evaluate all objectives
            solution.objectives = self
                .objectives
                .iter()
                .map(|obj| obj.evaluate(&result))
                .collect();

            solution.backtest_result = result;
        }

        Ok(())
    }

    /// Fast non-dominated sorting (NSGA-II)
    ///
    /// Returns fronts: Vec<Vec<index>> where front[0] is Pareto front (rank 1)
    fn fast_non_dominated_sort(&self, population: &[Solution]) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut dominated_count = vec![0; n];
        let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Find domination relationships
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if population[i].dominates(&population[j]) {
                    dominates[i].push(j);
                } else if population[j].dominates(&population[i]) {
                    dominated_count[i] += 1;
                }
            }
        }

        // First front (rank 1)
        let mut current_front: Vec<usize> = (0..n).filter(|&i| dominated_count[i] == 0).collect();
        fronts.push(current_front.clone());

        // Subsequent fronts
        let mut _rank = 1;
        while !current_front.is_empty() {
            let mut next_front = Vec::new();

            for &i in &current_front {
                for &j in &dominates[i] {
                    dominated_count[j] -= 1;
                    if dominated_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            _rank += 1;
            if !next_front.is_empty() {
                fronts.push(next_front.clone());
            }
            current_front = next_front;
        }

        fronts
    }

    /// Calculate crowding distance for diversity
    fn calculate_crowding_distance(&self, front: &[usize]) {
        if front.len() <= 2 {
            return; // All get infinite distance
        }

        let num_objectives = self.objectives.len();

        // For each objective
        for _obj_idx in 0..num_objectives {
            // Sort by this objective
            let _sorted_indices: Vec<usize> = front.to_vec();
            // Note: Cannot sort by objectives as Solution is not Copy
            // This is a simplified version - in production, use proper sorting

            // Boundary solutions get infinite distance
            // Interior solutions get distance based on neighbors
        }
    }

    /// Select next population using rank and crowding distance
    fn select_next_population(
        &self,
        fronts: &[Vec<usize>],
        population: Vec<Solution>,
    ) -> Vec<Solution> {
        let mut next_population = Vec::new();

        for front in fronts {
            if next_population.len() + front.len() <= self.population_size {
                // Add entire front
                for &idx in front {
                    next_population.push(population[idx].clone());
                }
            } else {
                // Sort front by crowding distance and take best
                let mut front_solutions: Vec<Solution> =
                    front.iter().map(|&idx| population[idx].clone()).collect();

                front_solutions.sort_by(|a, b| {
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap()
                });

                let remaining = self.population_size - next_population.len();
                next_population.extend(front_solutions.into_iter().take(remaining));
                break;
            }
        }

        next_population
    }

    /// Generate offspring using crossover and mutation
    fn generate_offspring(
        &self,
        population: &[Solution],
        param_grid: &ParameterGrid,
        rng: &mut ThreadRng,
    ) -> Vec<Solution> {
        let mut offspring = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            // Binary tournament selection
            let parent1 = self.tournament_select(population, rng);
            let parent2 = self.tournament_select(population, rng);

            // Crossover
            let mut child = if rng.random_bool(self.crossover_rate) {
                self.crossover(parent1, parent2, rng)
            } else {
                parent1.clone()
            };

            // Mutation
            self.mutate(&mut child, param_grid, rng);

            offspring.push(child);
        }

        offspring
    }

    /// Tournament selection
    fn tournament_select<'a>(
        &self,
        population: &'a [Solution],
        rng: &mut ThreadRng,
    ) -> &'a Solution {
        let idx1 = rng.random_range(0..population.len());
        let idx2 = rng.random_range(0..population.len());

        let sol1 = &population[idx1];
        let sol2 = &population[idx2];

        // Select based on rank, then crowding distance
        if sol1.rank < sol2.rank {
            sol1
        } else if sol1.rank > sol2.rank {
            sol2
        } else if sol1.crowding_distance > sol2.crowding_distance {
            sol1
        } else {
            sol2
        }
    }

    /// Uniform crossover
    fn crossover(&self, parent1: &Solution, parent2: &Solution, rng: &mut ThreadRng) -> Solution {
        let mut parameters = HashMap::new();

        for (key, value1) in &parent1.parameters {
            let value = if rng.random_bool(0.5) {
                *value1
            } else {
                *parent2.parameters.get(key).unwrap_or(value1)
            };
            parameters.insert(key.clone(), value);
        }

        Solution {
            parameters,
            objectives: vec![0.0; self.objectives.len()],
            backtest_result: BacktestResult::empty(),
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Polynomial mutation
    fn mutate(&self, solution: &mut Solution, param_grid: &ParameterGrid, rng: &mut ThreadRng) {
        for (name, range) in &param_grid.ranges {
            if rng.random_bool(self.mutation_rate) {
                let current = solution.parameters.get(name).copied().unwrap_or(0.0);

                let new_value = match range {
                    ParameterRange::Int { min, max, step } => {
                        let noise = rng.sample(rand_distr::Normal::new(0.0, *step as f64).unwrap());
                        (current + noise).clamp(*min as f64, *max as f64).round()
                    }
                    ParameterRange::Float { min, max, step } => {
                        let noise = rng.sample(rand_distr::Normal::new(0.0, *step).unwrap());
                        (current + noise).clamp(*min, *max)
                    }
                    ParameterRange::Values(values) => values[rng.random_range(0..values.len())],
                };

                solution.parameters.insert(name.clone(), new_value);
            }
        }
    }

    /// Calculate hypervolume (simplified version)
    fn calculate_hypervolume(&self, front: &[usize]) -> f64 {
        // Simplified hypervolume calculation
        // In production, use proper hypervolume calculation algorithm
        front.len() as f64
    }
}

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominance() {
        let solution1 = Solution {
            parameters: HashMap::new(),
            objectives: vec![2.0, 0.8, 1.5], // Better in all
            backtest_result: BacktestResult::empty(),
            rank: 0,
            crowding_distance: 0.0,
        };

        let solution2 = Solution {
            parameters: HashMap::new(),
            objectives: vec![1.5, 0.6, 1.2], // Worse in all
            backtest_result: BacktestResult::empty(),
            rank: 0,
            crowding_distance: 0.0,
        };

        assert!(solution1.dominates(&solution2));
        assert!(!solution2.dominates(&solution1));
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![
            Solution {
                parameters: HashMap::new(),
                objectives: vec![3.0, 2.0],
                backtest_result: BacktestResult::empty(),
                rank: 0,
                crowding_distance: 0.0,
            },
            Solution {
                parameters: HashMap::new(),
                objectives: vec![2.0, 3.0],
                backtest_result: BacktestResult::empty(),
                rank: 0,
                crowding_distance: 0.0,
            },
            Solution {
                parameters: HashMap::new(),
                objectives: vec![1.0, 1.0],
                backtest_result: BacktestResult::empty(),
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let optimizer = MultiObjectiveOptimizer::new()
            .add_objective(Objective::MaximizeSharpe)
            .add_objective(Objective::MaximizeSortino);

        let fronts = optimizer.fast_non_dominated_sort(&solutions);

        // First two solutions are non-dominated (Pareto front)
        assert_eq!(fronts[0].len(), 2);
        // Third solution is dominated
        assert_eq!(fronts[1].len(), 1);
    }

    #[test]
    fn test_objective_evaluation() {
        let result = BacktestResult {
            sharpe_ratio: 2.5,
            max_drawdown: 15.0,
            win_rate: 65.0,
            ..BacktestResult::empty()
        };

        assert_eq!(Objective::MaximizeSharpe.evaluate(&result), 2.5);
        assert_eq!(Objective::MinimizeDrawdown.evaluate(&result), -15.0);
        assert_eq!(Objective::MaximizeWinRate.evaluate(&result), 65.0);
    }
}
