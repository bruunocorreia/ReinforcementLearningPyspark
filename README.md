# ReinforcementLearningPyspark

Multi-armed bandit models using PySpark

## Multi-Armed Bandit for Conversion Rate Experiments

This code provides a simple and efficient implementation of a Multi-Armed Bandit algorithm for conversion rate experiments. The code can be used to optimize experiments with large amounts of data.

## Adaptation to the code classes and functions

The `Mab` class represents the Multi-Armed Bandit algorithm. It has the following attributes:

* `priors`: a DataFrame containing the previous experiment results.
* `variable_test`: a list of possible values for the test variable.
* `reward`: the DataFrame column that contains the conversion rate metric.
* `log_results`: a boolean indicating whether experiment results should be logged.

The `EpsilonGreedy()` function implements the Epsilon-Greedy algorithm. It takes a DataFrame containing the previous experiment results and an epsilon value as parameters and returns the next test variable to be selected.

The `Ucb()` function implements the UCB algorithm. It takes a DataFrame containing the previous experiment results as a parameter and returns the next test variable to be selected.

The `ThompsonSampling()` function implements the Thompson Sampling algorithm. It takes a DataFrame containing the previous experiment results as a parameter and returns the next test variable to be selected.

## Example of use

```python

## Notes

* The DataFrame `df_priors` should contain the previous experiment results, with the following columns:
    * `teste`: the test variable.
    * `reward`: the conversion rate metric.
* The list `variable_test` should contain all possible values for the test variable.
* The parameter `epsilon` of the `EpsilonGreedy()` function should be a value between 0 and 1. The higher the value of epsilon, the more exploratory the algorithm will be.
* The parameter `log_results` of the `Mab` class can be used to log experiment results. The results will be logged to the console.

```python
# Create the Mab object
mab = Mab(priors=df_priors, variable_test=variable_test, reward="qtd_contatacoes")

# Choose the next test variable using the Epsilon-Greedy algorithm
next_test = mab.EpsilonGreedy(epsilon=0.1)

# Choose the next test variable using the UCB algorithm
next_test = mab.Ucb()

# Choose the next test variable using the Thompson Sampling algorithm
next_test = mab.ThompsonSampling()
