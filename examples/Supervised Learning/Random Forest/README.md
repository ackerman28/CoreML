# Random Forest Classifier

An ensemble of Decision Trees using Bootstrap Aggregating (Bagging).

## Mathematical Overview
For a set of $B$ trees, the final prediction $\hat{y}$ for a sample $x$ is:

$$\hat{y} = \text{mode} \{ T_1(x), T_2(x), \dots, T_B(x) \}$$

## Key Features
- **Bootstrap Sampling**: Draws $N$ samples with replacement.
- **Majority Voting**: Robust classification.
- **Scalability**: Reuses the core `DecisionTree` logic.