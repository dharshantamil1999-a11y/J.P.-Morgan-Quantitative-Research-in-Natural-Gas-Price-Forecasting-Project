import numpy as np
import pandas as pd
import math

# Load the dataset
file_path = 'Task 3 and 4_Loan_Data.csv'  # Change if needed
loan_data = pd.read_csv(file_path)

# MSE-based bucketization function
def mse_bucketization(fico_scores, n_buckets):
    sorted_scores = np.sort(fico_scores)
    n = len(sorted_scores)
    indices = np.linspace(0, n, n_buckets + 1, dtype=int)
    buckets = []
    for i in range(n_buckets):
        start_idx = indices[i]
        end_idx = indices[i + 1] if indices[i + 1] < n else n
        bucket_scores = sorted_scores[start_idx:end_idx]
        mean_score = np.mean(bucket_scores)
        boundaries = (bucket_scores[0], bucket_scores[-1])
        buckets.append({
            'start': boundaries[0],
            'end': boundaries[1],
            'mean': mean_score,
            'count': len(bucket_scores)
        })
    return buckets

# Log-likelihood functions
def log_likelihood(k, n):
    if n == 0 or k == 0 or k == n:
        return 0.0
    p = k / n
    return k * math.log(p) + (n - k) * math.log(1 - p)

def optimal_buckets(scores, defaults, totals, n_buckets):
    n = len(scores)
    dp = [[-float('inf')] * (n_buckets + 1) for _ in range(n)]
    boundary = [[-1] * (n_buckets + 1) for _ in range(n)]
    cum_defaults = np.cumsum(defaults)
    cum_totals = np.cumsum(totals)
    for i in range(n):
        d = cum_defaults[i]
        t = cum_totals[i]
        dp[i][1] = log_likelihood(d, t)
    for b in range(2, n_buckets + 1):
        for i in range(b - 1, n):
            for j in range(b - 2, i):
                d = cum_defaults[i] - cum_defaults[j]
                t = cum_totals[i] - cum_totals[j]
                val = dp[j][b - 1] + log_likelihood(d, t)
                if val > dp[i][b]:
                    dp[i][b] = val
                    boundary[i][b] = j
    boundaries = []
    idx = n - 1
    b = n_buckets
    while b > 1:
        j = boundary[idx][b]
        boundaries.append(scores[j + 1])
        idx = j
        b -= 1
    boundaries = sorted(boundaries)
    return boundaries

if __name__ == "__main__":
    n_buckets = 5
    fico_scores = loan_data['fico_score'].values
    # MSE-based bucketization
    mse_buckets = mse_bucketization(fico_scores, n_buckets)
    print("MSE-Based Buckets:")
    for i, b in enumerate(mse_buckets, 1):
        print(f"Bucket {i}: {b['start']} - {b['end']}, mean = {b['mean']:.2f}, count = {b['count']}")

    # Log-likelihood based bucketization
    fico_default_df = loan_data.groupby('fico_score').agg({'default': ['sum', 'count']})
    fico_default_df.columns = ['defaults', 'total']
    fico_default_df = fico_default_df.reset_index()
    scores = fico_default_df['fico_score'].values
    defaults = fico_default_df['defaults'].values
    totals = fico_default_df['total'].values
    boundaries = optimal_buckets(scores, defaults, totals, n_buckets)
    print("\nLog-Likelihood Based Buckaries:")
    print(boundaries)
