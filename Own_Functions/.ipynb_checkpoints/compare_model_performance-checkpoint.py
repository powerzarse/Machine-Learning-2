from scipy.stats import ttest_rel

def compare_model_performance(fold_performance_model_1, avg_perf_model_1, fold_performance_model_2, avg_perf_model_2):
    """
    Compare the performance of two models using a paired t-test.

    Args:
    fold_performance_model_1 (list): Performance of each fold for the first model.
    avg_perf_model_1 (float): Average performance of the first model.
    fold_performance_model_2 (list): Performance of each fold for the second model.
    avg_perf_model_2 (float): Average performance of the second model.
    """
    
    # Perform a paired t-test
    t_statistic, p_value = ttest_rel(fold_performance_model_1, fold_performance_model_2)

    print(f"Model 1 Average Performance: {avg_perf_model_1}")
    print(f"Model 2 Average Performance: {avg_perf_model_2}")
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)

    if p_value < 0.05:
        print("The difference in performance is statistically significant.")
        if avg_perf_model_1 > avg_perf_model_2:
            print("Model 1 performs significantly better than Model 2.")
        else:
            print("Model 2 performs significantly better than Model 1.")
    else:
        print("No significant difference in performance.")
    
    return t_statistic, p_value

# Example usage
# compare_model_performance(fold_performance_model_1, avg_perf_model_1, fold_performance_model_2, avg_perf_model_2)
# t_statistic, p_value = compare_model_performance(fold_performance_model_1, avg_perf_model_1, fold_performance_model_2, avg_perf_model_2)
