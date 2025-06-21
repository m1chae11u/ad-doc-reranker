import numpy as np
from scipy import stats

def compare_models(model_a_scores, model_b_scores, model_c_scores, alpha=0.05):
    """
    Performs paired t-tests between Model A and Models B and C,
    and computes 95% confidence intervals for the mean difference.
    
    Args:
        model_a_scores, model_b_scores, model_c_scores: Lists or arrays of scores (same length).
        alpha: Significance level (default 0.05 for 95% CI).
    
    Returns:
        Dict with t-test results and confidence intervals.
    """

    assert len(model_a_scores) == len(model_b_scores) == len(model_c_scores), "All models must have same number of samples."

    def test_pair(a, b, name_b):
        diff = np.array(a) - np.array(b)
        mean_diff = np.mean(diff)
        sem = stats.sem(diff)
        df = len(diff) - 1
        t_stat, p_value = stats.ttest_rel(a, b)

        # Compute 95% confidence interval
        ci_margin = stats.t.ppf(1 - alpha / 2, df) * sem
        ci = (mean_diff - ci_margin, mean_diff + ci_margin)

        return {
            f"vs_{name_b}": {
                "mean_difference": mean_diff,
                "confidence_interval": ci,
                "t_statistic": t_stat,
                "p_value": p_value,
                "statistically_significant": p_value < alpha
            }
        }

    results = {}
    results.update(test_pair(model_a_scores, model_b_scores, "ModelAB"))
    results.update(test_pair(model_a_scores, model_c_scores, "ModelAC"))
    results.update(test_pair(model_b_scores, model_c_scores, "ModelBC"))

    return results

# Example usage:
mrr_a = [0.0033, 0.0051, 0.0051, 0.0055, 0.0065, 0.0064]
mrr_b = [0.0035, 0.0026, 0.0022, 0.0036, 0.0038, 0.0036]
mrr_c = [0.0072, 0.0072, 0.0073, 0.0085, 0.0082, 0.0082]

results = compare_models(mrr_a, mrr_b, mrr_c)

import pprint
pprint.pprint(results)

dir_a = [0.4619, 0.8898, 0.9061, 0.4598, -0.8517, -0.4629]
dir_b = [0.7773, 1.35, 1.6382, 1.0754, -0.7896, -0.7658]
dir_c = [1.8217, 2.2337, 2.7944, 2.2342, -0.5477, -0.5215]

results = compare_models(mrr_a, mrr_b, mrr_c)

import pprint
pprint.pprint(results)