""" A) generate meta-training and meta-testing data """
from experiments.data_sim import SinusoidDataset
task_environment = SinusoidDataset()
meta_train_data = task_environment.generate_meta_train_data(n_tasks=20, n_samples=5)
meta_test_data = task_environment.generate_meta_test_data(n_tasks=20, n_samples_context=5, n_samples_test=50)


""" B) Meta-Learning with PACOH-MAP """
from meta_learn import SparseGPRegressionMetaLearned
meta_gp = SparseGPRegressionMetaLearned(meta_train_data, weight_decay=0.2)
meta_gp.meta_fit(meta_test_data, log_period=1000)


"""  C) Meta-Testing with PACOH-MAP """
x_context, y_context, x_test, y_test = meta_test_data[0]

# target training in (x_ontext, y_context) & predictions for x_test
pred_mean, pred_std = meta_gp.predict(x_context, y_context, x_test)

# confidence intervals predictions in x_test 
ucb, lcb = meta_gp.confidence_intervals(x_context, y_context, x_test, confidence=0.9)

# compute evaluation metrics on one target task
log_likelihood, rmse, calib_error = meta_gp.eval(x_context, y_context, x_test, y_test)

# compute evaluation metrics for multiple tasks / test datasets
log_likelihood, rmse, calib_error = meta_gp.eval_datasets(meta_test_data)
# Do it again with Cauchy

""" A) generate meta-training and meta-testing data """
from experiments.data_sim import CauchyDataset
task_environment = CauchyDataset()
meta_train_data = task_environment.generate_meta_train_data(n_tasks=20, n_samples=5)
meta_test_data = task_environment.generate_meta_test_data(n_tasks=20, n_samples_context=5, n_samples_test=50)


""" B) Meta-Learning with PACOH-MAP """
meta_gp = SparseGPRegressionMetaLearned(meta_train_data, weight_decay=0.2)
meta_gp.meta_fit(meta_test_data, log_period=1000)


"""  C) Meta-Testing with PACOH-MAP """
x_context, y_context, x_test, y_test = meta_test_data[0]

# target training in (x_ontext, y_context) & predictions for x_test
pred_mean, pred_std = meta_gp.predict(x_context, y_context, x_test)

# confidence intervals predictions in x_test 
ucb, lcb = meta_gp.confidence_intervals(x_context, y_context, x_test, confidence=0.9)

# compute evaluation metrics on one target task
log_likelihood, rmse, calib_error = meta_gp.eval(x_context, y_context, x_test, y_test)

# compute evaluation metrics for multiple tasks / test datasets
log_likelihood, rmse, calib_error = meta_gp.eval_datasets(meta_test_data)

print(f"For the Cauchy dataset, the RMSE is {rmse:.2f} and the calibration error is {calib_error:.2f}, with a log-likelihood of {log_likelihood:.2f}")
