import numpy as np

from scipy.stats import norm

from lifelines.statistics import logrank_test
from scipy.special import digamma
import torch
import torch.nn as nn


def calculate_theta_paper(data, time_col, event_col, treatment_col, alpha=0.05):
    """
    Estimate theta and test statistics using both:
      - manuscript approximation (Var ~ n/4)
      - exact variance (sum over risk sets)

    Returns with both versions.
    """
    df = data[[time_col, event_col, treatment_col]].copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    n_events = int((df[event_col] == 1).sum())

    # Score U(0) and exact information I(0)
    U0 = 0.0
    I0 = 0.0
    r1 = int((df[treatment_col] == 1).sum())
    r0 = int((df[treatment_col] == 0).sum())

    for _, row in df.iterrows():
        x = row[treatment_col]
        if row[event_col] == 1:
            if (r0 + r1) > 0:
                U0 += x - (r1 / (r0 + r1))
                I0 += (r0 * r1) / ((r0 + r1) ** 2)  # exact variance contribution
        # update risk set
        if x == 1:
            r1 -= 1
        else:
            r0 -= 1

    # Approximate variance denominator
    denom_approx = (n_events / 4.0) if n_events > 0 else np.nan
    denom_exact = I0 if I0 > 0 else np.nan

    # --- Approximate theta (paper version)
    L_stat   = (-U0) / np.sqrt(denom_approx) if denom_approx > 0 else np.nan
    theta_hat = (-U0) / denom_approx if denom_approx > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(abs(L_stat))) if np.isfinite(L_stat) else np.nan

    crit = norm.ppf(1 - alpha/2)
    test_result = int(abs(L_stat) > crit) if np.isfinite(L_stat) else 0

    # --- Exact theta (empirical info)
    L_exact   = (-U0) / np.sqrt(denom_exact) if denom_exact > 0 else np.nan
    theta_exact = (-U0) / denom_exact if denom_exact > 0 else np.nan
    p_value_exact = 2 * (1 - norm.cdf(abs(L_exact))) if np.isfinite(L_exact) else np.nan
    test_result_exact = int(abs(L_exact) > crit) if np.isfinite(L_exact) else 0

    # lifelines log-rank for reference
    g0 = df[time_col][df[treatment_col] == 0]
    g1 = df[time_col][df[treatment_col] == 1]
    e0 = df[event_col][df[treatment_col] == 0]
    e1 = df[event_col][df[treatment_col] == 1]
    lr = logrank_test(g0, g1, event_observed_A=e0, event_observed_B=e1)

    logrank_stat = float(lr.test_statistic)
    logrank_pval = float(lr.p_value)
    test_logrank = int(logrank_pval < alpha)

    return {
        # Approximate (paper’s)
        "theta_hat": theta_hat,
        "L_stat": L_stat,
        "p_value": p_value,
        "test_result": test_result,

        # Exact (empirical info)
        "theta_exact": theta_exact,
        "L_exact": L_exact,
        "p_value_exact": p_value_exact,
        "test_result_exact": test_result_exact,

        # Reference: lifelines
        "logrank_stat": logrank_stat,
        "logrank_pval": logrank_pval,
        "test_result_logrank": test_logrank,

        # Bookkeeping
        "n_events": n_events,
        "U0": U0,
        "I0_exact": I0
    }


def calculate_theta(data, time_col, event_col, treatment_col, weight_factor):

    """
    Perform log-rank test and calculate theta.
    
    Args:
        data (DataFrame): The dataset containing survival times, events, and treatments.
        time_col (str): The column name for survival time.
        event_col (str): The column name for event indicator (1 = event, 0 = censored).
        treatment_col (str): The column name for treatment indicator (1 = treatment, 0 = control).
        weight_factor (float): The factor to calculate theta scaling (e.g., 4 / sample_size).
    
    Returns:
        float: Theta value calculated from the log-rank test.
    """
    logrank_result = logrank_test(
        data[time_col][data[treatment_col] == 0],
        data[time_col][data[treatment_col] == 1],
        event_observed_A=data[event_col][data[treatment_col] == 0],
        event_observed_B=data[event_col][data[treatment_col] == 1]
    )

    standard_normal_stat = np.sqrt(logrank_result.test_statistic)
    if data[time_col][data[treatment_col] == 1].mean() < data[time_col][data[treatment_col] == 0].mean():
        standard_normal_stat = -standard_normal_stat

    theta = standard_normal_stat * np.sqrt(weight_factor)
    crit = norm.ppf(1 - 0.025)
    test_result = int(standard_normal_stat > crit)
    return theta



def CP(n_final, n_interim, alpha, theta_IA, theta_IC):
    za = norm.ppf(1 - alpha / 2)
    t = n_interim / n_final

    weights_IA = np.sqrt(t) / np.sqrt(4 / n_interim)
    weights_IC = np.sqrt(1 - t) / np.sqrt(4 / (n_final - n_interim))

    CP = 1 - norm.cdf(
        (za - weights_IA * theta_IA - weights_IC * theta_IC) /
        np.sqrt(1 - t)
    )
    return CP



def train_CQRNN_PPI(
    data_organized,
    covariate_cols,
    n_dummy_cols=0,
    y_max=200,
    n_epochs=200,
    batch_size=64,
    n_hidden=64,
    device='cpu',
    seed=None
    ):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    taus = [0.10, 0.5, 0.95]
    n_quantiles = len(taus)


    if covariate_cols:
        x_train = np.column_stack([data_organized[col] for col in covariate_cols] +
                                  [data_organized['treatment']])
    else:
        x_train = data_organized[['treatment']].to_numpy()
    y_train = data_organized['Y_ob_interim'].to_numpy().reshape(-1, 1)
    cen_indicator = (1 - data_organized['event_IA']).to_numpy().reshape(-1, 1)

    x_train_torch = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
    cen_indicator_torch = torch.tensor(cen_indicator).float().to(device)
    taus_torch = torch.tensor(taus).reshape(1, -1).float().to(device)

    input_dim = x_train_torch.shape[1]
    model_original = Model_mlp(input_dim, n_hidden, n_quantiles).to(device)
    optimizer = torch.optim.Adam(model_original.parameters(), lr=0.01)

    best_loss = float("inf")
    counter = 0
    patience = 10
    stagnation_model_state = None

    for ep in range(n_epochs):
        if ep == int(n_epochs * 0.7):
            optimizer.param_groups[0]['lr'] /= 10
        if ep == int(n_epochs * 0.9):
            optimizer.param_groups[0]['lr'] /= 10

        permutation = torch.randperm(x_train_torch.size()[0])
        epoch_loss = 0.0

        for i in range(0, x_train_torch.size()[0], batch_size):
            indices_batch = permutation[i:i+batch_size]
            x_batch = x_train_torch[indices_batch]
            y_batch = y_train_torch[indices_batch]
            cen_batch = cen_indicator_torch[indices_batch]
            y_pred = model_original(x_batch)
            loss = cqrnn_loss(y_pred, y_batch, cen_batch, taus, taus_torch, y_max)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(permutation)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter == 1:
                stagnation_model_state = model_original.state_dict()

        if counter >= patience:
            break

    if stagnation_model_state is not None:
        model_original.load_state_dict(stagnation_model_state)

    model_original.eval()
    with torch.no_grad():
        y_pred_all_quantiles = model_original(x_train_torch)
    y_pred_median_np = y_pred_all_quantiles[:, 1].detach().cpu().numpy()
    data_organized['y_pred_median_np'] = y_pred_median_np

    data_organized['Y_CQRNN'] = np.where(
        data_organized['event_IA'] == 1,
        data_organized['Y_ob_interim'],
        data_organized['y_pred_median_np']
    )


    CQRNN_result = calculate_theta_paper(
            data_organized, 'Y_CQRNN', 'event_GT', 'treatment'
        )
    theta_CQRNN = CQRNN_result['theta_hat']
    theta_CQRNN_exact = CQRNN_result['theta_exact']

    data_organized['Y_CQRNN_maxob'] = data_organized['Y_ob_interim']
    max_ob = data_organized.loc[data_organized['event_IA'] == 1, 'Y_ob_interim'].max()
    should_impute = (data_organized['event_IA'] == 0) & \
                    (data_organized['y_pred_median_np'] <= max_ob)
    data_organized.loc[should_impute, 'Y_CQRNN_maxob'] = \
        data_organized.loc[should_impute, 'y_pred_median_np']
    data_organized['event_CQRNN_maxob'] = data_organized['event_IA'].copy()
    data_organized.loc[should_impute, 'event_CQRNN_maxob'] = 1

    n_events_CQRNN = data_organized['event_CQRNN_maxob'].sum()
    theta_CQRNN_maxob = calculate_theta(
        data_organized,
        'Y_CQRNN_maxob',
        'event_CQRNN_maxob',
        'treatment',
        4 / n_events_CQRNN
    )
    

    return {
        'data_organized': data_organized,
        'theta_CQRNN': theta_CQRNN,
        'theta_CQRNN_exact': theta_CQRNN_exact,
        'theta_CQRNN_maxob': theta_CQRNN_maxob
    }




def quantile_loss(y_pred, y_true, cen_indicator, taus, taus_torch):
	tau_block = taus_torch.repeat((cen_indicator.shape[0],1)) # need this stacked in shape (n_batch, n_quantiles)
	loss = torch.sum((cen_indicator<1)*(y_pred  - y_true)*((1-tau_block)-1.*(y_pred<y_true)),dim=1)
	loss = torch.mean(loss)
	return loss


class Model_mlp(nn.Module):
    def __init__(self, input_dim, n_hidden, n_quantiles):
        super(Model_mlp, self).__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.layer2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.layer3 = nn.Linear(n_hidden, n_quantiles, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.gelu(x)
        x = self.layer2(x)
        x = torch.nn.functional.gelu(x)
        x = self.layer3(x)
        return x
    
    
def cqrnn_loss(y_pred, y_true, cen_indicator, taus, taus_torch, y_max):
	# this is CQRNN loss as in paper
	# we've taken care to implement the loss without for loops, so things can be parallelised quickly
	# but the downside is that this becomes harder to read and match up with the description in the paper
	# so we also include cqrnn_loss_slowforloops() 
	# just note they both do the same thing

	y_pred_detach = y_pred.detach()
	tau_block = taus_torch.repeat((cen_indicator.shape[0],1)) # need this stacked in shape (n_batch, n_quantiles), 
	loss_obs = quantile_loss(y_pred, y_true, cen_indicator, taus, taus_torch)

	# use argmin to get nearest quantile
	torch_abs = torch.abs(y_true - y_pred_detach[:,:-1]) # ignore the final quantile, which represents 1.0, so use [:-1]
	estimated_quantiles = torch.max(tau_block[:,:-1]*(torch_abs==torch.min(torch_abs,dim=1).values.view(torch_abs.shape[0],1)),dim=1).values
		
	# compute weights, eq 11, portnoy 2003, want weights to be in shape (batch_size x n_quantiles-1)
	weights = (tau_block[:,:-1]<estimated_quantiles.reshape(-1,1))*1. + (tau_block[:,:-1]>=estimated_quantiles.reshape(-1,1))*(tau_block[:,:-1]-estimated_quantiles.reshape(-1,1))/(1-estimated_quantiles.reshape(-1,1))

	# now compute censored loss using weight* censored value, + (1-weight)* fictionally large value
	loss_cens = torch.sum((cen_indicator>0) * \
						  (weights * (y_pred[:,:-1]  - y_true)*((1-tau_block[:,:-1])-1.*(y_pred[:,:-1]<y_true)) + \
				          (1-weights)*(y_pred[:,:-1]  - y_max )*((1-tau_block[:,:-1])-1.*(y_pred[:,:-1]<y_max ))) \
				          ,dim=1)
	loss_cens = torch.mean(loss_cens)

	return loss_obs + loss_cens