from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from lifelines import LogNormalAFTFitter, WeibullAFTFitter
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm
from torch import nn
from torch.nn import functional as F

######## LOG-RANK AND CONDITIONAL POWER ########

def logrank_effect(time, event, treatment) -> float:
    """Compute the marginal log-rank effect."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    treatment = np.asarray(treatment, dtype=int)
    score = 0.0
    for event_time in np.unique(time[event == 1]):
        at_risk = time >= event_time
        failures = (time == event_time) & (event == 1)
        r = int(at_risk.sum())
        r1 = int(treatment[at_risk].sum())
        d = int(failures.sum())
        d1 = int(treatment[failures].sum())
        score += d1 - d * r1 / r
    return -score / (event.sum() / 4.0)


def conditional_power(
    theta_interim: float,
    theta_future: float,
    interim_events: int,
    final_events: int,
    alpha: float = 0.05,
) -> float:
    """Combine realized IA evidence with a method-specific future effect."""
    information_fraction = interim_events / final_events
    observed_weight = np.sqrt(information_fraction) * np.sqrt(interim_events / 4.0)
    future_weight = np.sqrt(1.0 - information_fraction) * np.sqrt((final_events - interim_events) / 4.0)
    critical = norm.ppf(1.0 - alpha / 2.0)
    z = (critical - observed_weight * theta_interim - future_weight * theta_future) / np.sqrt(
        1.0 - information_fraction
    )
    return float(1.0 - norm.cdf(z))


######## METHOD RESULTS ########


@dataclass(frozen=True)
class MethodResult:
    """Method name and future-effect estimate."""

    method: str
    theta_hat: float


def result_from_predictions(
    method: str,
    data: pd.DataFrame,
    predictions: np.ndarray,
) -> MethodResult:
    """Complete only censored IA times and re-estimate the marginal effect."""
    censored = data["interim_event"].to_numpy() == 0
    predictions = np.asarray(predictions, dtype=float)
    # Replace censored times and mark completed records as events.
    augmented_time = data["interim_time"].to_numpy(float).copy()
    augmented_time[censored] = predictions
    augmented_event = np.ones(len(data), dtype=int)
    estimate = logrank_effect(augmented_time, augmented_event, data["treatment"])
    return MethodResult(method=method, theta_hat=estimate)

######## BASELINE ########

def run_baseline(data: pd.DataFrame) -> MethodResult:
    """Reuse the observed IA marginal effect as the assumed future effect."""
    estimate = logrank_effect(data["interim_time"], data["interim_event"], data["treatment"])
    return MethodResult(method="Baseline", theta_hat=estimate)

######## WEIBULL AND LOG-NORMAL AFT ########

def _linear_predictor(model, parameter: str, data: pd.DataFrame) -> np.ndarray:
    """Evaluate a fitted AFT linear predictor."""
    coefficients = model.params_.loc[parameter]
    value = np.full(len(data), float(coefficients.get("Intercept", 0.0)))
    slopes = coefficients.drop(labels="Intercept", errors="ignore")
    return value + data[list(slopes.index)].to_numpy(float) @ slopes.to_numpy(float)


def _weibull_conditional_median(
    model: WeibullAFTFitter,
    data: pd.DataFrame,
    censoring_times: np.ndarray,
) -> np.ndarray:
    """Return m satisfying H(m | x) - H(c | x) = log(2)."""
    log_scale = _linear_predictor(model, "lambda_", data)
    # Use the intercept-only Weibull shape.
    shape = np.exp(float(model.params_.loc[("rho_", "Intercept")]))
    log_hazard_at_c = shape * (np.log(censoring_times) - log_scale)
    log_hazard_at_m = np.logaddexp(log_hazard_at_c, np.log(np.log(2.0)))
    return np.exp(log_scale + log_hazard_at_m / shape)


def _lognormal_conditional_median(
    model: LogNormalAFTFitter,
    data: pd.DataFrame,
    censoring_times: np.ndarray,
) -> np.ndarray:
    """Return m satisfying S(m | x) / S(c | x) = 0.5."""
    location = _linear_predictor(model, "mu_", data)
    # Use the intercept-only log-normal scale.
    scale = np.exp(float(model.params_.loc[("sigma_", "Intercept")]))
    z_at_c = (np.log(censoring_times) - location) / scale
    z_at_m = norm.isf(0.5 * norm.sf(z_at_c))
    return np.exp(location + scale * z_at_m)


def _run_aft(
    data: pd.DataFrame,
    covariate_columns: tuple[str, ...],
    model,
    method_name: str,
    conditional_median,
) -> MethodResult:
    """Fit one parametric AFT model and impute conditional median event times."""
    features = ["treatment", *covariate_columns]
    fit_data = data[["interim_time", "interim_event", *features]].copy()
    formula = " + ".join(features)
    model.fit(fit_data, duration_col="interim_time", event_col="interim_event", formula=formula)

    censored = data["interim_event"].to_numpy() == 0
    prediction_data = data.loc[censored, features]
    censoring_times = data.loc[censored, "interim_time"].to_numpy(float)
    predictions = conditional_median(model, prediction_data, censoring_times)
    return result_from_predictions(method_name, data, predictions)


def run_weibull_aft(data: pd.DataFrame, covariate_columns: tuple[str, ...]) -> MethodResult:
    """Fit Weibull AFT and impute conditional median event times."""
    return _run_aft(
        data,
        covariate_columns,
        WeibullAFTFitter(),
        "Weibull AFT",
        _weibull_conditional_median,
    )


def run_lognormal_aft(data: pd.DataFrame, covariate_columns: tuple[str, ...]) -> MethodResult:
    """Fit log-normal AFT and impute conditional median event times."""
    return _run_aft(
        data,
        covariate_columns,
        LogNormalAFTFitter(),
        "Log-normal AFT",
        _lognormal_conditional_median,
    )

######## CQRNN ########

class CQRNN(nn.Module):
    """Three-layer censored quantile network."""

    def __init__(self, input_dim: int, hidden_units: int, quantiles: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(F.gelu(self.layer2(F.gelu(self.layer1(x)))))


def _pinball(prediction: torch.Tensor, target: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(tau * residual, (tau - 1.0) * residual)


def portnoy_loss(
    prediction: torch.Tensor,
    observed_time: torch.Tensor,
    event: torch.Tensor,
    taus: torch.Tensor,
    tail_time: float,
) -> torch.Tensor:
    """Compute observed and censored quantile losses."""
    tau = taus.view(1, -1).expand_as(prediction)
    event_mask = event.view(-1, 1)
    observed_loss = _pinball(prediction, observed_time, tau).sum(dim=1)

    # Match censoring times to the nearest nonterminal quantile without gradient.
    finite_prediction = prediction[:, :-1]
    finite_tau = tau[:, :-1]
    detached = finite_prediction.detach()
    distances = torch.abs(detached - observed_time)
    nearest = distances == torch.min(distances, dim=1, keepdim=True).values
    tau_at_censor = torch.max(finite_tau * nearest, dim=1).values.view(-1, 1)
    weights = torch.where(
        finite_tau < tau_at_censor,
        torch.ones_like(finite_tau),
        (finite_tau - tau_at_censor) / (1.0 - tau_at_censor),
    )
    censor_loss = (
        weights * _pinball(finite_prediction, observed_time, finite_tau)
        + (1.0 - weights)
        * _pinball(
            finite_prediction,
            torch.full_like(observed_time, tail_time),
            finite_tau,
        )
    ).sum(dim=1)
    return torch.mean(
        event_mask[:, 0] * observed_loss
        + (1.0 - event_mask[:, 0]) * censor_loss
    )


def _fit_original_direct_median(
    data: pd.DataFrame,
    covariate_columns: tuple[str, ...],
    model_config: dict,
    seed: int,
) -> np.ndarray:
    """Fit the three-output CQRNN and predict direct medians."""
    torch.manual_seed(seed)
    features = [*covariate_columns, "treatment"]
    x = data[features].to_numpy(np.float32)
    y = data["interim_time"].to_numpy(np.float32).reshape(-1, 1)
    event = data["interim_event"].to_numpy(np.float32)
    taus_np = np.asarray(model_config["quantiles"], dtype=np.float32)
    tail_time = float(model_config["tail_time"])
    epsilon = float(model_config.get("direct_prediction_epsilon", 1e-6))
    device = torch.device(model_config["device"])
    x_tensor = torch.tensor(x, device=device)
    y_tensor = torch.tensor(y, device=device)
    event_tensor = torch.tensor(event, device=device)
    taus = torch.tensor(taus_np, device=device)
    model = CQRNN(x.shape[1], int(model_config["hidden_units"]), len(taus_np)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_config["learning_rate"]))
    epochs = int(model_config["epochs"])
    batch_size = int(model_config["batch_size"])
    patience = int(model_config["patience"])
    best_loss = float("inf")
    counter = 0
    stagnation_state = None
    for epoch in range(epochs):
        if epoch in {int(epochs * 0.7), int(epochs * 0.9)}:
            optimizer.param_groups[0]["lr"] *= 0.1
        order = torch.randperm(len(x_tensor))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].to(device)
            loss = portnoy_loss(
                model(x_tensor[indices]),
                y_tensor[indices],
                event_tensor[indices],
                taus,
                tail_time,
            )
            epoch_loss += float(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(order)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter == 1:
                stagnation_state = model.state_dict()
            if counter >= patience:
                break
    if stagnation_state is not None:
        model.load_state_dict(stagnation_state)
    censored = data["interim_event"].to_numpy() == 0
    with torch.no_grad():
        predictions = model(x_tensor[torch.tensor(censored, device=device)])
    raw_predictions = predictions[:, 1].cpu().numpy()
    return np.maximum(raw_predictions, epsilon)


def run_cqrnn(
    data: pd.DataFrame,
    covariate_columns: tuple[str, ...],
    model_config: dict,
    seed: int,
) -> MethodResult:
    """Fit CQRNN and estimate the augmented marginal effect."""
    predictions = _fit_original_direct_median(data, covariate_columns, model_config, seed)
    return result_from_predictions("CQRNN", data, predictions)

######## YE--SHAO--YI ########

def _event_grid(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return np.unique(time[event == 1])


def _cox_score_sum(
    time: np.ndarray,
    event: np.ndarray,
    treatment: np.ndarray,
    log_hazard_ratio: float,
) -> float:
    """Treatment-only Cox score with Breslow handling of tied events."""
    score = 0.0
    for event_time in _event_grid(time, event):
        at_risk = time >= event_time
        failures = (time == event_time) & (event == 1)
        risk_treatment = np.sum(at_risk & (treatment == 1))
        risk_control = np.sum(at_risk & (treatment == 0))
        if risk_treatment == 0:
            expected_treatment_fraction = 0.0
        elif risk_control == 0:
            expected_treatment_fraction = 1.0
        else:
            expected_treatment_fraction = expit(
                log_hazard_ratio + np.log(risk_treatment) - np.log(risk_control)
            )
        score += treatment[failures].sum()
        score -= failures.sum() * expected_treatment_fraction
    return float(score)


def _solve_score(score_function) -> float:
    # Bracket log(HR) over the finite floating-point range.
    lower = float(np.log(np.finfo(float).tiny))
    upper = float(np.log(np.finfo(float).max))
    return float(brentq(score_function, lower, upper))


def _derived_outcomes(
    time: np.ndarray,
    event: np.ndarray,
    treatment: np.ndarray,
    log_hazard_ratio: float,
) -> np.ndarray:
    """Observed-arm derived outcomes in Ye et al. Section 3."""
    outcomes = np.zeros(len(time), dtype=float)
    relative_hazard = np.exp(log_hazard_ratio)
    treatment_arm = treatment == 1
    control_arm = ~treatment_arm

    for event_time in _event_grid(time, event):
        at_risk = time >= event_time
        failures = (time == event_time) & (event == 1)
        failures_at_time = failures.sum()
        risk_treatment = np.sum(at_risk & treatment_arm)
        risk_control = np.sum(at_risk & control_arm)
        denominator = relative_hazard * risk_treatment + risk_control

        outcomes[treatment_arm] += (risk_control / denominator) * (
            failures[treatment_arm].astype(float)
            - at_risk[treatment_arm].astype(float)
            * relative_hazard
            * failures_at_time
            / denominator
        )
        outcomes[control_arm] += (relative_hazard * risk_treatment / denominator) * (
            failures[control_arm].astype(float)
            - at_risk[control_arm].astype(float) * failures_at_time / denominator
        )
    return outcomes


def _arm_slopes(
    covariates: np.ndarray,
    outcomes: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Regress derived outcomes on arm-centered covariates within each arm."""
    slopes = []
    for arm in (0, 1):
        in_arm = treatment == arm
        centered = covariates[in_arm] - covariates[in_arm].mean(axis=0)
        slopes.append(np.linalg.pinv(centered.T @ centered) @ (centered.T @ outcomes[in_arm]))
    return slopes[0], slopes[1]


def run_ye2024(data: pd.DataFrame, covariate_columns: tuple[str, ...]) -> MethodResult:
    """Estimate the beneficial-direction adjusted log-hazard effect."""
    time = data["interim_time"].to_numpy(float)
    event = data["interim_event"].to_numpy(int)
    treatment = data["treatment"].to_numpy(int)
    covariates = data[list(covariate_columns)].to_numpy(float)
    # Fit the ordinary Cox score before the arm-specific adjustment.
    ordinary_log_hr = _solve_score(
        lambda value: _cox_score_sum(time, event, treatment, value)
    )
    outcomes = _derived_outcomes(time, event, treatment, ordinary_log_hr)
    beta_control, beta_treatment = _arm_slopes(covariates, outcomes, treatment)
    centered = covariates - covariates.mean(axis=0)
    adjustment = np.sum(
        treatment * (centered @ beta_treatment)
        - (1 - treatment) * (centered @ beta_control)
    )
    adjusted_log_hr = _solve_score(
        lambda value: _cox_score_sum(time, event, treatment, value) - adjustment
    )
    # Convert log(HR) to the positive-benefit convention.
    return MethodResult(method="Ye2024", theta_hat=-adjusted_log_hr)
