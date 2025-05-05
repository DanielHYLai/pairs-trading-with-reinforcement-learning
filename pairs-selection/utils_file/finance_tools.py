import numpy as np


def Mean_Return(target):
    target = np.array(target)
    result = target.mean() * 12

    return result


def Sharpe_Ratio(target, risk_free_rate):
    target = np.array(target)
    excess_return = target - risk_free_rate / 100
    mean_return = excess_return.mean() * 12
    std_return = excess_return.std() * np.sqrt(12)
    result = mean_return / std_return

    return result


def Sortino_Ratio(target, risk_free_rate):
    target = np.array(target)
    excess_return = target - risk_free_rate / 100
    mean_return = excess_return.mean() * 12
    down_std_return = np.std(excess_return[excess_return < 0]) * np.sqrt(12)
    temp_std = 0.01
    if down_std_return != 0:
        result = mean_return / down_std_return
    else:
        result = mean_return / temp_std

    return result


def Profit_Factor(target):
    target = np.array(target)
    profit = target[target > 0].sum()
    loss = np.abs(target[target < 0].sum())
    temp_loss = 0.01
    if loss != 0:
        result = profit / loss
    else:
        result = profit / temp_loss

    return result


def Max_Drawdown(target):
    target = np.array(target)
    cumulative_returns = (1 + target).cumprod()
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    result = np.min(drawdown)

    return result


def Calmar_Ratio(target, risk_free_rate):
    target = np.array(target)
    excess_return = target - risk_free_rate / 100
    mean_return = excess_return.mean() * 12
    max_dd = Max_Drawdown(target)
    temp_dd = 1
    if max_dd != 0:
        result = mean_return / abs(max_dd)
    else:
        result = mean_return / temp_dd

    return result
