import numpy as np


def mean_return(target):
    target = np.array(target)

    result = target.mean() * 12
    # print(f"Mean Return: {result: .2f}")

    return result


def sharpe_ratio(target):
    target = np.array(target)
    mean_return = target.mean() * 12
    std_return = target.std() * np.sqrt(12)

    result = mean_return / std_return
    # print(f"Sharpe Ratio: {result: .2f}")

    return result


def sortino_ratio(target):
    target = np.array(target)
    mean_return = target.mean() * 12
    down_std_return = np.std(target[target < 0]) * np.sqrt(12)
    temp_std = 0.01
    if down_std_return != 0:
        result = mean_return / down_std_return
    else:
        result = mean_return / temp_std

    # print(f"Sortino Ratio: {result: .2f}")

    return result


def profit_factor(target):
    target = np.array(target)
    profit = target[target > 0].sum()
    loss = np.abs(target[target < 0].sum())
    temp_loss = 0.01
    if loss != 0:
        result = profit / loss
    else:
        result = profit / temp_loss

    # print(f"Profit Factor: {result: .2f}")

    return result


def max_drawdown(target):
    target = np.array(target)

    cumulative_returns = (1 + target).cumprod()
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    result = np.min(drawdown)
    # print(f"Max Drawdown: {result: .2f}")

    return result


def calmar_ratio(target):
    target = np.array(target)

    annual_return = target.mean() * 12
    max_dd = max_drawdown(target)
    temp_dd = 1
    if max_dd != 0:
        result = annual_return / abs(max_dd)
    else:
        result = annual_return / temp_dd

    # print(f"Calmar Ratio: {result: .2f}")

    return result
