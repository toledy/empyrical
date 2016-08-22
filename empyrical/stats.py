#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import pandas as pd
import numpy as np
from scipy import stats

from empyrical.utils import nanmean


APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    YEARLY: 1
}


def _adjust_returns(returns, adjustment_factor):
    """
    Returns a new pd.Series adjusted by adjustment_factor. Optimizes for the
    case of adjustment_factor being 0

    Parameters
    ----------
    returns : pd.Series
    adjustment_factor : series / float

    Returns
    -------
    pd.Series
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns.copy()
    return returns - adjustment_factor


def annualization_factor(period, annualization):
    """
    Return annualization factor from period entered or if a custom
    value is passed in.

    Parameters
    ----------
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str
    :param annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: str, optional

    :rtype: float
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def cum_returns(returns, starting_value=0):
    """Compute cumulative returns from simple returns.

    :param pandas.Series returns: Returns of the strategy as a percentage,
        noncumulative. Time series with decimal returns. Example::
          2015-07-16   -0.012143
          2015-07-17    0.045350
          2015-07-18    0.030957
          2015-07-19    0.004902
          Freq: D, dtype: float64
    :param starting_value: The starting returns.
    :type starting_value: float, optional

    :rtype: pandas.Series
    """

    # df_price.pct_change() adds a nan in first position, we can use
    # that to have cum_logarithmic_returns start at the origin so that
    # df_cum.iloc[0] == starting_value
    # Note that we can't add that ourselves as we don't know which dt
    # to use.

    if len(returns) < 1:
        return np.nan

    if pd.isnull(returns.iloc[0]):
        returns = returns.copy()
        returns.iloc[0] = 0.

    # For increased numerical accuracy, input is converted to log returns
    # where it is possible to sum instead of multiplying.
    # PI((1+r_i)) - 1 = exp(ln(PI(1+r_i)))     # x = exp(ln(x))
    #                 = exp(SIGMA(ln(1+r_i))   # ln(a*b) = ln(a) + ln(b)
    df_cum = np.exp(np.log1p(returns).cumsum())

    if starting_value == 0:
        return df_cum - 1
    else:
        return df_cum * starting_value


def aggregate_returns(returns, convert_to):
    """Aggregates returns by week, month, or year.

    :param returns: Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :type returns: pandas.Series
    :param convert_to: Can be 'weekly', 'monthly', or 'yearly'.
    :type convert_to: str

    :rtype: pandas.Series
    """

    def cumulate_returns(x):
        return cum_returns(x)[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns):
    """Determines the maximum drawdown of a strategy.

    :param pandas.Series returns: Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.

    :rtype: float

    See `wikipedia <https://en.wikipedia.org/wiki/Drawdown_(economics)>`_
    for more details.
    """

    if len(returns) < 1:
        return np.nan

    cumulative = cum_returns(returns, starting_value=100)
    max_return = cumulative.cummax()
    return cumulative.sub(max_return).div(max_return).min()


def annual_return(returns, period=DAILY, annualization=None):
    """Determines the mean annual growth rate of returns.

    :param pandas.Series returns:
        Periodic returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param period: Defines the periodicity of the 'returns' data for purposes
        of annualizing. Value ignored if `annualization` parameter is
        specified. Defaults are::
            {'monthly': 12,
             'weekly': 52,
             'daily': 252}
    :type period: str, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: int, optional

    :rtype: float
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    num_years = float(len(returns)) / ann_factor
    start_value = 100
    end_value = cum_returns(returns, starting_value=start_value).iloc[-1]
    total_return = (end_value - start_value) / start_value
    annual_return = (1. + total_return) ** (1. / num_years) - 1

    return annual_return


def annual_volatility(returns, period=DAILY, alpha=2.0,
                      annualization=None):
    """Determines the annual volatility of a strategy.

    :param pandas.Series returns:
        Periodic returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param alpha:
        Scaling relation (Levy stability exponent).
    :type alpha: float, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: int, optional

    :rtype: float
    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    volatility = returns.std() * (ann_factor ** (1.0 / alpha))

    return volatility


def calmar_ratio(returns, period=DAILY, annualization=None):
    """Determines the calmar ratio, or drawdown ratio, of a strategy.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.


    :rtype: float

    See `wikipedia <https://en.wikipedia.org/wiki/Calmar_ratio>`_
    for more details.
    """

    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, risk_free=0.0, required_return=0.0,
                annualization=APPROX_BDAYS_PER_YEAR):
    """Determines the omega ratio of a strategy.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param risk_free:
        Constant risk-free return throughout the period
    :type risk_free: int, float
    :param required_return:
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.
    :type required_return: float, optional
    :param annualization:
        Factor used to convert the required_return into a daily
        value. Enter 1 if no time period conversion is necessary.
    :type annualization: int, optional

    :rtype: float

    See `wikipedia <https://en.wikipedia.org/wiki/Omega_ratio>`_
    for more details.
    """

    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** \
            (1. / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns, risk_free=0, period=DAILY, annualization=None):
    """
    Determines the sharpe ratio of a strategy.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param risk_free:
        Constant risk-free return throughout the period.
    :type risk_free: int, float
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: int, optional

    :rtype: float or numpy.nan

    See `wikipedia <https://en.wikipedia.org/wiki/Sharpe_ratio>`_
    for more details.
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    returns_risk_adj = _adjust_returns(returns, risk_free).dropna()

    if np.std(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * \
        np.sqrt(ann_factor)


def sortino_ratio(returns, required_return=0, period=DAILY,
                  annualization=None, _downside_risk=None):
    """
    Determines the sortino ratio of a strategy.

    :param returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :type returns: pandas.Series or pandas.DataFrame
    :param required_return:
        Minimum acceptable return
    :type required_return: float or pandas.Series
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: int, optional
    :param _downside_risk : float, optional
        The downside risk of the given inputs, if known. Will be calculated if
        not provided.
    :type _downside_risk: float, optional

    :returns: Depends on input type
    :rtype: pandas.Series ==> float
    :rtype: pandas.DataFrame ==> pandas.Series
    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    if len(returns) < 2:
        return np.nan

    adj_returns = _adjust_returns(returns, required_return)
    mu = nanmean(adj_returns, axis=0)
    dsr = (_downside_risk if _downside_risk is not None
           else downside_risk(returns, required_return))
    sortino = mu / dsr
    if len(returns.shape) == 2:
        sortino = pd.Series(sortino, index=returns.columns)
    return sortino * ann_factor


def downside_risk(returns, required_return=0, period=DAILY,
                  annualization=None):
    """Determines the downside deviation below a threshold

    :param returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :type returns: pandas.Series or pandas.DataFrame
    :param required_return:
        Minimum acceptable return
    :type required_return: float or pandas.Series
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: int, optional

    :returns: depends on input type
    :rtype: pandas.Series ==> float
    :rtype: pandas.DataFrame ==> pandas.Series
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    downside_diff = _adjust_returns(returns, required_return)
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)
    if len(returns.shape) == 2:
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk


def information_ratio(returns, factor_returns):
    """Determines the Information ratio of a strategy.

    :param returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :type returns: pandas.Series of pandas.Dataframe
    :param factor_returns:
        Benchmark return to compare returns against.
    :type factor_returns: float or pandas.Series

    :rtype: float

    See `wikipedia <https://en.wikipedia.org/wiki/information_ratio>`_
    for more details.
    """

    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


def alpha_beta(returns, factor_returns, risk_free=0.0, period=DAILY,
               annualization=None):
    """Calculates annualized alpha and beta.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param pandas.Series factor_returns:
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         This is in the same style as returns.
    :param risk_free:
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    :type risk_free: int, float, optional
    :param period:
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :type annualization: int, optional

    :rtype: (float, float)
    """

    b = beta(returns, factor_returns, risk_free)
    a = alpha(returns, factor_returns, risk_free, period, annualization,
              _beta=b)
    return a, b


def alpha(returns, factor_returns, risk_free=0.0, period=DAILY,
          annualization=None, _beta=None):
    """Calculates annualized alpha.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param pandas.Series factor_returns:
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         This is in the same style as returns.
    :param risk_free:
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    :type risk_free: int, float, optional
    :param period: Defines the periodicity of the 'returns' data for purposes
        of annualizing. Value ignored if `annualization` parameter is
        specified. See details in :func:`~empyrical.annual_return`.
    :type period: str, optional
    :param annualization:
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~empyrical.stats.annual_return`.
    :type annualization: int, optional
    :param _beta:
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.
    :type _beta: float, optional

    :rtype: float
    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    if _beta is None:
        b = beta(returns, factor_returns, risk_free)
    else:
        b = _beta

    adj_returns = _adjust_returns(returns, risk_free)
    adj_factor_returns = _adjust_returns(factor_returns, risk_free)
    alpha_series = adj_returns - (b * adj_factor_returns)

    return alpha_series.mean() * ann_factor


def beta(returns, factor_returns, risk_free=0.0):
    """Calculates beta.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.
    :param pandas.Series factor_returns:
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         This is in the same style as returns.
    :param risk_free:
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    :type risk_free: int, float, optional

    :rtype: float
    """

    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan
    # Filter out dates with np.nan as a return value
    joint = pd.concat([_adjust_returns(returns, risk_free),
                       factor_returns], axis=1).dropna()
    if len(joint) < 2:
        return np.nan

    if np.absolute(joint.var().iloc[1]) < 1.0e-30:
        return np.nan

    return np.cov(joint.values.T, ddof=0)[0, 1] / np.var(joint.iloc[:, 1])


def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.

    :rtype: float
    """

    if len(returns) < 2:
        return np.nan

    returns = returns.dropna()

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns.values)[2]

    return rhat ** 2


def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    :param pandas.Series returns:
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.cum_returns`.

    :rtype: float
    """

    if len(returns) < 1:
        return np.nan

    # Be tolerant of nan's
    returns = returns.dropna()
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def cagr(returns, period=DAILY, annualization=None):
    """
    Compute compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~empyrical.stats.annual_return`.

    Returns
    -------
    float, np.nan
        The CAGR value.

    """
    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    no_years = len(returns) / float(ann_factor)
    ending_value = cum_returns(returns, starting_value=1).iloc[-1]

    return ending_value ** (1. / no_years) - 1


SIMPLE_STAT_FUNCS = [
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio,
    cagr
]

FACTOR_STAT_FUNCS = [
    information_ratio,
    alpha,
    beta,
]
