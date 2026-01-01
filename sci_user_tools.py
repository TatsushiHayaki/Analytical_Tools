"""
Last update: Jan-1-2026
Owner: Tatsushi Hayaki

# ToDo: Make 'Q1', 'Q3', 'Median', 'Kurt', 'Skew', 'Range', and other options available
# ToDo: Address issues for datetime columns
"""
__version__ = '1.2.10'
import numpy as np
import pandas as pd
import os
from functools import wraps
from inspect import signature

def print_control(func):
    """
    Decorator to control printed output for proc_means() and proc_freq().
    When noprint=False is specified (by default), the wrapped function
    will print the result(s) in a well-formatted manner.
    """
    # def nan_ints(df):
    #     """
    #     Converts float/object columns to nullable integer types
    #     (i.e., pandas.Int64Dtype) for display when safe to do so.
    #     Columns containing decimal values will not be converted
    #     even if they have a float data type.
    #     """
    #     df = df.copy()
    #     varlist = list(df.columns)
    #     varlist = [v for v in varlist if v not in ('MIN', 'MAX')] # excluding columns you don't apply the function to
    #     for var in varlist:
    #         is_float = 'float' in str(df[var].dtype)
    #         is_object = 'object' in str(df[var].dtype)
    #         if (is_float or is_object):
    #             df[var] = df[var].astype("Int64", errors='ignore')
    #     return df
    def nan_ints(df):
        df = df.copy()
        varlist = list(df.columns)
        varlist = [v for v in varlist if v not in ('MIN', 'MAX')] # excluding columns you don't apply the function to
        for var in varlist:
            s = pd.to_numeric(df[var], errors='coerce')
            if s.notna().any():
                if (s.dropna() % 1 == 0).all():
                    df[var] = s.astype("Int64")
                else:
                    df[var] = s
            else:
                df[var] = s
        return df

    list_output = True
    def print_grouped(indata, group_cols, recursion_level=0):
        """
        Recursively print a DataFrame grouped by multiple columns.
        """
        if recursion_level == len(group_cols):
            if not list_output:
                print(nan_ints(indata)) # Prefer not to suppress indices when list_output is False in proc_freq().
                print()
                return
            print(nan_ints(indata).to_string(index=False))
            print()
            return
        grp = group_cols[recursion_level]
        for cat in indata[grp].unique():
            print(f"---- {grp}: {cat} ----")
            temp = indata[indata[grp] == cat].drop(columns=[grp])
            print_grouped(temp, group_cols, recursion_level+1)

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # Collecting arguments for the wrapped function
        bound = signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        noprint = bound.arguments['noprint']
        byvar = bound.arguments['byvar']
        nonlocal list_output
        list_output = kwargs.pop('list_output', True)
        if noprint:
            return result
        if not list_output:
            if byvar:
                group_cols = list(result.filter(regex='Group').columns)
                print_grouped(result, group_cols, 0)
                return
            print(nan_ints(result)) # Prefer not to suppress indices when list_output is False in proc_freq().
            return
        if byvar:
            group_cols = list(result.filter(regex='Group').columns)
            print_grouped(result, group_cols, 0)
            return
        print(nan_ints(result).to_string(index=False))
    return wrapper

@print_control
def proc_means(indata, varlist, byvar=None, weight=None, where=None,
               items=['N', 'NMISS', 'MEAN', 'STDDEV', 'SE', 'MIN', 'MAX'],
               vardef="WDF", noprint=False
               ):

    """
    Mimics the behavior of the standard `PROC MEANS` procedure in SAS
    (currently does not support CLASS and other statement).

    Available statistic keywords for the `items` parameter:
    ['N', 'NMISS', 'MEAN', 'VAR', 'STDDEV', 'SE', 'MIN', 'MAX'].

    Parameters:
    -----------
    indata : DataFrame
        The input DataFrame.
    varlist : list
        List of variables for which descriptive statistics are computed.
    byvar : str/list, optional
        A categorical variable by which the procedure is carried out.
    weight : str, optional
        Column name for the weight variable.
    where : str, optional
        Query the columns of a DataFrame with a boolean expression.
    items : list, optional
        List of statistics to compute (e.g., ['MEAN', 'VAR']).
    vardef : str, optional
        One of {"DF", "N", "WDF", "WGT"}; specifies the divisor used in variance calculation.
    noprint : bool, optional
        If True, suppresses output printing and returns the result as a DataFrame.

    Example:
    --------
    >>> proc_means(df, ['rating01', 'rating02', 'rating03'], byvar='segment', weight='wt', where='(filter_a==2) & (filte_b==1)')
    This is equivalent to the following SAS procedure:
        proc means N NMISS MEAN VAR STDDEV MIN MAX data=df;
            var rating01 rating02 rating03;
            weight wt;
            by segment;
            where filter_a==2 AND filter_b==1;
        run;

    Note:
    -----
    For the computation of weighted standard error, the keyword `STDERR` in `PROC MEANS` in SAS
    is available only when `vardef=`DF`. In contrast, by specifying `vardef=WDF` or `vardef=WGT`,
    this function computes the standard error by dividing the variance by the number of observations
    rather than by the sum of weights. In SPSS, by default the weighted standard error is defined by
    sqrt(variance / sum(weight)), which we find often yields unrealistic results.

    If the weight variable contains missing values, weighted statistics may not be computed correctly.
    """
    indata = indata.copy()

    if weight is None:
        weight = '_one_'
        indata[weight] = 1
    if where:
        indata = indata.query(where)
    if isinstance(varlist, str):
        varlist = [varlist]
    if byvar is None:
        byvar = []
    elif isinstance(byvar, str):
        byvar = [byvar]

    def statistics(indata, varlist, group_label='ALL'):
        # outdata = pd.DataFrame(columns=['Group', 'Variable', 'N', 'NMISS', 'MEAN', 'VAR', 'STDDEV', 'SE', 'MIN', 'MAX'])
        outdata = pd.DataFrame()
        for var in varlist:
            ma = np.ma.MaskedArray(indata[var], mask=indata[var].isna())
            w  = np.ma.MaskedArray(indata[weight], mask=indata[var].isna())
            mean_wtd = np.ma.average(ma, weights=indata[weight])

            # Computation of Variance
            CSS = (w * (ma - mean_wtd) **2).sum()
            if   vardef  == "DF":
                divisor  = ma.count()-1
                variance = CSS/divisor
                stderr   = np.sqrt(variance/w.sum())
            elif vardef  == "N":
                divisor  = ma.count()
                variance = CSS/divisor
                stderr   = np.sqrt(variance/w.sum())
            elif vardef  == "WDF":
                divisor  = w.sum()-1
                variance = CSS/divisor
                stderr   = np.sqrt(variance/ma.count()) # see the docstring
            elif vardef  == "WGT":
                divisor  = w.sum()
                variance = CSS/divisor
                stderr   = np.sqrt(variance/ma.count()) # see the docstring
            else:
                raise ValueError("Invalid vardef argument specified.")

            raw_data = pd.DataFrame([{
                 'Group'    : group_label,
                 'Variable' : var,
                 'N'        : indata[var].count(),
                 'NMISS'    : indata[var].isna().sum(),
                 'MEAN'     : mean_wtd,
                 'VAR'      : variance,
                 'STDDEV'   : np.sqrt(variance),
                 'SE'       : stderr,
                 'MIN'      : indata[var].min(),
                 'MAX'      : indata[var].max(),
                 }])
            outdata = pd.concat([outdata, raw_data], axis=0, ignore_index=True)
            outdata = outdata[['Group', 'Variable']+items]
        return outdata

    if not byvar:
        result = statistics(indata, varlist)[['Group', 'Variable'] + items]
        return result

    # out = pd.DataFrame(columns=['Group','Variable', 'N', 'NMISS', 'MEAN', 'VAR', 'STDDEV', 'SE', 'MIN', 'MAX'])
    out = pd.DataFrame()
    grouped = indata.groupby(byvar, dropna=False, sort=True)
    for keys, sub in grouped:
        if len(byvar) == 1:
            group_label = str(keys)
        elif len(byvar) > 1:
            group_label = '|'.join([str(v) for v in keys])
        subout = statistics(sub, varlist, group_label=group_label)
        out = pd.concat([out, subout], axis=0, ignore_index=True)
    if len(byvar) == 1:
        result = out[['Group', 'Variable'] + items]
    elif len(byvar) > 1:
        out[[f'Group{i+1}' for i in range(len(byvar))]] = out['Group'].str.split('|', expand=True)
        group_cols = list(out.filter(regex='Group\d+').columns)
        result = out[group_cols + ['Variable'] + items]
    return result


@print_control
def proc_freq(indata, varlist, byvar=None, weight=None, where=None,
              list_output=True, noprint=False
              ):
    """
    Mimics the behavior of the standard `PROC FREQ` procedure in SAS
    (currently does not support CLASS and other statement).
    Missing values are included in the output by default.

    Parameters:
    -----------
    indata : DataFrame
        The input DataFrame.
    varlist : list
        List of variables for which frequencies are calculated.
    byvar : str/list, optional
        A categorical variable by which the procedure is carried out.
    where : str, optional
        Query the columns of a DataFrame with a boolean expression.
    list_output : bool, optional
        Corresponds to the "/list" option in SAS. If False, returns a two-way cross-tabulation instead.
    noprint : bool, optional
        If True, suppresses printing and returns the output as a DataFrame.

    Example:
    --------
    >>> proc_freq(df, ['var1', 'var2', 'var3'], byvar='segment', where='(filter_a==2) & (filter_b==1)')
    This is equivalent to the following SAS procedure:
        proc freq data=df;
            tables var1 * var2 * var3 /list missing;
            by segment;
            where filter_a=2 AND filter_b=1;
        run;
    '''
    Note: If a weight is missing, the corresponding observation is excluded from the calculation.
    """
    indata = indata.copy()
    if weight is None:
        weight = '_one_'
        indata[weight] = 1
    if where:
        indata = indata.query(where)
    if isinstance(varlist, str):
        varlist = [varlist]
    if byvar is None:
        byvar = []
    elif isinstance(byvar, str):
        byvar = [byvar]

    def frequency(indata, varlist, group_label='ALL'):
        if weight == '_one_':
            result = (indata
                .groupby(varlist, dropna=False)
                .size()
                .reset_index(name='Frequency')
                .sort_values(varlist, na_position='first')
            )
        else:
            indata = indata[~indata[weight].isna()]
            w = indata[weight].values
            result = pd.DataFrame()
            result[weight+"_masked"] = w
            for var in varlist:
                ma = np.ma.MaskedArray(indata[var], mask=indata[var].isna())
                result[var] = ma
                result[var+"_wtd"] = (ma * w).filled(np.nan)
            result = (result
                .groupby(varlist, dropna=False)[weight+"_masked"]
                .sum()
                .reset_index(name="Frequency")
                .sort_values(varlist, na_position="first")
            )
        result['Percent'] = (result['Frequency'] / result['Frequency'].sum()*100)
        result['Cumulative_Frequency'] = result['Frequency'].cumsum()
        result['Cumulative_Percent'] = result['Percent'].cumsum()
        result['Group'] = group_label
        result = result[['Group']+varlist+['Frequency', 'Percent', 'Cumulative_Frequency', 'Cumulative_Percent']]
        return result

    def two_way_crosstab(indata, varlist, group_label='ALL'):
        assert len(varlist) == 2, "Sorry, only two-way cross-tabulation supported."
        if weight == '_one_':
            result = (indata
                    .groupby(varlist, dropna=False)
                    .size()
                    .reset_index(drop=False)
                    .sort_values(by=varlist, na_position='first')
                    .set_index(varlist)[0] # adding [0] to make it a Series
                    .unstack(fill_value=0)
                    )
        else:
            indata = indata[~indata[weight].isna()]
            w = indata[weight].values
            result = pd.DataFrame()
            result[weight+"_masked"] = w
            for var in varlist:
                ma = np.ma.MaskedArray(indata[var], mask=indata[var].isna())
                result[var] = ma
                result[var+"_wtd"] = (ma * w).filled(np.nan)
            result = (result
                    .groupby(varlist, dropna=False)[weight+"_masked"]
                    .sum()
                    .reset_index(drop=False)
                    .sort_values(by=varlist, na_position='first')
                    .set_index(varlist)[weight+"_masked"] # adding [weight+"_masked"] to make it a Series
                    .unstack(fill_value=0)
                    )
        result['Total'] = result.sum(axis=1)
        result.loc['Total'] = result.sum(axis=0)
        return result

    # Applying frequency function
    if list_output:
        if not byvar:
            result = frequency(indata, varlist).drop(columns=['Group']).reset_index(drop=True)
            return result

        # out = pd.DataFrame(columns=['Group']+varlist+['Frequency', 'Percent', 'Cumulative_Frequency', 'Cumulative_Percent'])
        out = pd.DataFrame()
        grouped = indata.groupby(byvar, dropna=False, sort=True)
        for keys, sub in grouped:
            if len(byvar) == 1:
                group_label = str(keys)
            elif len(byvar) > 1:
                group_label = '|'.join([str(v) for v in keys])
            subout = frequency(sub, varlist, group_label=group_label)
            out = pd.concat([out, subout], axis=0, ignore_index=True)
        if len(byvar) == 1:
            result = out
        elif len(byvar) > 1:
            out[[f'Group{i+1}' for i in range(len(byvar))]] = out['Group'].str.split('|', expand=True)
            group_cols = list(out.filter(regex='Group\d+').columns)
            result = out[group_cols + varlist + ['Frequency', 'Percent', 'Cumulative_Frequency', 'Cumulative_Percent']]
        return result

    # Applying two_way_crosstab function
    else:
        if not byvar:
            result = two_way_crosstab(indata, varlist)
            return result
        assert len(byvar) == 1, 'Sorry, multiple byvar argument not supported for two-way crosstab.'
        byvar = byvar[0]
        out = pd.DataFrame()
        group_list = indata[byvar].unique()
        for key in group_list:
            sub = indata.loc[indata[byvar]==key]
            group_label = str(key)
            subout = two_way_crosstab(sub, varlist, group_label=group_label)
            subout['Group'] = group_label
            out = pd.concat([out, subout], axis=0, ignore_index=False)
        result = out
        return result


def grab_datainfo(indata, metadata=None, output_dir=False, filename='datainfo.txt'):
    """
    Motivated by ``info`` function in ``expss`` package for R, this function provides
    variables description for dataset and store the information to your specified location.

    The assumed situations are either of the following:
    (1) an SPSS dataset along with its metadata obtained by ``pyreadstat`` module, or
    (2) any Pandas DataFrame object without metadata.

    Example:
    --------
    >>> grab_datainfo(tab_delimited_text_data, json_metadata, output_dir=r'Output/', filename='datainfo.txt')
    >>> grab_datainfo(SPSS_data, SPSS_metadata, output_dir=r'Output/', filename='datainfo.txt')
    """

    df = indata.copy()
    # extension = os.path.splitext(indata)[1]
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir += filename

    if metadata:
        print('-- analyzing SPSS dataset with its metadata...')
        output                = pd.DataFrame()
        output['Class']       = pd.Series(df.dtypes)
        output['Length']      = len(df)
        output['NA']          = df.isna().sum()
        output['NotNA']       = output['Length'] - output['NA']
        output['Distinct']    = df.nunique(dropna=False).to_frame()
        output['Label']       = pd.Series(metadata.column_names_to_labels).to_frame()
        output['ValueLabels'] = output.index.map(metadata.variable_value_labels)
        output['Min']         = df.min(numeric_only=True)
        output['1st Qu.']     = df.select_dtypes(include=[np.number]).quantile(0.25)
        output['Mean']        = df.mean(numeric_only=True)
        output['Median']      = df.select_dtypes(include=[np.number]).quantile(0.5)
        output['3rd Qu.']     = df.select_dtypes(include=[np.number]).quantile(0.75)
        output['Max']         = df.max(numeric_only=True)
        output['Frequency']   = pd.Series({col: df[col].value_counts(sort=True, dropna=False).to_dict() for col in df.columns}).to_frame()
        output.insert(0, 'Name', output.index)
        output.to_csv(output_dir, sep='\t', index=False, na_rep='NaN')

    else:
        print('-- analyzing text dataset without metadata...')
        output                = pd.DataFrame()
        output['Class']       = pd.Series(df.dtypes)
        output['Length']      = len(df)
        output['NA']          = df.isna().sum()
        output['NotNA']       = output['Length'] - output['NA']
        output['Distinct']    = df.nunique(dropna=False).to_frame()
        output['Label']       = np.nan
        output['ValueLabels'] = np.nan
        output['Min']         = df.min(numeric_only=True)
        output['1st Qu.']     = df.select_dtypes(include=[np.number]).quantile(0.25)
        output['Mean']        = df.mean(numeric_only=True)
        output['Median']      = df.select_dtypes(include=[np.number]).quantile(0.5)
        output['3rd Qu.']     = df.select_dtypes(include=[np.number]).quantile(0.75)
        output['Max']         = df.max(numeric_only=True)
        output['Frequency']   = pd.Series({col: df[col].value_counts(sort=True, dropna=False).to_dict() for col in df.columns}).to_frame()
        output.insert(0, 'Name', output.index)
        output.to_csv(output_dir, sep='\t', index=False, na_rep='NaN')

    print('-- datainfo export complete.')


if __name__ == '__main__':
    testdt = pd.DataFrame({
        'id'              : [0, 1, 2, 3, 4, 5],
        'age'             : [30, 10, 40, 30, 60, 80],
        'gender'          : ['male','male','male','female','female','female'],
        'osat'            : [np.nan, 9,6,8, 8, 10],
        'attr_01'         : [np.nan, 8,4,7, np.nan, 10],
        'attr_02'         : [np.nan, 3,9,np.nan, 7, 5],
        'attr_03'         : [np.nan, np.nan,np.nan, 1, 5, 9],
        'wb'              : [np.nan, 1.2, 1.4, 0.6, np.nan, 0.8],
        'attr03_filter'   : [np.nan, 1,1,1,2,2],
        "segment"         : [np.nan, 1,1,2,2,2],
        "segment_txt"     : [np.nan, "mass", "mass", "middle", "middle", "middle"]
    })
    proc_freq(testdt, ['segment', 'attr03_filter', 'attr_03'])
    # print('='*80+'\n')
    # proc_means(testdt, ['osat', 'attr_01', 'attr_02', 'attr_03'], weight='wb')
