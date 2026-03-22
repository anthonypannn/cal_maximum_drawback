"""
指数回测分析工具
计算指定年份到现在的年化收益率、波动率和夏普比率
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def calculate_performance_metrics(price_series):
    """
    计算单个价格序列的绩效指标
    
    参数:
        price_series: 价格序列（Series），索引为日期
    
    返回:
        annualized_return: 年化收益率
        annualized_volatility: 年化波动率
        sharpe_ratio: 夏普比率（收益率/波动率）
    """
    if len(price_series) < 2:
        return np.nan, np.nan, np.nan
    
    # 计算日收益率
    daily_returns = price_series.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return np.nan, np.nan, np.nan
    
    # 计算总收益率
    total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    
    # 计算年数
    start_date = price_series.index[0]
    end_date = price_series.index[-1]
    years = (end_date - start_date).days / 365
    
    if years <= 0:
        return np.nan, np.nan, np.nan
    
    # 计算年化收益率
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # 计算年化波动率（假设一年252个交易日）
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    
    # 计算夏普比率
    if annualized_volatility != 0 and not np.isnan(annualized_volatility):
        sharpe_ratio = annualized_return / annualized_volatility
    else:
        sharpe_ratio = np.nan
    
    return annualized_return, annualized_volatility, sharpe_ratio


def analyze_single_index(df, date_col, price_col, start_year=None):
    """
    分析单个指数的绩效指标
    
    参数:
        df: 数据框
        date_col: 日期列名
        price_col: 价格列名
        start_year: 开始年份，默认为None（使用所有数据）
    
    返回:
        字典，包含指数名称和绩效指标
    """
    # 确保日期列是datetime类型
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 筛选指定年份之后的数据
    if start_year is not None:
        start_date = pd.Timestamp(f'{start_year}-01-01')
        df = df[df[date_col] >= start_date]
    
    if len(df) < 2:
        return {
            '指数名称': price_col,
            '年化收益率(%)': np.nan,
            '年化波动率(%)': np.nan,
            '夏普比率': np.nan
        }
    
    # 设置日期为索引
    df = df.set_index(date_col)
    price_series = df[price_col]
    
    # 数据清洗：剔除值为0的数据（0表示该时点指数数据还未出现）
    price_series = price_series[price_series != 0]
    
    # 剔除NaN值
    price_series = price_series.dropna()
    
    if len(price_series) < 2:
        return {
            '指数名称': price_col,
            '年化收益率(%)': np.nan,
            '年化波动率(%)': np.nan,
            '夏普比率': np.nan
        }
    
    # 计算绩效指标
    ann_return, ann_vol, sharpe = calculate_performance_metrics(price_series)
    
    return {
        '指数名称': price_col,
        '年化收益率(%)': ann_return * 100 if not np.isnan(ann_return) else np.nan,
        '年化波动率(%)': ann_vol * 100 if not np.isnan(ann_vol) else np.nan,
        '夏普比率': sharpe
    }


def analyze_all_indices(input_file, output_dir='output', start_year=None):
    """
    分析所有指数的绩效指标并输出到Excel
    
    参数:
        input_file: 输入文件路径
        output_dir: 输出文件夹路径
        start_year: 开始年份，默认为None
    """
    # 读取数据
    print(f"正在读取数据文件: {input_file}")
    df = pd.read_excel(input_file)
    
    # 获取列名
    columns = df.columns.tolist()
    date_col = columns[0]  # 第一列是日期
    index_cols = columns[1:]  # 其他列是指数
    
    print(f"检测到 {len(index_cols)} 个指数: {', '.join(index_cols)}")
    
    # 确定分析期间
    if start_year is None:
        print(f"分析期间: 全部数据")
    else:
        print(f"分析期间: {start_year}年初至今")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 分析每个指数
    print(f"开始计算各指数的绩效指标...")
    results = []
    
    for index_name in index_cols:
        print(f"  处理指数: {index_name}")
        result = analyze_single_index(df, date_col, index_name, start_year)
        results.append(result)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d')
    if start_year is not None:
        output_file = os.path.join(output_dir, f'{start_year}年开始的回测统计_{timestamp}.xlsx')
    else:
        output_file = os.path.join(output_dir, f'全周期回测统计_{timestamp}.xlsx')
    
    # 保存到Excel
    result_df.to_excel(output_file, index=False, sheet_name='回测统计')
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    print(f"\n结果预览:")
    print(result_df.to_string(index=False))
    
    return output_file


def main():
    """
    主函数
    """
    # 配置参数
    input_file = 'data/SPX.GI.xlsx'  # 输入文件路径
    output_dir = 'output'  # 输出目录
    
    # 计算开始年份（默认为当前年份往前推10年）
    current_year = datetime.now().year
    start_year = current_year - 9  # 默认10年
    
    print("=" * 60)
    print("指数回测分析工具")
    print("=" * 60)
    print(f"当前年份: {current_year}")
    print(f"分析期间: {start_year}年初至今（{current_year - start_year}年）")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    # 处理数据
    try:
        output_file = analyze_all_indices(input_file, output_dir, start_year)
        print("\n任务执行成功！")
    except Exception as e:
        print(f"\n错误：执行过程中出现异常")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

