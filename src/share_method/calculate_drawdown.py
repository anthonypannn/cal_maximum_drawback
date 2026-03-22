"""
指数最大回撤计算工具
计算任意指数的每年度最大回撤，并输出为Excel文件
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def calculate_drawdown(price_series):
    """
    计算回撤序列
    
    参数:
        price_series: 价格序列（Series）
    
    返回:
        drawdown: 回撤序列
        cummax: 累积最大值序列
    """
    # 计算累积最大值
    cummax = price_series.cummax()
    # 计算回撤
    drawdown = (price_series - cummax) / cummax
    return drawdown, cummax


def find_max_drawdown_period(price_series, drawdown_series):
    """
    找到最大回撤的开始和结束时间
    
    参数:
        price_series: 价格序列
        drawdown_series: 回撤序列
    
    返回:
        max_dd: 最大回撤值
        start_date: 开始日期
        end_date: 结束日期
        duration: 持续天数
    """
    if len(drawdown_series) == 0 or drawdown_series.isna().all():
        return np.nan, None, None, np.nan
    
    # 找到最大回撤的位置（最小值位置，因为回撤是负数）
    max_dd_idx = drawdown_series.idxmin()
    max_dd = drawdown_series.loc[max_dd_idx]
    
    # 找到最大回撤结束时间（就是最低点的时间）
    end_date = max_dd_idx
    
    # 找到开始时间（最大回撤之前的最高点）
    prices_before = price_series.loc[:max_dd_idx]
    start_idx = prices_before.idxmax()
    start_date = start_idx
    
    # 计算持续时间（天数）
    duration = (end_date - start_date).days
    
    return max_dd, start_date, end_date, duration


def calculate_yearly_drawdown(df, date_col, price_col, freq='D'):
    """
    计算指定指数的年度最大回撤
    
    参数:
        df: 数据框
        date_col: 日期列名
        price_col: 价格列名
        freq: 数据频率，默认为'D'（日）
    
    返回:
        result_df: 包含年度最大回撤信息的数据框
    """
    # 确保日期列是datetime类型
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 提取年份
    df['年份'] = df[date_col].dt.year
    
    # 按年份分组计算
    results = []
    
    for year, group_df in df.groupby('年份'):
        if len(group_df) < 2:  # 数据点太少，跳过
            continue
        
        # 重置索引，使用日期作为索引
        group_df = group_df.set_index(date_col)
        price_series = group_df[price_col]
        
        # 数据清洗：剔除值为0的数据（0表示该时点指数数据还未出现）
        price_series = price_series[price_series != 0]
        
        # 如果清洗后数据不足，跳过该年
        if len(price_series) < 2:
            continue
        
        # 计算回撤
        drawdown, cummax = calculate_drawdown(price_series)
        
        # 找到最大回撤及其时间范围
        max_dd, start_date, end_date, duration = find_max_drawdown_period(
            price_series, drawdown
        )
        
        results.append({
            '年份': year,
            '最大回撤': max_dd,
            '回撤开始时间': start_date,
            '回撤结束时间': end_date,
            '持续天数': duration
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def process_all_indices(input_file, output_dir='output', freq='D'):
    """
    处理所有指数，计算年度最大回撤并输出到Excel
    
    参数:
        input_file: 输入文件路径
        output_dir: 输出文件夹路径
        freq: 数据频率
    """
    # 读取数据
    print(f"正在读取数据文件: {input_file}")
    df = pd.read_excel(input_file)
    
    # 获取列名
    columns = df.columns.tolist()
    date_col = columns[0]  # 第一列是日期
    index_cols = columns[1:]  # 其他列是指数
    
    print(f"检测到 {len(index_cols)} 个指数: {', '.join(index_cols)}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d')
    output_file = os.path.join(output_dir, f'{input_file.split("/")[-1].split(".")[0]}_指数回撤分析结果_{timestamp}.xlsx')
    
    # 创建Excel写入器
    print(f"开始计算各指数的年度最大回撤...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for index_name in index_cols:
            print(f"  处理指数: {index_name}")
            
            # 计算该指数的年度最大回撤
            result_df = calculate_yearly_drawdown(df, date_col, index_name, freq)
            
            # 格式化输出
            # 最大回撤转换为百分比显示
            result_df['最大回撤(%)'] = result_df['最大回撤'] * 100
            
            # 重新排列列顺序
            output_cols = ['年份', '最大回撤(%)', '回撤开始时间', '回撤结束时间', '持续天数']
            result_df = result_df[output_cols]
            
            # 写入到不同的sheet
            sheet_name = f"{index_name}_回撤分析"
            # Excel的sheet名称最多31个字符
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:28] + '...'
            
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"    完成，共 {len(result_df)} 年的数据")
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    return output_file


def main():
    """
    主函数
    """
    # 配置参数
    input_file = 'data/SPX.GI.xlsx'   # 输入文件路径
    output_dir = 'output'  # 输出目录
    freq = 'D'  # 数据频率：'D'=日, 'W'=周, 'M'=月
    
    print("=" * 60)
    print("指数最大回撤计算工具")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    # 处理数据
    try:
        output_file = process_all_indices(input_file, output_dir, freq)
        print("\n任务执行成功！")
    except Exception as e:
        print(f"\n错误：执行过程中出现异常")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

