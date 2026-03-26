"""
黄金跌破年线后的最大回撤分析

口径说明：
1. 年线 = 250 个交易日收盘价简单移动平均（MA250）
2. 跌破年线 = 前一交易日收盘价不低于年线，且当日收盘价低于年线
3. 跌破后的观察区间 = 从跌破当日开始，到重新站上年线前一日为止；
   如果样本结束前一直未重新站上年线，则观察到样本最后一日
4. 后续最大回撤 = 以跌破当日收盘价为起点，在上述观察区间内统计到最低点的最大跌幅

数据处理说明：
1. 输入文件为 Wind/EDB 风格导出，前置空白和元信息不作为正式数据
2. 仅保留 1975-01-02 起的日频数据，避免将早期月频数据混入 MA250 口径
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
SHARE_METHOD_DIR = SRC_ROOT / "share_method"

if str(SHARE_METHOD_DIR) not in sys.path:
    sys.path.append(str(SHARE_METHOD_DIR))

from backtesting_analysis import calculate_performance_metrics


MA_WINDOW = 250
MIN_DAILY_DATE = pd.Timestamp("1975-01-02")
DEFAULT_SHEET_NAME = "Sheet1"
INPUT_DIR = PROJECT_ROOT / "data" / "20260326-黄金年线回撤"
OUTPUT_DIR = PROJECT_ROOT / "output" / "20260326-黄金年线回撤"


def _is_non_empty(value: object) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def load_gold_price_data(
    input_file: Path,
    sheet_name: str = DEFAULT_SHEET_NAME,
    min_daily_date: pd.Timestamp = MIN_DAILY_DATE,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    读取 Wind/EDB 风格的黄金价格数据。

    已识别格式：
    - 前 29 行为空白
    - 第 30-34 行为元信息
    - 第 35 行起为正式数据
    - 第 2 列为日期，第 3 列为价格，第 1 列为空列
    """
    raw_df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)
    non_empty_rows = raw_df.apply(lambda row: row.map(_is_non_empty).any(), axis=1)

    if not non_empty_rows.any():
        raise ValueError(f"输入文件不存在有效数据区域: {input_file}")

    first_non_empty_idx = non_empty_rows[non_empty_rows].index[0]
    metadata_start_idx = first_non_empty_idx
    data_start_idx = metadata_start_idx + 5

    metadata_df = raw_df.iloc[metadata_start_idx:data_start_idx].reset_index(drop=True)
    data_df = raw_df.iloc[data_start_idx:].copy()

    if data_df.shape[1] < 3:
        raise ValueError(f"输入文件列数不足，无法识别日期列与价格列: {input_file}")

    series_name = str(metadata_df.iloc[3, 2]).strip() if _is_non_empty(metadata_df.iloc[3, 2]) else "黄金价格"
    ticker = str(metadata_df.iloc[4, 2]).strip() if _is_non_empty(metadata_df.iloc[4, 2]) else ""

    cleaned_df = pd.DataFrame(
        {
            "日期": pd.to_datetime(data_df.iloc[:, 1], errors="coerce"),
            series_name: pd.to_numeric(data_df.iloc[:, 2], errors="coerce"),
        }
    )

    cleaned_df = cleaned_df.dropna(subset=["日期", series_name])
    cleaned_df = cleaned_df[cleaned_df[series_name] != 0]
    cleaned_df = cleaned_df.drop_duplicates(subset=["日期"], keep="last")
    cleaned_df = cleaned_df.sort_values("日期").reset_index(drop=True)
    cleaned_df = cleaned_df[cleaned_df["日期"] >= min_daily_date].reset_index(drop=True)

    if cleaned_df.empty:
        raise ValueError(f"清洗后无可用于日频年线分析的数据: {input_file}")

    ticker_mapping = {series_name: ticker}
    return cleaned_df, ticker_mapping


def analyze_single_index_below_yearline(
    df: pd.DataFrame,
    price_col: str,
    ticker: str = "",
    ma_window: int = MA_WINDOW,
) -> pd.DataFrame:
    """
    计算单个价格序列历史上每次跌破年线后、到重新站上年线前的最大跌幅。
    """
    working_df = df[["日期", price_col]].copy()
    working_df = working_df.dropna(subset=[price_col])
    working_df = working_df[working_df[price_col] != 0].reset_index(drop=True)

    if len(working_df) < ma_window + 2:
        return pd.DataFrame()

    working_df["年线"] = working_df[price_col].rolling(ma_window, min_periods=ma_window).mean()
    working_df = working_df.dropna(subset=["年线"]).reset_index(drop=True)

    if len(working_df) < 2:
        return pd.DataFrame()

    working_df["低于年线"] = working_df[price_col] < working_df["年线"]
    previous_below = working_df["低于年线"].shift(1)
    break_mask = working_df["低于年线"] & (previous_below == False)
    break_indices = working_df.index[break_mask].tolist()

    results: list[dict[str, object]] = []

    for event_no, start_idx in enumerate(break_indices, start=1):
        recovery_candidates = working_df.index[
            (working_df.index > start_idx) & (~working_df["低于年线"])
        ].tolist()

        if recovery_candidates:
            recovery_idx = recovery_candidates[0]
            observation_end_idx = recovery_idx - 1
            recovery_date = working_df.loc[recovery_idx, "日期"]
            recovered = "是"
        else:
            observation_end_idx = len(working_df) - 1
            recovery_date = pd.NaT
            recovered = "否"

        event_df = working_df.loc[start_idx:observation_end_idx, ["日期", price_col]].copy()
        event_df = event_df.set_index("日期")
        price_series = event_df[price_col]

        break_date = working_df.loc[start_idx, "日期"]
        break_price = working_df.loc[start_idx, price_col]
        yearline_value = working_df.loc[start_idx, "年线"]
        observation_end_date = working_df.loc[observation_end_idx, "日期"]
        lowest_date = price_series.idxmin()
        lowest_price = price_series.loc[lowest_date]

        interval_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1 if len(price_series) >= 1 else np.nan
        drop_from_break = (lowest_price / break_price) - 1 if break_price != 0 else np.nan
        max_drawdown = drop_from_break
        dd_start_date = break_date
        dd_end_date = lowest_date
        dd_duration = (dd_end_date - dd_start_date).days if pd.notna(dd_end_date) else np.nan

        annualized_return, annualized_volatility, sharpe_ratio = calculate_performance_metrics(price_series)

        results.append(
            {
                "指数名称": price_col,
                "指数代码": ticker,
                "事件序号": event_no,
                "跌破年线日期": break_date,
                "跌破年线收盘价": break_price,
                "跌破当日年线": yearline_value,
                "是否重新站上年线": recovered,
                "重新站上年线日期": recovery_date,
                "观察区间结束日期": observation_end_date,
                "区间交易日数量": len(price_series),
                "区间收益率": interval_return,
                "跌破后最低点日期": lowest_date,
                "跌破后最低点价格": lowest_price,
                "跌破日至最低点跌幅": drop_from_break,
                "后续最大回撤": max_drawdown,
                "后续最大回撤开始日期": dd_start_date,
                "后续最大回撤结束日期": dd_end_date,
                "后续最大回撤持续天数": dd_duration,
                "年化收益率": annualized_return,
                "年化波动率": annualized_volatility,
                "夏普比率": sharpe_ratio,
            }
        )

    return pd.DataFrame(results)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    按指数汇总历史跌破年线事件表现。
    """
    if result_df.empty:
        return pd.DataFrame()

    summary_df = (
        result_df.groupby(["指数名称", "指数代码"], dropna=False)
        .agg(
            跌破年线次数=("事件序号", "count"),
            首次跌破年线日期=("跌破年线日期", "min"),
            最近一次跌破年线日期=("跌破年线日期", "max"),
            最差后续最大回撤=("后续最大回撤", "min"),
            平均后续最大回撤=("后续最大回撤", "mean"),
            中位数后续最大回撤=("后续最大回撤", "median"),
            平均观察交易日数量=("区间交易日数量", "mean"),
            未重新站上年线次数=("是否重新站上年线", lambda s: (s == "否").sum()),
        )
        .reset_index()
    )
    return summary_df


def format_percentage_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    将小数比例转换为百分数，便于导出查看。
    """
    formatted_df = df.copy()
    for column in columns:
        if column in formatted_df.columns:
            formatted_df[column] = formatted_df[column] * 100
    return formatted_df


def build_sheet_name(index_name: str) -> str:
    """
    生成符合 Excel 限制的 sheet 名称。
    """
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]
    sheet_name = index_name
    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")
    return sheet_name[:31]


def process_file(input_file: Path, output_dir: Path = OUTPUT_DIR, ma_window: int = MA_WINDOW) -> Path:
    """
    处理输入文件并输出结果。
    """
    print(f"正在读取文件: {input_file}")
    df, ticker_mapping = load_gold_price_data(input_file)

    index_columns = [column for column in df.columns if column != "日期"]
    print(f"检测到 {len(index_columns)} 个价格序列: {', '.join(index_columns)}")
    print(f"样本起始日期（清洗后）: {df['日期'].min().date()}")
    print(f"样本结束日期（清洗后）: {df['日期'].max().date()}")
    print(f"年线计算方式: {ma_window} 个交易日简单移动平均线（MA{ma_window}）")

    all_results: list[pd.DataFrame] = []
    for index_name in index_columns:
        print(f"  处理序列: {index_name}")
        result_df = analyze_single_index_below_yearline(
            df=df,
            price_col=index_name,
            ticker=ticker_mapping.get(index_name, ""),
            ma_window=ma_window,
        )
        if not result_df.empty:
            all_results.append(result_df)
            print(f"    完成，识别到 {len(result_df)} 次跌破年线事件")
        else:
            print("    数据不足或未识别到有效事件")

    if not all_results:
        raise ValueError("未识别到任何可分析的跌破年线事件。")

    final_result_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_results(final_result_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"{input_file.stem}_跌破年线后最大回撤分析_{timestamp}.xlsx"

    result_export_df = format_percentage_columns(
        final_result_df,
        ["区间收益率", "跌破日至最低点跌幅", "后续最大回撤", "年化收益率", "年化波动率"],
    )
    summary_export_df = format_percentage_columns(
        summary_df,
        ["最差后续最大回撤", "平均后续最大回撤", "中位数后续最大回撤"],
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_export_df.to_excel(writer, sheet_name="汇总统计", index=False)

        for index_name in index_columns:
            index_result_df = result_export_df[result_export_df["指数名称"] == index_name].copy()
            if index_result_df.empty:
                continue
            index_result_df.to_excel(
                writer,
                sheet_name=build_sheet_name(index_name),
                index=False,
            )

    print(f"分析完成，结果已保存到: {output_file}")
    return output_file


def find_input_file(input_dir: Path = INPUT_DIR) -> Path:
    """
    从目标目录中找到唯一的 xlsx 输入文件。
    """
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"未在目录中找到 Excel 文件: {input_dir}")
    if len(files) > 1:
        raise ValueError(f"目录下存在多个 Excel 文件，请保留一个输入文件后重试: {input_dir}")
    return files[0]


def main() -> None:
    input_file = find_input_file()

    print("=" * 72)
    print("黄金跌破年线后的最大回撤分析")
    print("=" * 72)
    print("输入数据说明：自动识别 Wind/EDB 导出格式，并清洗前置空白及元信息")
    print("样本范围说明：仅保留 1975-01-02 起的日频数据")
    print("年线定义：250 个交易日收盘价简单移动平均线（MA250）")
    print("跌破定义：前一日不低于年线、当日收盘价低于年线")
    print("观察区间：跌破当日至重新站上年线前一日；若未修复则到样本末尾")
    print("=" * 72)

    try:
        process_file(input_file=input_file)
        print("任务执行成功。")
    except Exception as exc:
        print("任务执行失败。")
        print(f"错误信息: {exc}")
        raise


if __name__ == "__main__":
    main()
