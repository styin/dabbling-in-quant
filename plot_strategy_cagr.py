import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("strategy_cagr_comparison.csv")

    # Keep a consistent ordering for strategies in the legend and line colors.
    strategy_order = [
        "#2 Only",
        "#3 Only",
        "#4 Only",
        "#5 Only",
        "EQ #2-3",
        "EQ #2-4",
        "EQ #2-5",
        "EQ #3-5",
    ]
    start_years = sorted(df["start_year"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax1 = axes[0]
    colors = plt.get_cmap("tab10")

    for i, strategy in enumerate(strategy_order):
        sdf = (
            df[df["strategy"] == strategy]
            .sort_values("start_year")
            .set_index("start_year")
            .reindex(start_years)
        )
        ax1.plot(
            start_years,
            sdf["strategy_cagr"] * 100,
            marker="o",
            linewidth=2,
            color=colors(i),
            label=strategy,
        )

    # Shared benchmark across strategies for each start year.
    sp = (
        df[["start_year", "sp500_cagr"]]
        .drop_duplicates()
        .sort_values("start_year")
        .set_index("start_year")
        .reindex(start_years)
    )
    ax1.plot(
        start_years,
        sp["sp500_cagr"] * 100,
        marker="s",
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="S&P 500",
    )

    ax1.set_title("CAGR by Start Year")
    ax1.set_xlabel("Start Year")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_xticks(start_years)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)

    # Secondary view: excess CAGR vs S&P 500 makes relative performance clearer.
    ax2 = axes[1]
    for i, strategy in enumerate(strategy_order):
        sdf = (
            df[df["strategy"] == strategy]
            .sort_values("start_year")
            .set_index("start_year")
            .reindex(start_years)
        )
        ax2.plot(
            start_years,
            sdf["excess_cagr"] * 100,
            marker="o",
            linewidth=2,
            color=colors(i),
            label=strategy,
        )

    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.set_title("Excess CAGR vs S&P 500")
    ax2.set_xlabel("Start Year")
    ax2.set_ylabel("Excess CAGR (%)")
    ax2.set_xticks(start_years)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Top-Rank Company Strategies vs S&P 500", fontsize=14)
    fig.savefig("strategy_cagr_plot.png", dpi=180)


if __name__ == "__main__":
    main()
