import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_preparation():
    # -------------------------
    # LOAD Synthea data
    # -------------------------
    encounters = pd.read_csv("./data/synthea/encounters.csv")

    # Keep only inpatient encounters
    encounters["START"] = pd.to_datetime(encounters["START"])
    encounters["STOP"] = pd.to_datetime(encounters["STOP"])
    inp = encounters[encounters["ENCOUNTERCLASS"] == "inpatient"].copy()

    # Compute LOS in days
    inp["LOS_days"] = (inp["STOP"] - inp["START"]).dt.total_seconds() / (3600 * 24)  # type: ignore

    # See results
    print(inp[["Id", "START", "STOP", "LOS_days"]].head())

    # Number of admissions

    # Show LOS-days distribution
    print(inp["LOS_days"].describe())

    # Show statistics of LOS > 7 days compared to others
    long_stays = inp[inp["LOS_days"] > 7]
    short_stays = inp[inp["LOS_days"] <= 7]
    print(f"Long stays (>7 days): {len(long_stays)}")
    print(f"Short stays (<=7 days): {len(short_stays)}")
    print(f"Average LOS for long stays: {long_stays['LOS_days'].mean()}")
    print(f"Average LOS for short stays: {short_stays['LOS_days'].mean()}")

    # Show how many has between 6 and 8 days
    between_6_and_8 = inp[(inp["LOS_days"] >= 6) & (inp["LOS_days"] <= 8)]
    print(f"Number of stays between 6 and 8 days: {len(between_6_and_8)}")

    # ============================================================================
    # CREATE PERCENTILE VISUALIZATION
    # ============================================================================

    _fig, ax = plt.subplots(figsize=(10, 6))

    color_accent = "#ff7f0e"

    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(inp["LOS_days"], p) for p in percentiles]

    bars = ax.bar([f"{p}th" for p in percentiles], percentile_values, color=color_accent, alpha=0.8, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("LOS (days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentile", fontsize=12, fontweight="bold")
    ax.set_title("LOS Percentiles", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Add value labels on top of bars
    for bar, value in zip(bars, percentile_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/synthea_los_percentiles.png", dpi=300, bbox_inches="tight")
    print("Percentiles plot saved to './figures/synthea_los_percentiles.png'")

    # -------------------------
    # BUILD INPUTS = SNOMED BAG-OF-CODES
    # -------------------------
    conditions = pd.read_csv("./data/synthea/conditions.csv")

    # Keep only conditions for inpatient encounters
    conditions = conditions[conditions["ENCOUNTER"].isin(inp["Id"])]

    # For each encounter kept, join SNOMED codes into a space-separated string
    encounter_codes = conditions.groupby("ENCOUNTER")["CODE"].apply(lambda x: " ".join(str(c) for c in x)).reset_index()

    encounter_codes.columns = ["ENCOUNTER", "CONDITIONS"]

    # Add LOS in encounter_codes by merging ENCOUNTER from encounter_codes with Id from inp (if no conditions, CONDITIONS will be NaN)
    encounter_codes = encounter_codes.merge(inp[["Id", "LOS_days"]], left_on="ENCOUNTER", right_on="Id", how="left")

    # Count and print NaN values in CONDITIONS
    num_nans = encounter_codes["CONDITIONS"].isna().sum()
    print(f"Number of encounters with no conditions: {num_nans}")

    # drop Id column
    encounter_codes = encounter_codes.drop(columns=["Id"])

    # add procedures too
    procedures = pd.read_csv("./data/synthea/procedures.csv")
    procedures = procedures[procedures["ENCOUNTER"].isin(inp["Id"])]
    encounter_procedures = procedures.groupby("ENCOUNTER")["CODE"].apply(lambda x: " ".join(str(c) for c in x)).reset_index()
    encounter_procedures.columns = ["ENCOUNTER", "PROCEDURES"]

    # merge procedures into encounter_codes (if no procedures, PROCEDURES will be NaN)
    encounter_codes = encounter_codes.merge(encounter_procedures, on="ENCOUNTER", how="left")

    # Count and print NaN values in PROCEDURES
    num_nans = encounter_codes["PROCEDURES"].isna().sum()
    print(f"Number of encounters with no procedures: {num_nans}")

    # Statistics nb codes in CONDITIONS and PROCEDURES
    encounter_codes["NUM_CONDITIONS"] = encounter_codes["CONDITIONS"].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
    encounter_codes["NUM_PROCEDURES"] = encounter_codes["PROCEDURES"].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
    print("Statistics of number of condition codes:")
    print(encounter_codes["NUM_CONDITIONS"].describe())
    print("Statistics of number of procedure codes:")
    print(encounter_codes["NUM_PROCEDURES"].describe())

    # Encode classes, LOS > 7 days = 1, else 0
    encounter_codes["LONG_STAY"] = encounter_codes["LOS_days"].apply(lambda x: 1 if x > 7 else 0)

    return encounter_codes


def save_encounter_codes(df, filepath="./data/synthea/encounter_data.tsv"):
    df.to_csv(filepath, sep="\t", index=False)
    print(f"Encounter data saved to {filepath}")


if __name__ == "__main__":
    df_data = data_preparation()
    save_encounter_codes(df_data)
