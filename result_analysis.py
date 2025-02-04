import sys
import os

import numpy as np
import pandas as pd
from typing import Literal
from PointClass import Point
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon


def perform_mcnemars_test(data, geoparser_a, geoparser_b):
    """
    Führt McNemar's Test für zwei Geotagging-Modelle durch.

    Parameters:
    - data: DataFrame mit den Ergebnissen.
    - geoparser_a: Präfix des ersten Geoparsers (z. B. 'edinburgh').
    - geoparser_b: Präfix des zweiten Geoparsers (z. B. 'irchel').

    Returns:
    - dict mit Teststatistik und p-Wert.
    """
    # Kontingenztabelle basierend auf den Erkennungsergebnissen
    n_00 = len(data[(data[f"{geoparser_a}_NER_result"].isnull()) & (data[f"{geoparser_b}_NER_result"].isnull())])
    n_01 = len(data[(data[f"{geoparser_a}_NER_result"].isnull()) & (data[f"{geoparser_b}_NER_result"].notnull())])
    n_10 = len(data[(data[f"{geoparser_a}_NER_result"].notnull()) & (data[f"{geoparser_b}_NER_result"].isnull())])
    n_11 = len(data[(data[f"{geoparser_a}_NER_result"].notnull()) & (data[f"{geoparser_b}_NER_result"].notnull())])

    table = [[n_00, n_01], [n_10, n_11]]

    # McNemar's Test durchführen
    result = mcnemar(table, exact=True)

    return {"Test Statistic": result.statistic, "p-value": result.pvalue}



def perform_wilcoxon_test(data, geoparser_a, geoparser_b):
    """
    Führt den Wilcoxon Signed-Rank Test für zwei Modelle basierend auf den Fehlerdistanzen durch.

    Parameters:
    - data: DataFrame mit den Fehlerdistanzen beider Modelle.
    - geoparser_a: Präfix des ersten Geoparsers (z. B. 'edinburgh').
    - geoparser_b: Präfix des zweiten Geoparsers (z. B. 'irchel').

    Returns:
    - dict mit Teststatistik und p-Wert.
    """
    # Fehlerdistanzen auswählen und NaN-Werte entfernen
    errors_a = data[f"{geoparser_a}_distance_error"].dropna()
    errors_b = data[f"{geoparser_b}_distance_error"].dropna()

    # Gemeinsame Indizes auswählen
    common_indices = errors_a.index.intersection(errors_b.index)

    # Wilcoxon-Test durchführen
    stat, p = wilcoxon(errors_a[common_indices], errors_b[common_indices])

    return {"Test Statistic": stat, "p-value": p}


def plot_error_distribution_boxplots(europe_data, africa_data):
    """
    Creates horizontal boxplots for error distances grouped by region (Europe, Africa) and geoparser type.

    Parameters:
    - europe_data: DataFrame for Europe.
    - africa_data: DataFrame for Africa.

    Returns:
    - fig: The matplotlib Figure with the boxplots.
    """
    plot_data = []

    for region, data in [("Europe", europe_data), ("Africa", africa_data)]:
        for geoparser, prefix in [("Edinburgh", "edinburgh"), ("Irchel", "irchel")]:
            column = f"{prefix}_distance_error"
            if column in data:
                for value in data[column].dropna():
                    plot_data.append({
                        "Region": region,
                        "Geoparser": geoparser,
                        "Error Distance (km)": value / 1000,
                    })

    plot_df = pd.DataFrame(plot_data)

    # Create boxplots
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(
        data=plot_df,
        x="Error Distance (km)",
        y="Region",
        hue="Geoparser",
        ax=ax,
        showfliers=False,
        orient="h"
    )
    ax.set_title("Error Distance Distribution by Region and Geoparser")
    ax.set_xlabel("Error Distance (km)")
    ax.set_ylabel("Region")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    return fig





def plot_accuracyATk_barchart(europe_data, africa_data, thresholds):
    """
    Creates a bar chart for Accuracy@k values using seaborn standard colors for geoparsers
    and hatching for regions, with bars placed side by side.

    Parameters:
    - europe_data: DataFrame for Europe.
    - africa_data: DataFrame for Africa.
    - thresholds: List of distance thresholds (k values).

    Returns:
    - fig: The matplotlib Figure with the modified bar chart.
    """
    import matplotlib.patches as mpatches

    # Prepare the data
    plot_data = []
    for region, data in [("Europe", europe_data), ("Africa", africa_data)]:
        for geoparser in ["edinburgh", "irchel"]:
            accuracies = [accuracy_at_k(data, f"{geoparser}_distance_error", k) for k in thresholds]
            for i, threshold in enumerate(thresholds):
                plot_data.append({
                    "Region": region,
                    "Geoparser": geoparser.title(),
                    "Threshold (km)": threshold / 1000,
                    "Accuracy": accuracies[i],
                })

    plot_df = pd.DataFrame(plot_data)

    # Define styles
    colors = sns.color_palette("deep")[:2]  # Default Seaborn colors: orange, blue
    hatches = {"Europe": "/", "Africa": None}
    bar_width = 0.2
    x_positions = range(len(thresholds))

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot bars
    bar_offset = 0  # Offset for adjusting bar positions
    for region, hatch in hatches.items():
        for geoparser, color in zip(["Edinburgh", "Irchel"], colors):
            sub_data = plot_df[(plot_df["Region"] == region) & (plot_df["Geoparser"] == geoparser)]
            ax.bar(
                [x + bar_offset for x in x_positions],
                sub_data["Accuracy"],
                width=bar_width,
                color=color,
                edgecolor="black",
                hatch=hatch,
                label=f"{geoparser} ({region})",
            )
            bar_offset += bar_width  # Move to the next position for bars

    # Set x-axis
    ax.set_xticks([x + (bar_width * 1.5) for x in x_positions])  # Center the ticks
    ax.set_xticklabels([f"{k / 1000} km" for k in thresholds])
    ax.set_xlabel("Distance Threshold (km)")

    # Set y-axis
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.6)

    legend_hatches = {"Europe": "///", "Africa": None}

    # Configure the legend
    geoparser_patches = [
        mpatches.Patch(color=color, label=geoparser)
        for color, geoparser in zip(colors, ["Edinburgh", "Irchel"])
    ]
    region_patches = [
        mpatches.Patch(hatch=hatch, facecolor="white", edgecolor="black", label=region)
        for region, hatch in legend_hatches.items()
    ]
    ax.legend(handles=geoparser_patches + region_patches, loc="upper left")

    # Add title and grid
    ax.set_title("Accuracy@k by Region and Geoparser (Hatching and Colors)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    return fig








def accuracy_at_k(data, distance_column, k):
    """
    Calculate Accuracy@k for a given distance threshold.

    Parameters:
    - data: DataFrame containing the resolution distance errors.
    - distance_column: The column name with the distance errors.
    - k: The distance threshold (in kilometers).

    Returns:
    - Accuracy@k: Proportion of predictions within the threshold k.
    """
    within_threshold = data[distance_column].notnull() & (data[distance_column] <= k)
    return within_threshold.sum() / len(data) if len(data) > 0 else 0


def calculate_log_auc(data, distance_column):
    """
    Calculate AUC using the discrete approximation of the integral.

    Parameters:
    - data: DataFrame containing distance errors.
    - distance_column: Column name with the error distances.

    Returns:
    - AUC: Area under the curve based on normalized logarithmic error distances.
    """
    # Maximum logarithmic error based on the Earth's half-circumference
    max_log_error = np.log(20039)

    # Logarithmic transformation and normalization
    data['log_distance_error'] = data[distance_column].apply(
        lambda x: np.log(x + 1) if pd.notnull(x) and x >= 0 else None
    )
    data['normalized_log_error'] = data['log_distance_error'] / max_log_error

    # Drop NaN values (invalid or missing distances)
    normalized_errors = data['normalized_log_error'].dropna()

    # Calculate the AUC as the average of normalized logarithmic errors
    auc = normalized_errors.mean() if not normalized_errors.empty else 0

    return auc




def calculate_metrics(data, col, total_column):
    """
    Berechnet Precision, Recall und F1-Score für spezifische und lose Definitionen von NER-Ergebnissen.

    Parameters:
    - data: DataFrame mit den Ergebnissen.
    - specific_column: Spalte mit spezifisch korrekten Erkennungen.
    - loosly_column: Spalte mit lose korrekten Erkennungen.
    - total_column: Spalte mit allen erkannten Ergebnissen (inkl. falscher Erkennungen).

    Returns:
    - Dictionary mit Precision, Recall und F1-Score für beide Metriken.
    """
    true_positive = len(data[data[col].notnull()])  # Korrekt erkannte Ergebnisse
    total_actual = len(data)  # Alle tatsächlichen Ergebnisse
    total_predicted = len(data[data[total_column].notnull()])  # Alle Vorhersagen (inkl. falsche)

    # Precision: Anteil korrekt erkannter unter allen vorhergesagten
    precision = true_positive / total_predicted if total_predicted > 0 else 0

    # Recall: Anteil korrekt erkannter unter allen tatsächlichen
    recall = true_positive / total_actual if total_actual > 0 else 0

    # F1-Score: Harmonie zwischen Precision und Recall
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {"Precision": precision, "Recall": recall, "F1": f1_score}

    return results


def distance_error(row, geoparser: Literal["irchel", "edinburgh"]):
    actual_Point = Point(x=row["longitude"], y=row["latitude"])
    if not actual_Point.x or not actual_Point.y:
        return None

    geoparser_dict = row[f"{geoparser}_NER_result"]
    if geoparser_dict is None:
        return None

    recognized_toponym = next(iter(geoparser_dict))

    pred_lon = geoparser_dict[recognized_toponym]["longitude"]
    pred_lat = geoparser_dict[recognized_toponym]["latitude"]

    if not pred_lon or not pred_lat:
        return None
    pred_lon = float(pred_lon)
    pred_lat = float(pred_lat)

    pred_Point = Point(x=pred_lon, y=pred_lat)
    if not pred_Point.x or not pred_Point.y:
        return None

    dist = actual_Point.distHaversine(pred_Point)
    return dist


def match_toponym(row, geoparser: Literal["irchel", "edinburgh"]):
    """
    Generalized function to match toponyms based on geoparser data.
    Parameters:
    - row: Data row containing the location and geoparser results.
    - geoparser: Which geoparser to use ('irchel' or 'edinburgh').
    - loosely: Whether to use loose matching (substring) or exact matching.
    """
    toponym = row["location"].lower()
    if geoparser == "irchel":
        dict_list = row.get("irchel_geoparser_res")
        if dict_list is None:
            return None

        for entry in dict_list:
            for key, value in entry.items():
                if toponym == key.lower():
                    return {key: value}
        return None

    elif geoparser == "edinburgh":
        dict_list = row.get("edinburgh_geoparser_res")
        if dict_list is None:
            return None

        for entry in dict_list:
            entry_name = entry["name"].lower()
            if toponym == entry_name:
                latitude = entry['latitude'] if entry['latitude'] != "" else None
                longitude = entry['longitude'] if entry['longitude'] != "" else None
                return {
                    entry['name']: {
                        'latitude': latitude,
                        'longitude': longitude,
                    }
                }
        return None

    else:
        raise ValueError(f"Invalid geoparser: {geoparser}")

def summarize_results_to_csv(europe_data, africa_data, metrics_europe, metrics_africa, aucs_europe, aucs_africa, thresholds, output_file):
    """
    Erstellt eine zusammengefasste CSV-Datei mit Ergebnissen für Geoparser-Region-Kombinationen.

    Parameter:
    - europe_data: DataFrame für Europa.
    - africa_data: DataFrame für Afrika.
    - metrics_europe: Metriken für Europa (Edinburgh und Irchel).
    - metrics_africa: Metriken für Afrika (Edinburgh und Irchel).
    - aucs_europe: AUC-Werte für Europa.
    - aucs_africa: AUC-Werte für Afrika.
    - thresholds: Liste der k-Werte.
    - output_file: Pfad zur CSV-Datei, in die die Ergebnisse geschrieben werden.
    """
    rows = []

    def add_summary_rows(data, metrics, aucs, region):
        mcnemar_result = perform_mcnemars_test(data, geoparser_a="edinburgh", geoparser_b="irchel")
        wilcoxon_result = perform_wilcoxon_test(data, geoparser_a="edinburgh", geoparser_b="irchel")

        for geoparser in ["edinburgh", "irchel"]:
            total_rows = len(data)
            row = {
                "Geoparser": geoparser.title(),
                "Region": region,
                "Total Rows": total_rows,
                "Rows with Results": len(data[data[f"{geoparser}_geoparser_res"].notnull()]),
                "Percentage with Results": f"{len(data[data[f'{geoparser}_geoparser_res'].notnull()]) / total_rows * 100:.2f}%",
                "Correctly recognized Toponyms": len(data[data[f"{geoparser}_NER_result"].notnull()]),
                "Percentage correctly recognized": f"{len(data[data[f'{geoparser}_NER_result'].notnull()]) / total_rows * 100:.2f}%",
                "Median Distance Error": f"{np.nanmedian(data[f'{geoparser}_distance_error']):.2f}" if not data[f'{geoparser}_distance_error'].isna().all() else "N/A",
                "Mean Distance Error": f"{np.nanmean(data[f'{geoparser}_distance_error']):.2f}" if not data[f'{geoparser}_distance_error'].isna().all() else "N/A",
                "AUC": f"{aucs[geoparser]:.4f}",
                "Precision": f"{metrics[geoparser]['Precision']:.2f}",
                "Recall": f"{metrics[geoparser]['Recall']:.2f}",
                "F1": f"{metrics[geoparser]['F1']:.2f}",
            }

            # Add Accuracy@k for each threshold
            for k in thresholds:
                accuracy = accuracy_at_k(data, f"{geoparser}_distance_error", k)
                row[f"Accuracy@{k}"] = f"{accuracy:.2f}"

            if geoparser == "edinburgh":  # Nur einmal hinzufügen, um Redundanz zu vermeiden
                row["McNemar Test Statistic"] = mcnemar_result["Test Statistic"]
                row["McNemar p-value"] = mcnemar_result["p-value"]
                row["Wilcoxon Test Statistic"] = wilcoxon_result["Test Statistic"]
                row["Wilcoxon p-value"] = wilcoxon_result["p-value"]

            rows.append(row)

    # Add summaries for Europe and Africa
    add_summary_rows(europe_data, metrics_europe, aucs_europe, "Europe")
    add_summary_rows(africa_data, metrics_africa, aucs_africa, "Africa")

    # Convert rows to DataFrame and save as CSV
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_file, index=False)

    return summary_df

def main(europe_file, africa_file, output_dir):
    # Dateien laden
    europe_data = pd.read_pickle(europe_file)
    africa_data = pd.read_pickle(africa_file)

    # NER-Ergebnisse bereinigen
    for data in [europe_data, africa_data]:
        data['edinburgh_geoparser_res'] = data['edinburgh_geoparser_res'].apply(lambda x: None if isinstance(x, list) and len(x) == 0 else x)
        data['edinburgh_geoparser_res'] = data['edinburgh_geoparser_res'].apply(lambda x: None if x == -1 else x)
        data['irchel_geoparser_res'] = data['irchel_geoparser_res'].apply(lambda x: None if isinstance(x, list) and len(x) == 0 else x)
        data['irchel_geoparser_res'] = data['irchel_geoparser_res'].apply(lambda x: None if x == -1 else x)

    # NER Matching anwenden
    for name, data in [("Europe", europe_data), ("Africa", africa_data)]:
        data["edinburgh_NER_result"] = data.apply(lambda x: match_toponym(x, "edinburgh"), axis=1)
        data["irchel_NER_result"] = data.apply(lambda x: match_toponym(x, "irchel"), axis=1)

    # Metriken berechnen
    metrics_europe = {
        "edinburgh": calculate_metrics(europe_data, col="edinburgh_NER_result", total_column="edinburgh_geoparser_res"),
        "irchel": calculate_metrics(europe_data, col="irchel_NER_result", total_column="irchel_geoparser_res"),
    }

    metrics_africa = {
        "edinburgh": calculate_metrics(africa_data, col="edinburgh_NER_result", total_column="edinburgh_geoparser_res"),
        "irchel": calculate_metrics(africa_data, col="irchel_NER_result", total_column="irchel_geoparser_res"),
    }

    # Distance Errors berechnen
    for name, data in [("Europe", europe_data), ("Africa", africa_data)]:
        data["edinburgh_distance_error"] = data.apply(lambda x: distance_error(x, "edinburgh"), axis=1)
        data["irchel_distance_error"] = data.apply(lambda x: distance_error(x, "irchel"), axis=1)

    # AUC und Accuracy@k berechnen
    thresholds = [1000, 5000, 10000, 50_000, 100_000, 161000, 500000]
    aucs_europe = {
        "edinburgh": calculate_log_auc(europe_data, "edinburgh_distance_error"),
        "irchel": calculate_log_auc(europe_data, "irchel_distance_error"),
    }
    aucs_africa = {
        "edinburgh": calculate_log_auc(africa_data, "edinburgh_distance_error"),
        "irchel": calculate_log_auc(africa_data, "irchel_distance_error"),
    }

    # Statistische Tests durchführen
    # McNemar's Test
    mcnemar_europe = perform_mcnemars_test(europe_data, geoparser_a="edinburgh", geoparser_b="irchel")
    mcnemar_africa = perform_mcnemars_test(africa_data, geoparser_a="edinburgh", geoparser_b="irchel")
    print(f"McNemar's Test Europe: {mcnemar_europe}")
    print(f"McNemar's Test Africa: {mcnemar_africa}")

    # Wilcoxon Signed-Rank Test
    wilcoxon_europe = perform_wilcoxon_test(europe_data, geoparser_a="edinburgh", geoparser_b="irchel")
    wilcoxon_africa = perform_wilcoxon_test(africa_data, geoparser_a="edinburgh", geoparser_b="irchel")
    print(f"Wilcoxon Signed-Rank Test Europe: {wilcoxon_europe}")
    print(f"Wilcoxon Signed-Rank Test Africa: {wilcoxon_africa}")


    output_csv = os.path.join(output_dir, "geoparser_summary.csv")
    summary_df = summarize_results_to_csv(
        europe_data,
        africa_data,
        metrics_europe,
        metrics_africa,
        aucs_europe,
        aucs_africa,
        thresholds,
        output_csv
    )

    # Plots erstellen und speichern
    plots = {
        "Error Distance Boxplots": plot_error_distribution_boxplots(europe_data, africa_data),
        "Accuracy@k Comparison": plot_accuracyATk_barchart(europe_data, africa_data, thresholds),
    }
    for name, fig in plots.items():
        fig.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pdf"), bbox_inches='tight')
        plt.close(fig)

    print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python result_analysis.py <europe.pkl> <africa.pkl> <output_dir>")
        print("Default Inputs taken: python result_analysis.py result/europe.pkl result/africa.pkl result/")
        europe_file = "result/europe.pkl"
        africa_file = "result/africa.pkl"
    else:
        europe_file = sys.argv[1]
        africa_file = sys.argv[2]

    output_dir = sys.argv[3] if len(sys.argv) > 3 else "result/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(europe_file, africa_file, output_dir)