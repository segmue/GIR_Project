import sys
import os

import numpy as np
import pandas as pd
from typing import Literal
from PointClass import Point

import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_distribution_boxplots(data):
    """
    Create horizontal boxplots of error distances for each combination of geoparser (Edinburgh/Irchel)
    and matching type (Specific/Loosely), with color distinction for matching types.

    Parameters:
    - data: DataFrame containing the resolution distance errors.

    Returns:
    - fig: The matplotlib Figure object containing the boxplots.
    """
    import pandas as pd

    # Prepare data for boxplots
    plot_data = []
    for geoparser, prefix in [("Edinburgh", "edinburgh"), ("Irchel", "irchel")]:
        for match_type, column_suffix in [("Specific", "distance_error_specific"), ("Loosely", "distance_error_loosly")]:
            column = f"{prefix}_{column_suffix}"
            if column in data:
                for value in data[column].dropna():
                    plot_data.append({
                        "Geoparser": geoparser,
                        "Matching Type": match_type,
                        "Error Distance (km)": value / 1000  # Convert to kilometers
                    })

    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Create horizontal boxplots with hue
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(
        data=plot_df,
        x="Error Distance (km)",
        y="Geoparser",
        hue="Matching Type",
        ax=ax,
        showfliers=False,  # Exclude outliers for cleaner visualization
        orient="h"  # Horizontal orientation
    )
    ax.set_title("Error Distance Distribution by Geoparser and Matching Type")
    ax.set_xlabel("Error Distance (km)")
    ax.set_ylabel("Geoparser")
    ax.legend(title="Matching Type", loc="upper right")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    return fig


def plot_accuracyATk_barchart(accuracies_at_k, thresholds):
    """
    Create a bar chart of Accuracy@k values for Edinburgh and Irchel and return the plot object.

    Parameters:
    - accuracies_at_k: Dictionary with accuracy values for both geoparsers (specific and loosely).
    - thresholds: List of k values used in the calculation.

    Returns:
    - fig: The matplotlib Figure object.
    """
    import pandas as pd

    # Prepare data for plotting
    data = []
    for geoparser in ["edinburgh", "irchel"]:
        for match_type in ["specific", "loosly"]:
            for i, threshold in enumerate(thresholds):
                data.append({
                    "Geoparser": f"{geoparser.title()} ({match_type.title()})",
                    "Threshold (km)": threshold / 1000,  # Convert to kilometers
                    "Accuracy": accuracies_at_k[f"{geoparser}_{match_type}"][i]
                })

    df = pd.DataFrame(data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Threshold (km)",
        y="Accuracy",
        hue="Geoparser",
        ax=ax
    )
    ax.set_title("Accuracy@k Comparison")
    ax.set_xlabel("Distance Threshold (km)")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Geoparser")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
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


def calculate_auc(data, distance_column, thresholds):
    max_threshold = max(thresholds)
    normalized_thresholds = [t / max_threshold for t in thresholds]

    accuracies = [accuracy_at_k(data, distance_column, k) for k in thresholds]

    auc = np.trapz(accuracies, normalized_thresholds)
    return auc, accuracies


def calculate_metrics(data, specific_column, loosly_column, total_column):
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
    results = {}
    for col, col_name in zip([specific_column, loosly_column], ["Specific", "Loosly"]):
        true_positive = len(data[data[col].notnull()])  # Korrekt erkannte Ergebnisse
        total_actual = len(data)  # Alle tatsächlichen Ergebnisse
        total_predicted = len(data[data[total_column].notnull()])  # Alle Vorhersagen (inkl. falsche)

        # Precision: Anteil korrekt erkannter unter allen vorhergesagten
        precision = true_positive / total_predicted if total_predicted > 0 else 0

        # Recall: Anteil korrekt erkannter unter allen tatsächlichen
        recall = true_positive / total_actual if total_actual > 0 else 0

        # F1-Score: Harmonie zwischen Precision und Recall
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[col_name] = {"Precision": precision, "Recall": recall, "F1": f1_score}

    return results


def distance_error(row, geoparser: Literal["irchel", "edinburgh"], loosely: bool = False):
    actual_Point = Point(x = row["longitude"], y = row["latitude"])
    if not actual_Point.x or not actual_Point.y:
        return None

    geoparser_dict = row[f"{geoparser}_NER_result_specific"] if not loosely else row[f"{geoparser}_NER_result_loosly"]
    if geoparser_dict is None:
        return None

    recognized_toponym = next(iter(geoparser_dict))


    pred_lon = geoparser_dict[recognized_toponym]["longitude"]
    pred_lat = geoparser_dict[recognized_toponym]["latitude"]

    if not pred_lon or not pred_lat:
        return None
    pred_lon = float(pred_lon)
    pred_lat = float(pred_lat)

    pred_Point = Point(x = pred_lon, y = pred_lat)
    if not pred_Point.x or not pred_Point.y:
        return None

    dist = actual_Point.distHaversine(pred_Point)
    return dist



def _match_toponym(row, geoparser: Literal["irchel", "edinburgh"], loosely: bool = False):
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
                if (loosely and toponym in key.lower()) or (not loosely and toponym == key.lower()):
                    return {key: value}
        return None

    elif geoparser == "edinburgh":
        dict_list = row.get("edinburgh_geoparser_res")
        if dict_list is None:
            return None

        for entry in dict_list:
            entry_name = entry["name"].lower()
            if (loosely and toponym in entry_name) or (not loosely and toponym == entry_name):
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

def get_matched_toponym_specific(row, geoparser: Literal["irchel", "edinburgh"] = "irchel"):
    """
    Get exact matches of toponyms from the geoparser data.
    """
    return _match_toponym(row, geoparser, loosely=False)

def get_matched_toponym_loosly(row, geoparser: Literal["irchel", "edinburgh"] = "irchel"):
    """
    Get loose matches (substring) of toponyms from the geoparser data.
    """
    return _match_toponym(row, geoparser, loosely=True)



def summarize_results(data, metrics_irchel, metrics_edinburgh, aucs,accuracies_at_k):
    # Irchel Summary
    total_rows = len(data)
    irchel_results = {
        "Total Rows": total_rows,
        "Rows with Results": len(data[data["irchel_geoparser_res"].notnull()]),
        "Percentage with Results": f"{len(data[data['irchel_geoparser_res'].notnull()]) / total_rows * 100:.2f}%",
        "Rows without Results": len(data[data["irchel_geoparser_res"].isnull()]),
        "Percentage without Results": f"{len(data[data['irchel_geoparser_res'].isnull()]) / total_rows * 100:.2f}%",
        "Correctly recognized Toponyms (Specific)": len(data[data["irchel_NER_result_specific"].notnull()]),
        "Correctly recognized Toponyms (Loosly)": len(data[data["irchel_NER_result_loosly"].notnull()]),
        "Percentage correctly recognized (Specific)": f"{len(data[data['irchel_NER_result_specific'].notnull()]) / total_rows * 100:.2f}%",
        "Percentage correctly recognized (Loosly)": f"{len(data[data['irchel_NER_result_loosly'].notnull()]) / total_rows * 100:.2f}%",
        "Median Distance Error (Specific)": f"{np.nanmedian(data['irchel_distance_error_specific']):.2f}" if not data['irchel_distance_error_specific'].isna().all() else "N/A",
        "Median Distance Error (Loosly)": f"{np.nanmedian(data['irchel_distance_error_loosly']):.2f}" if not data['irchel_distance_error_loosly'].isna().all() else "N/A",
        "AUC (Specific)": f"{aucs['irchel_specific']:.4f}",
        "AUC (Loosly)": f"{aucs['irchel_loosly']:.4f}",
        "Accuracy@1000 (Specific)": f"{accuracies_at_k['irchel_specific'][0]:.2f}",
        "Accuracy@5000 (Specific)": f"{accuracies_at_k['irchel_specific'][1]:.2f}",
        "Accuracy@10000 (Specific)": f"{accuracies_at_k['irchel_specific'][2]:.2f}",
        "Accuracy@50000 (Specific)": f"{accuracies_at_k['irchel_specific'][3]:.2f}",
        "Accuracy@100000 (Specific)": f"{accuracies_at_k['irchel_specific'][4]:.2f}",
        "Accuracy@161000 (Specific)": f"{accuracies_at_k['irchel_specific'][5]:.2f}",
        "Accuracy@1000 (Loosly)": f"{accuracies_at_k['irchel_loosly'][0]:.2f}",
        "Accuracy@5000 (Loosly)": f"{accuracies_at_k['irchel_loosly'][1]:.2f}",
        "Accuracy@10000 (Loosly)": f"{accuracies_at_k['irchel_loosly'][2]:.2f}",
        "Accuracy@50000 (Loosly)": f"{accuracies_at_k['irchel_loosly'][3]:.2f}",
        "Accuracy@100000 (Loosly)": f"{accuracies_at_k['irchel_loosly'][4]:.2f}",
        "Accuracy@161000 (Loosly)": f"{accuracies_at_k['irchel_loosly'][5]:.2f}",
        "Precision (Specific)": f"{metrics_irchel['Specific']['Precision']:.2f}",
        "Recall (Specific)": f"{metrics_irchel['Specific']['Recall']:.2f}",
        "F1 (Specific)": f"{metrics_irchel['Specific']['F1']:.2f}",
        "Precision (Loosly)": f"{metrics_irchel['Loosly']['Precision']:.2f}",
        "Recall (Loosly)": f"{metrics_irchel['Loosly']['Recall']:.2f}",
        "F1 (Loosly)": f"{metrics_irchel['Loosly']['F1']:.2f}",
    }
    irchel_summary = pd.DataFrame([irchel_results])

    # Edinburgh Summary
    edinburgh_results = {
        "Total Rows": total_rows,
        "Rows with Results": len(data[data["edinburgh_geoparser_res"].notnull()]),
        "Percentage with Results": f"{len(data[data['edinburgh_geoparser_res'].notnull()]) / total_rows * 100:.2f}%",
        "Rows without Results": len(data[data["edinburgh_geoparser_res"].isnull()]),
        "Percentage without Results": f"{len(data[data['edinburgh_geoparser_res'].isnull()]) / total_rows * 100:.2f}%",
        "Correctly recognized Toponyms (Specific)": len(data[data["edinburgh_NER_result_specific"].notnull()]),
        "Correctly recognized Toponyms (Loosly)": len(data[data["edinburgh_NER_result_loosly"].notnull()]),
        "Percentage correctly recognized (Specific)": f"{len(data[data['edinburgh_NER_result_specific'].notnull()]) / total_rows * 100:.2f}%",
        "Percentage correctly recognized (Loosly)": f"{len(data[data['edinburgh_NER_result_loosly'].notnull()]) / total_rows * 100:.2f}%",
        "Median Distance Error (Specific)": f"{np.nanmedian(data['edinburgh_distance_error_specific']):.2f}" if not data['edinburgh_distance_error_specific'].isna().all() else "N/A",
        "Median Distance Error (Loosly)": f"{np.nanmedian(data['edinburgh_distance_error_loosly']):.2f}" if not data['edinburgh_distance_error_loosly'].isna().all() else "N/A",
        "AUC (Specific)": f"{aucs['edinburgh_specific']:.4f}",
        "AUC (Loosly)": f"{aucs['edinburgh_loosly']:.4f}",
        "Accuracy@1000 (Specific)": f"{accuracies_at_k['edinburgh_specific'][0]:.2f}",
        "Accuracy@5000 (Specific)": f"{accuracies_at_k['edinburgh_specific'][1]:.2f}",
        "Accuracy@10000 (Specific)": f"{accuracies_at_k['edinburgh_specific'][2]:.2f}",
        "Accuracy@50000 (Specific)": f"{accuracies_at_k['edinburgh_specific'][3]:.2f}",
        "Accuracy@100000 (Specific)": f"{accuracies_at_k['edinburgh_specific'][4]:.2f}",
        "Accuracy@161000 (Specific)": f"{accuracies_at_k['edinburgh_specific'][5]:.2f}",
        "Accuracy@1000 (Loosly)": f"{accuracies_at_k['edinburgh_loosly'][0]:.2f}",
        "Accuracy@5000 (Loosly)": f"{accuracies_at_k['edinburgh_loosly'][1]:.2f}",
        "Accuracy@10000 (Loosly)": f"{accuracies_at_k['edinburgh_loosly'][2]:.2f}",
        "Accuracy@50000 (Loosly)": f"{accuracies_at_k['edinburgh_loosly'][3]:.2f}",
        "Accuracy@100000 (Loosly)": f"{accuracies_at_k['edinburgh_loosly'][4]:.2f}",
        "Accuracy@161000 (Loosly)": f"{accuracies_at_k['edinburgh_loosly'][5]:.2f}",
        "Precision (Specific)": f"{metrics_edinburgh['Specific']['Precision']:.2f}",
        "Recall (Specific)": f"{metrics_edinburgh['Specific']['Recall']:.2f}",
        "F1 (Specific)": f"{metrics_edinburgh['Specific']['F1']:.2f}",
        "Precision (Loosly)": f"{metrics_edinburgh['Loosly']['Precision']:.2f}",
        "Recall (Loosly)": f"{metrics_edinburgh['Loosly']['Recall']:.2f}",
        "F1 (Loosly)": f"{metrics_edinburgh['Loosly']['F1']:.2f}",
    }
    edinburgh_summary = pd.DataFrame([edinburgh_results])

    return {"Irchel Geoparser": irchel_summary, "Edinburgh Geoparser": edinburgh_summary}


def main(output_file, output_dir):
    data = pd.read_pickle(output_file)
    data['edinburgh_geoparser_res'] = data['edinburgh_geoparser_res'].apply(lambda x: None if isinstance(x, list) and len(x) == 0 else x)
    data['irchel_geoparser_res'] = data['irchel_geoparser_res'].apply(lambda x: None if isinstance(x, list) and len(x) == 0 else x)

    # Step 1: Apply NER Matching
    data["edinburgh_NER_result_specific"] = data.apply(lambda x: get_matched_toponym_specific(x, "edinburgh"), axis=1)
    data["edinburgh_NER_result_loosly"] = data.apply(lambda x: get_matched_toponym_loosly(x, "edinburgh"), axis=1)
    data["irchel_NER_result_specific"] = data.apply(lambda x: get_matched_toponym_specific(x, "irchel"), axis=1)
    data["irchel_NER_result_loosly"] = data.apply(lambda x: get_matched_toponym_loosly(x, "irchel"), axis=1)

    # Step 2: Calculate Metrics
    metrics_edinburgh = calculate_metrics(
        data,
        specific_column="edinburgh_NER_result_specific",
        loosly_column="edinburgh_NER_result_loosly",
        total_column="edinburgh_geoparser_res"
    )

    metrics_irchel = calculate_metrics(
        data,
        specific_column="irchel_NER_result_specific",
        loosly_column="irchel_NER_result_loosly",
        total_column="irchel_geoparser_res"
    )

    # Step 3: Calculate Distance Errors
    thresholds = [1000, 5000, 10000, 50_000, 100_000, 161000, 500000]
    data["edinburgh_distance_error_specific"] = data.apply(lambda x: distance_error(x, "edinburgh"), axis=1)
    data["irchel_distance_error_specific"] = data.apply(lambda x: distance_error(x, "irchel"), axis=1)
    data["edinburgh_distance_error_loosly"] = data.apply(lambda x: distance_error(x, "edinburgh", loosely=True), axis=1)
    data["irchel_distance_error_loosly"] = data.apply(lambda x: distance_error(x, "irchel", loosely=True), axis=1)

    # Step 4: Calculate AUC
    edinburgh_auc_specific, edinburgh_accuracies_specific = calculate_auc(data, "edinburgh_distance_error_specific", thresholds)
    edinburgh_auc_loosly, edinburgh_accuracies_loosly = calculate_auc(data, "edinburgh_distance_error_loosly", thresholds)
    irchel_auc_specific, irchel_accuracies_specific = calculate_auc(data, "irchel_distance_error_specific", thresholds)
    irchel_auc_loosly, irchel_accuracies_loosly = calculate_auc(data, "irchel_distance_error_loosly", thresholds)

    # Step 5: Summarize Results
    summary_df = summarize_results(
        data,
        metrics_irchel,
        metrics_edinburgh,
        aucs={
            "edinburgh_specific": edinburgh_auc_specific,
            "edinburgh_loosly": edinburgh_auc_loosly,
            "irchel_specific": irchel_auc_specific,
            "irchel_loosly": irchel_auc_loosly
        },
        accuracies_at_k={
            "edinburgh_specific": edinburgh_accuracies_specific,
            "edinburgh_loosly": edinburgh_accuracies_loosly,
            "irchel_specific": irchel_accuracies_specific,
            "irchel_loosly": irchel_accuracies_loosly
        }
    )
    for name, df in summary_df.items():
        print(f"Summary for {name}:")
        for col in df.columns:
            print(f"{col}: {df[col].values[0]}")

    # Step 6: Plot and Save
        plots = {
            "Error Distance Boxplots": plot_error_distribution_boxplots(data),
            "Accuracy@k Comparison": plot_accuracyATk_barchart(
                {
                    "edinburgh_specific": edinburgh_accuracies_specific,
                    "edinburgh_loosly": edinburgh_accuracies_loosly,
                    "irchel_specific": irchel_accuracies_specific,
                    "irchel_loosly": irchel_accuracies_loosly,
                },
                thresholds
            )
        }

        for name, fig in plots.items():
            fig.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pdf"), bbox_inches='tight')
            plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python result_analysis.py <output.pkl> <output_dir>")
        print("Default Inputs taken: python result_analysis.py result/europe.pkl result/europe")
        output_file = "result/europe.pkl"
    else:
        output_file = sys.argv[1]

    output_dir = sys.argv[2] if len(sys.argv) > 2 else output_file.replace(".pkl", "")

    main(output_file, output_dir)