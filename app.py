import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PLOT_FOLDER = BASE_DIR / "static" / "plots"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_file(file_path: Path, extension: str) -> pd.DataFrame:
    # One loading path keeps error messages consistent for non-technical users.
    try:
        if extension == "csv":
            try:
                return pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding="latin1")
        return pd.read_excel(file_path, engine="openpyxl")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "This file uses an unsupported text encoding. Please save it as UTF-8 CSV or XLSX."
        ) from exc
    except Exception as exc:
        if extension == "xlsx":
            raise ValueError(
                "The Excel file could not be read. It may be corrupted, password-protected, or not a valid XLSX file."
            ) from exc
        raise ValueError("The file could not be read. Please check the format and try again.") from exc


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Predictable names make downstream tables and charts easier to understand.
    cleaned.columns = [
        str(column).strip() if str(column).strip() else f"Column_{index + 1}"
        for index, column in enumerate(cleaned.columns)
    ]

    cleaned = cleaned.dropna(axis=1, how="all")
    if cleaned.shape[1] == 0:
        raise ValueError("All columns are empty. Please upload a file with actual data.")

    for column in cleaned.columns:
        series = cleaned[column]
        if pd.api.types.is_numeric_dtype(series):
            continue

        normalized = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

        non_empty_count = normalized.notna().sum()
        if non_empty_count == 0:
            cleaned[column] = normalized
            continue

        numeric_candidate = pd.to_numeric(normalized, errors="coerce")
        numeric_ratio = numeric_candidate.notna().sum() / non_empty_count

        # Mixed columns are converted only when most values are truly numeric.
        if numeric_ratio >= 0.7:
            cleaned[column] = numeric_candidate
        else:
            cleaned[column] = normalized

    numeric_columns = cleaned.select_dtypes(include="number").columns
    text_columns = cleaned.columns.difference(numeric_columns)

    for column in numeric_columns:
        if cleaned[column].dropna().empty:
            cleaned[column] = cleaned[column].fillna(0)
        else:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .fillna("Unknown")
        )

    return cleaned


def generate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    try:
        stats = df.describe(include="all", datetime_is_numeric=True).transpose().fillna("")
    except TypeError:
        stats = df.describe(include="all").transpose().fillna("")
    for column in stats.columns:
        if pd.api.types.is_numeric_dtype(stats[column]):
            stats[column] = stats[column].round(2)
    return stats


def create_plots(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    plot_files: list[str] = []
    messages: list[str] = []
    run_id = uuid.uuid4().hex

    numeric_columns = list(df.select_dtypes(include="number").columns)
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    if numeric_columns:
        numeric_column = numeric_columns[0]
        numeric_values = df[numeric_column].dropna()

        if not numeric_values.empty:
            fig, axis = plt.subplots(figsize=(7, 4))
            bins = min(20, max(5, int(numeric_values.nunique())))
            axis.hist(numeric_values, bins=bins, color="#2563eb", edgecolor="#ffffff")
            axis.set_title(f"Distribution of {numeric_column}")
            axis.set_xlabel(numeric_column)
            axis.set_ylabel("Count")
            axis.grid(alpha=0.2, linestyle="--")
            fig.tight_layout()

            histogram_file = f"{run_id}_histogram.png"
            fig.savefig(PLOT_FOLDER / histogram_file)
            plt.close(fig)
            plot_files.append(histogram_file)
        else:
            messages.append("No usable numeric values were found for a histogram.")
    else:
        messages.append("No numeric columns were found, so the histogram was skipped.")

    if categorical_columns:
        categorical_column = min(
            categorical_columns,
            key=lambda column: df[column].astype("string").nunique(dropna=True),
        )

        counts = (
            df[categorical_column]
            .astype("string")
            .replace({"<NA>": "Unknown", "nan": "Unknown"})
            .fillna("Unknown")
            .value_counts()
            .head(10)
        )

        if not counts.empty:
            fig, axis = plt.subplots(figsize=(7, 4))
            counts.sort_values().plot(kind="barh", ax=axis, color="#16a34a")
            axis.set_title(f"Top values in {categorical_column}")
            axis.set_xlabel("Count")
            axis.set_ylabel(categorical_column)
            axis.grid(alpha=0.2, linestyle="--")
            fig.tight_layout()

            bar_file = f"{run_id}_bar.png"
            fig.savefig(PLOT_FOLDER / bar_file)
            plt.close(fig)
            plot_files.append(bar_file)
        else:
            messages.append("No usable text values were found for a bar chart.")
    else:
        messages.append("No categorical columns were found, so the bar chart was skipped.")

    return plot_files, messages


def generate_insight(df: pd.DataFrame) -> str:
    row_count, column_count = df.shape
    lines: list[str] = [f"This report covers {row_count} records across {column_count} fields."]

    if row_count < 5:
        lines.append(
            "The file is very small, so treat this as an early signal rather than a final decision guide."
        )

    numeric_columns = list(df.select_dtypes(include="number").columns)
    if numeric_columns:
        primary_numeric = numeric_columns[0]
        values = df[primary_numeric].dropna()

        if not values.empty:
            middle = values.median()
            low = values.min()
            high = values.max()
            lines.append(
                f"For {primary_numeric}, the middle value is {middle:,.2f}, with a range from {low:,.2f} to {high:,.2f}."
            )

            if middle != 0 and high > middle * 2:
                lines.append(
                    "A few records are much higher than the rest, which may be worth reviewing as unusual spikes."
                )
            elif middle != 0 and low < middle * 0.5:
                lines.append(
                    "Some records sit far below the middle value, suggesting a noticeable gap inside this dataset."
                )

    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    if categorical_columns:
        primary_categorical = categorical_columns[0]
        top_counts = df[primary_categorical].astype("string").fillna("Unknown").value_counts()

        if not top_counts.empty:
            top_label = top_counts.index[0]
            top_count = int(top_counts.iloc[0])
            share = (top_count / row_count) * 100 if row_count else 0
            lines.append(
                f"In {primary_categorical}, '{top_label}' appears most often ({share:.1f}% of records), showing a clear pattern."
            )

    if len(lines) == 1:
        lines.append("The file has limited variation, but the cleaned data is ready for a deeper manual review.")

    return " ".join(lines)


@app.errorhandler(413)
def file_too_large(_: Exception):
    return (
        render_template(
            "index.html",
            error=f"File is too large. Please upload a file under {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        ),
        413,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "preview_html": None,
        "stats_html": None,
        "plot_files": [],
        "plot_messages": [],
        "insight": None,
    }

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if uploaded_file is None:
            context["error"] = "Please choose a file before submitting."
            return render_template("index.html", **context)

        if uploaded_file.filename == "":
            context["error"] = "No file selected. Please upload a CSV or XLSX file."
            return render_template("index.html", **context)

        if not allowed_file(uploaded_file.filename):
            context["error"] = "Unsupported file type. Please upload a CSV or XLSX file."
            return render_template("index.html", **context)

        extension = uploaded_file.filename.rsplit(".", 1)[1].lower()
        temp_filename = f"{uuid.uuid4().hex}.{extension}"
        temp_file_path = UPLOAD_FOLDER / temp_filename

        try:
            uploaded_file.save(temp_file_path)
            if not temp_file_path.exists() or temp_file_path.stat().st_size == 0:
                raise ValueError("The uploaded file is empty. Please provide a file with data.")

            raw_df = load_file(temp_file_path, extension)
            if raw_df.empty:
                raise ValueError("The file has no rows to analyze.")

            cleaned_df = clean_data(raw_df)
            if cleaned_df.empty:
                raise ValueError("No usable data remained after cleaning.")

            stats_df = generate_statistics(cleaned_df)
            plot_files, plot_messages = create_plots(cleaned_df)
            insight = generate_insight(cleaned_df)

            context["preview_html"] = cleaned_df.head(10).to_html(
                index=False,
                classes="table",
                border=0,
                justify="left",
            )
            context["stats_html"] = stats_df.to_html(
                classes="table",
                border=0,
                justify="left",
            )
            context["plot_files"] = plot_files
            context["plot_messages"] = plot_messages
            context["insight"] = insight

        except pd.errors.EmptyDataError:
            context["error"] = "The file appears empty or unreadable. Please check the contents and try again."
        except ValueError as exc:
            context["error"] = str(exc)
        except Exception:
            context["error"] = "Something went wrong while processing the file. Please check the file and try again."
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
