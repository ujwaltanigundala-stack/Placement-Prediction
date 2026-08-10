from pathlib import Path
import sys
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from models.load import load_data, summarize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.EDA import (PLOT_FILENAMES, generate_all_plots, get_bivariate_stats, get_correlation_stats,
                     get_multivariate_stats, get_overview_stats, get_univariate_stats)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Path(app.root_path) / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)


def current_dataset_path():
    """Use the uploaded CSV for this browser session, otherwise use the default CSV."""
    uploaded_path = session.get("dataset_path")
    if uploaded_path and Path(uploaded_path).is_file():
        return Path(uploaded_path)
    return None


@app.route("/")
def index_page():
    return redirect(url_for("load_page"))


@app.route("/load", methods=["GET", "POST"])
def load_page():
    if request.method == "POST":
        dataset = request.files.get("dataset")
        if not dataset or not dataset.filename:
            flash("Choose a CSV file before uploading.", "error")
        elif not dataset.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "error")
        else:
            filename = f"{uuid4().hex}_{secure_filename(dataset.filename)}"
            saved_path = app.config["UPLOAD_FOLDER"] / filename
            dataset.save(saved_path)
            try:
                load_data(saved_path)
            except Exception as error:
                saved_path.unlink(missing_ok=True)
                flash(f"The CSV could not be read: {error}", "error")
            else:
                session["dataset_path"] = str(saved_path)
                session["dataset_name"] = dataset.filename
                session.pop("plots_dataset", None)
                flash("Dataset uploaded successfully. Open EDA to analyse it.", "success")
                return redirect(url_for("load_page"))

    df = load_data(current_dataset_path())
    summary = summarize(df)
    return render_template(
        "load.html",
        shape=summary["shape"],
        columns=summary["columns"],
        preview=df.head(10).to_dict(orient="records"),
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
    )


@app.route("/eda")
@app.route("/eda/<section>")
def eda_page(section="overview"):
    if section != "overview":
        return redirect(url_for("eda_page"))
    df = load_data(current_dataset_path())
    uploaded_path = current_dataset_path()
    if uploaded_path and session.get("plots_dataset") != str(uploaded_path):
        generate_all_plots(df)
        session["plots_dataset"] = str(uploaded_path)
    elif not uploaded_path and not all((config.PLOTS_DIR / plot).exists() for plot in PLOT_FILENAMES):
        generate_all_plots(df)
    return render_template(
        "eda.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        overview=get_overview_stats(df),
        univariate=get_univariate_stats(df),
        bivariate=get_bivariate_stats(df),
        multivariate=get_multivariate_stats(df),
        correlation=get_correlation_stats(df),
        plots=[url_for("static", filename=f"plots/{plot}") for plot in PLOT_FILENAMES if (config.PLOTS_DIR / plot).exists()],
    )


@app.route("/feature-engg")
def feature_engg_page():
    return render_template("feature_engg.html")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
