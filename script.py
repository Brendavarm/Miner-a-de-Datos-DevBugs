#!/usr/bin/env python3
"""EDA completo y legible para MovieLens ml-latest-small.

Responde:

1. ¿Qué datos tengo?
2. ¿Hay datos faltantes?
3. ¿Existen valores atípicos?
4. ¿Cómo se distribuyen los datos?
5. ¿Qué relaciones existen entre variables?

Uso:

    pip install pandas matplotlib seaborn

    python eda_movielens_legible.py \
        --data-dir ./ml-latest-small \
        --output-dir ./output
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# Utilidades
# ============================================================


def line(character="=", length=90):
    return character * length


def section(title):
    return f"\n{line()}\n{title}\n{line()}"


def save_plot(path, title):
    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/D"

    if decimals == 0:
        return f"{value:,.0f}"

    return f"{value:,.{decimals}f}"


def format_percent(value):
    return f"{value:.2f}%"


# ============================================================
# Carga de datos
# ============================================================


def load_data(data_dir):
    files = {
        "movies": "movies.csv",
        "ratings": "ratings.csv",
        "tags": "tags.csv",
        "links": "links.csv",
    }

    data = {}

    for name, filename in files.items():
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo requerido:\n{path}"
            )

        data[name] = pd.read_csv(
            path,
            encoding="utf-8",
        )

    return data


# ============================================================
# Preparación de datos
# ============================================================


def prepare_data(data):
    ratings = data["ratings"].copy()
    tags = data["tags"].copy()
    movies = data["movies"].copy()
    links = data["links"].copy()

    # Convertir timestamp a fecha
    for dataframe in (ratings, tags):
        dataframe["datetime"] = pd.to_datetime(
            dataframe["timestamp"],
            unit="s",
            errors="coerce",
            utc=True,
        )

        dataframe["year"] = dataframe["datetime"].dt.year

    # Normalizar etiquetas
    tags["tag_normalized"] = (
        tags["tag"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    tags["tag_valid"] = tags["tag_normalized"].ne("")

    # Contar géneros por película
    movies["genres"] = movies["genres"].fillna("")

    movies["genre_count"] = movies["genres"].apply(
        lambda genres: 0
        if genres in ("", "(no genres listed)")
        else len(genres.split("|"))
    )

    return ratings, tags, movies, links


# ============================================================
# Pregunta 1: ¿Qué datos tengo?
# ============================================================


def create_data_summary(data, output_dir):
    rows = []

    descriptions = {
        "movies": "Información de películas y géneros",
        "ratings": "Calificaciones dadas por usuarios",
        "tags": "Etiquetas asignadas por usuarios",
        "links": "Enlaces externos de películas",
    }

    for name, dataframe in data.items():
        rows.append(
            {
                "archivo": f"{name}.csv",
                "descripcion": descriptions[name],
                "filas": len(dataframe),
                "columnas": len(dataframe.columns),
                "columnas_lista": ", ".join(dataframe.columns),
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        output_dir / "data_summary.csv",
        index=False,
        encoding="utf-8",
    )

    return summary


# ============================================================
# Pregunta 2: ¿Hay datos faltantes?
# ============================================================


def create_missing_report(data, output_dir):
    rows = []

    for name, dataframe in data.items():
        for column in dataframe.columns:
            missing_count = int(dataframe[column].isna().sum())

            missing_percent = (
                missing_count / len(dataframe) * 100
                if len(dataframe) > 0
                else 0
            )

            empty_count = 0

            if pd.api.types.is_object_dtype(dataframe[column]):
                empty_count = int(
                    dataframe[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                )

            rows.append(
                {
                    "archivo": f"{name}.csv",
                    "columna": column,
                    "faltantes": missing_count,
                    "porcentaje_faltante": round(
                        missing_percent,
                        4,
                    ),
                    "cadenas_vacias": empty_count,
                }
            )

    report = pd.DataFrame(rows)

    report.to_csv(
        output_dir / "missing_values.csv",
        index=False,
        encoding="utf-8",
    )

    return report


# ============================================================
# Pregunta 3: ¿Existen valores atípicos?
# ============================================================


def calculate_iqr(series):
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "q1": 0,
            "q3": 0,
            "iqr": 0,
            "limite_inferior": 0,
            "limite_superior": 0,
            "atipicos": 0,
            "porcentaje_atipicos": 0,
        }

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (
        (numeric < lower) |
        (numeric > upper)
    )

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": lower,
        "limite_superior": upper,
        "atipicos": int(outlier_mask.sum()),
        "porcentaje_atipicos": outlier_mask.mean() * 100,
    }


def create_outlier_report(
    ratings,
    tags,
    movies,
    output_dir,
):
    valid_tags = tags.loc[tags["tag_valid"]].copy()

    variables = {
        "Valor de rating": ratings["rating"],
        "Ratings por usuario": ratings.groupby("userId").size(),
        "Ratings por película": ratings.groupby("movieId").size(),
        "Tags por usuario": valid_tags.groupby("userId").size(),
        "Tags por película": valid_tags.groupby("movieId").size(),
        "Usos por etiqueta": valid_tags.groupby(
            "tag_normalized"
        ).size(),
        "Géneros por película": movies["genre_count"],
    }

    rows = []

    for variable, values in variables.items():
        result = calculate_iqr(values)
        rows.append(
            {
                "variable": variable,
                **result,
            }
        )

    report = pd.DataFrame(rows)

    report.to_csv(
        output_dir / "outliers_iqr.csv",
        index=False,
        encoding="utf-8",
    )

    tag_frequency = variables["Usos por etiqueta"]

    if not tag_frequency.empty:
        plt.figure(figsize=(11, 5))

        sns.boxplot(
            x=tag_frequency.values,
            color="darkcyan",
        )

        plt.xlabel("Número de usos")

        save_plot(
            output_dir / "outliers_tag_frequency.png",
            "Valores atípicos en la frecuencia de etiquetas",
        )

    return report


# ============================================================
# Pregunta 4: ¿Cómo se distribuyen los datos?
# ============================================================


def create_distribution_plots(
    ratings,
    tags,
    movies,
    output_dir,
):
    valid_tags = tags.loc[tags["tag_valid"]].copy()
    tag_counts = valid_tags["tag_normalized"].value_counts()

    # Top 20 etiquetas
    if not tag_counts.empty:
        plt.figure(figsize=(12, 8))

        tag_counts.head(20).sort_values().plot(
            kind="barh",
            color="darkcyan",
        )

        plt.xlabel("Número de usos")
        plt.ylabel("Etiqueta")

        save_plot(
            output_dir / "top_20_tags.png",
            "Top 20 etiquetas más utilizadas",
        )

        # Distribución de etiquetas
        plt.figure(figsize=(10, 6))

        sns.histplot(
            tag_counts,
            bins=30,
            kde=True,
            color="steelblue",
        )

        plt.xlabel("Usos por etiqueta")
        plt.ylabel("Cantidad de etiquetas")

        save_plot(
            output_dir / "distribution_tag_frequency.png",
            "Distribución de usos por etiqueta",
        )

    # Distribución de calificaciones
    plt.figure(figsize=(10, 6))

    sns.histplot(
        ratings["rating"],
        discrete=True,
        color="seagreen",
    )

    plt.xlabel("Calificación")
    plt.ylabel("Cantidad de calificaciones")

    save_plot(
        output_dir / "distribution_ratings.png",
        "Distribución de calificaciones",
    )

    # Ratings por año
    ratings_by_year = (
        ratings
        .dropna(subset=["year"])
        .groupby("year")
        .size()
    )

    if not ratings_by_year.empty:
        plt.figure(figsize=(11, 6))

        ratings_by_year.plot(
            kind="bar",
            color="slateblue",
        )

        plt.xlabel("Año")
        plt.ylabel("Cantidad de ratings")

        save_plot(
            output_dir / "ratings_by_year.png",
            "Cantidad de ratings por año",
        )

    # Tags por año
    tags_by_year = (
        valid_tags
        .dropna(subset=["year"])
        .groupby("year")
        .size()
    )

    if not tags_by_year.empty:
        plt.figure(figsize=(11, 6))

        tags_by_year.plot(
            kind="bar",
            color="darkorange",
        )

        plt.xlabel("Año")
        plt.ylabel("Cantidad de tags")

        save_plot(
            output_dir / "tags_by_year.png",
            "Cantidad de tags por año",
        )

    # Géneros
    genre_counts = {}

    for genres in movies["genres"]:
        if not genres or genres == "(no genres listed)":
            continue

        for genre in genres.split("|"):
            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
            )

    if genre_counts:
        genre_series = pd.Series(
            genre_counts
        ).sort_values()

        plt.figure(figsize=(10, 7))

        genre_series.plot(
            kind="barh",
            color="mediumpurple",
        )

        plt.xlabel("Cantidad de películas")
        plt.ylabel("Género")

        save_plot(
            output_dir / "movie_genres.png",
            "Distribución de películas por género",
        )


# ============================================================
# Pregunta 5: Relaciones entre variables
# ============================================================


def create_relationship_analysis(
    ratings,
    tags,
    output_dir,
):
    valid_tags = tags.loc[tags["tag_valid"]].copy()

    ratings_per_user = ratings.groupby("userId").size()
    tags_per_user = valid_tags.groupby("userId").size()

    ratings_per_movie = ratings.groupby("movieId").size()
    tags_per_movie = valid_tags.groupby("movieId").size()

    # Actividad por usuario
    activity = pd.concat(
        [
            ratings_per_user.rename("ratings_count"),
            tags_per_user.rename("tags_count"),
        ],
        axis=1,
    ).fillna(0)

    activity.index.name = "userId"

    activity.reset_index().to_csv(
        output_dir / "user_activity.csv",
        index=False,
        encoding="utf-8",
    )

    # Actividad por película
    movie_activity = pd.concat(
        [
            ratings_per_movie.rename("ratings_count"),
            tags_per_movie.rename("tags_count"),
        ],
        axis=1,
    ).fillna(0)

    movie_activity.index.name = "movieId"

    movie_activity.reset_index().to_csv(
        output_dir / "movie_activity.csv",
        index=False,
        encoding="utf-8",
    )

    # Gráfico de relación por usuario
    if not activity.empty:
        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=activity,
            x="ratings_count",
            y="tags_count",
            alpha=0.7,
            color="crimson",
        )

        plt.xlabel("Ratings realizados por usuario")
        plt.ylabel("Tags creados por usuario")

        save_plot(
            output_dir / "relationship_user_activity.png",
            "Relación entre ratings y tags por usuario",
        )

    correlation = activity["ratings_count"].corr(
        activity["tags_count"]
    )

    return {
        "ratings_per_user": ratings_per_user.sort_values(
            ascending=False
        ),
        "tags_per_user": tags_per_user.sort_values(
            ascending=False
        ),
        "ratings_per_movie": ratings_per_movie.sort_values(
            ascending=False
        ),
        "tags_per_movie": tags_per_movie.sort_values(
            ascending=False
        ),
        "activity_correlation": correlation,
    }


# ============================================================
# Construcción del informe
# ============================================================


def add_top_items(
    report,
    title,
    series,
    item_label,
):
    report.append(title)

    if series.empty:
        report.append("  No hay datos disponibles.")
        return

    for position, (item, value) in enumerate(
        series.head(10).items(),
        start=1,
    ):
        report.append(
            f"  {position:>2}. "
            f"{item_label}: {item} | "
            f"Cantidad: {value:,.0f}"
        )


def build_readable_report(
    data,
    ratings,
    tags,
    movies,
    links,
    output_dir,
):
    valid_tags = tags.loc[tags["tag_valid"]].copy()
    tag_counts = valid_tags["tag_normalized"].value_counts()

    data_summary = create_data_summary(
        data,
        output_dir,
    )

    missing = create_missing_report(
        data,
        output_dir,
    )

    outliers = create_outlier_report(
        ratings,
        tags,
        movies,
        output_dir,
    )

    relationships = create_relationship_analysis(
        ratings,
        tags,
        output_dir,
    )

    total_tags = len(tags)
    valid_tag_count = len(valid_tags)

    # Cantidad real de etiquetas distintas
    unique_tags = tag_counts.size

    report = []

    report.extend(
        [
            "INFORME DE ANÁLISIS EXPLORATORIO DE DATOS",
            "DATASET: MOVIELENS ML-LATEST-SMALL",
            line(),
            "Este informe responde las cinco preguntas principales del EDA.",
            "Los valores fueron calculados directamente desde los archivos CSV.",
        ]
    )

    # --------------------------------------------------------
    # Pregunta 1
    # --------------------------------------------------------

    report.append(
        section("1. ¿QUÉ DATOS TENGO?")
    )

    report.append(
        "El dataset contiene información de películas, "
        "calificaciones, etiquetas y enlaces externos."
    )

    report.append("")

    for _, row in data_summary.iterrows():
        report.extend(
            [
                f"Archivo: {row['archivo']}",
                f"  Descripción: {row['descripcion']}",
                f"  Filas: {row['filas']:,}",
                f"  Columnas: {row['columnas']:,}",
                f"  Variables: {row['columnas_lista']}",
                "",
            ]
        )

    report.append("Interpretación:")

    report.append(
        f"  El conjunto contiene "
        f"{len(data['ratings']):,} calificaciones, "
        f"{valid_tag_count:,} aplicaciones válidas de etiquetas "
        f"y {len(data['movies']):,} películas."
    )

    # --------------------------------------------------------
    # Pregunta 2
    # --------------------------------------------------------

    report.append(
        section("2. ¿HAY DATOS FALTANTES?")
    )

    problems = missing[
        (missing["faltantes"] > 0) |
        (missing["cadenas_vacias"] > 0)
    ]

    if problems.empty:
        report.append(
            "Resultado: no se encontraron valores faltantes "
            "ni campos vacíos."
        )
    else:
        report.append(
            "Se encontraron los siguientes problemas:"
        )

        for _, row in problems.iterrows():
            report.append(
                f"  - {row['archivo']} / "
                f"{row['columna']}: "
                f"{row['faltantes']:,} faltantes "
                f"({row['porcentaje_faltante']:.2f}%) "
                f"y {row['cadenas_vacias']:,} cadenas vacías."
            )

    report.append("")
    report.append(
        "Archivo detallado generado: missing_values.csv"
    )

    # --------------------------------------------------------
    # Pregunta 3
    # --------------------------------------------------------

    report.append(
        section("3. ¿EXISTEN VALORES ATÍPICOS?")
    )

    report.append(
        "Se utilizó el método del rango intercuartílico (IQR). "
        "Un valor se considera atípico si queda fuera de "
        "Q1 - 1.5×IQR o Q3 + 1.5×IQR."
    )

    report.append("")

    for _, row in outliers.iterrows():
        status = "Sí" if row["atipicos"] > 0 else "No"

        report.extend(
            [
                f"Variable: {row['variable']}",
                f"  ¿Tiene atípicos?: {status}",
                f"  Q1: {format_number(row['q1'])}",
                f"  Q3: {format_number(row['q3'])}",
                f"  IQR: {format_number(row['iqr'])}",
                (
                    "  Límite inferior: "
                    f"{format_number(row['limite_inferior'])}"
                ),
                (
                    "  Límite superior: "
                    f"{format_number(row['limite_superior'])}"
                ),
                f"  Cantidad de atípicos: {row['atipicos']:,}",
                (
                    "  Porcentaje de atípicos: "
                    f"{format_percent(row['porcentaje_atipicos'])}"
                ),
                "",
            ]
        )

    report.append(
        "Archivo detallado generado: outliers_iqr.csv"
    )

    # --------------------------------------------------------
    # Pregunta 4
    # --------------------------------------------------------

    report.append(
        section("4. ¿CÓMO SE DISTRIBUYEN LOS DATOS?")
    )

    report.append("Distribución de etiquetas:")
    report.append(
        f"  Aplicaciones totales: {total_tags:,}"
    )
    report.append(
        f"  Aplicaciones válidas: {valid_tag_count:,}"
    )
    report.append(
        f"  Etiquetas únicas: {unique_tags:,}"
    )

    if not tag_counts.empty:
        report.extend(
            [
                (
                    "  Promedio de usos por etiqueta: "
                    f"{tag_counts.mean():.2f}"
                ),
                (
                    "  Mediana de usos por etiqueta: "
                    f"{tag_counts.median():.2f}"
                ),
                (
                    "  Desviación estándar: "
                    f"{tag_counts.std():.2f}"
                ),
                (
                    "  Menor frecuencia: "
                    f"{tag_counts.min():,}"
                ),
                (
                    "  Mayor frecuencia: "
                    f"{tag_counts.max():,}"
                ),
            ]
        )

        for number in (10, 20, 50, 100):
            amount = tag_counts.head(number).sum()

            percentage = (
                amount / valid_tag_count * 100
                if valid_tag_count
                else 0
            )

            report.append(
                f"  Top {number} etiquetas: "
                f"{amount:,} usos "
                f"({percentage:.2f}% del total)"
            )

    report.append("")
    report.append("Distribución de calificaciones:")
    report.append(
        f"  Promedio: {ratings['rating'].mean():.2f}"
    )
    report.append(
        f"  Mediana: {ratings['rating'].median():.2f}"
    )
    report.append(
        f"  Mínima: {ratings['rating'].min():.1f}"
    )
    report.append(
        f"  Máxima: {ratings['rating'].max():.1f}"
    )

    report.append("")
    report.append(
        "Interpretación: revise los gráficos generados "
        "para observar si las distribuciones son uniformes, "
        "concentradas o de cola larga."
    )

    # --------------------------------------------------------
    # Pregunta 5
    # --------------------------------------------------------

    report.append(
        section("5. ¿QUÉ RELACIONES EXISTEN ENTRE VARIABLES?")
    )

    report.append(
        "Se analizaron relaciones mediante agrupaciones "
        "por usuario y película."
    )

    report.append("")
    report.append("Cobertura del dataset:")
    report.append(
        f"  Usuarios con ratings: "
        f"{ratings['userId'].nunique():,}"
    )
    report.append(
        f"  Usuarios con tags: "
        f"{valid_tags['userId'].nunique():,}"
    )
    report.append(
        f"  Películas con ratings: "
        f"{ratings['movieId'].nunique():,}"
    )
    report.append(
        f"  Películas con tags: "
        f"{valid_tags['movieId'].nunique():,}"
    )

    report.append("")

    add_top_items(
        report,
        "Usuarios con más ratings:",
        relationships["ratings_per_user"],
        "userId",
    )

    report.append("")

    add_top_items(
        report,
        "Películas con más ratings:",
        relationships["ratings_per_movie"],
        "movieId",
    )

    report.append("")

    add_top_items(
        report,
        "Usuarios con más tags:",
        relationships["tags_per_user"],
        "userId",
    )

    report.append("")

    add_top_items(
        report,
        "Películas con más tags:",
        relationships["tags_per_movie"],
        "movieId",
    )

    report.append("")

    correlation = relationships["activity_correlation"]

    if pd.isna(correlation):
        report.append(
            "Correlación entre ratings y tags: no disponible."
        )
    else:
        report.append(
            "Correlación entre cantidad de ratings y tags "
            f"por usuario: {correlation:.4f}"
        )

    report.append(
        "Interpretación: una correlación positiva indica "
        "que los usuarios que califican más películas tienden "
        "también a crear más etiquetas."
    )

    report.append(
        "Nota: userId y movieId son identificadores, por lo "
        "que no se debe interpretar su correlación numérica directa."
    )

    # --------------------------------------------------------
    # Archivos generados
    # --------------------------------------------------------

    report.append(
        section("ARCHIVOS GENERADOS")
    )

    report.append("Reportes CSV:")

    report.extend(
        [
            "  - data_summary.csv",
            "  - missing_values.csv",
            "  - outliers_iqr.csv",
            "  - user_activity.csv",
            "  - movie_activity.csv",
        ]
    )

    report.append("")
    report.append("Gráficos PNG:")

    for image in sorted(output_dir.glob("*.png")):
        report.append(f"  - {image.name}")

    report.extend(
        [
            "",
            "Referencia del dataset:",
            "F. Maxwell Harper y Joseph A. Konstan, 2015.",
            "The MovieLens Datasets: History and Context.",
            line(),
        ]
    )

    report_path = output_dir / "eda_report_legible.txt"

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )


# ============================================================
# Programa principal
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Genera un informe EDA legible de MovieLens."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./ml-latest-small"),
        help="Directorio que contiene los archivos CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directorio donde se guardarán los resultados.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(line())
    print("EDA DE MOVIELENS ML-LATEST-SMALL")
    print(line())
    print(
        f"Directorio de datos: "
        f"{args.data_dir.resolve()}"
    )
    print(
        f"Directorio de salida: "
        f"{args.output_dir.resolve()}"
    )
    print()

    try:
        print("[1/4] Cargando archivos CSV...")
        data = load_data(args.data_dir)

        print("[2/4] Preparando datos...")
        ratings, tags, movies, links = prepare_data(data)

        print("[3/4] Generando gráficos...")
        create_distribution_plots(
            ratings,
            tags,
            movies,
            args.output_dir,
        )

        print("[4/4] Generando reportes...")
        build_readable_report(
            data,
            ratings,
            tags,
            movies,
            links,
            args.output_dir,
        )

    except Exception as error:
        print()
        print("ERROR: el análisis no pudo completarse.")
        print(f"Detalle: {error}")
        raise

    print()
    print(line())
    print("ANÁLISIS COMPLETADO CORRECTAMENTE")
    print(line())
    print(
        "Informe principal: "
        f"{args.output_dir / 'eda_report_legible.txt'}"
    )
    print(
        f"Resultados guardados en: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()