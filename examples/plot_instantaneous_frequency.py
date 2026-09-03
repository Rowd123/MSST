"""Plot instantaneous-frequency estimates from a real signal stored as CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from instantaneous_frequency import ridge_frequency
from msst_y import msst_y


def load_csv(path: Path, column: int, skip_rows: int) -> np.ndarray:
    """Load one numeric signal column from a CSV file."""

    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        for line_number, row in enumerate(rows, start=1):
            if line_number <= skip_rows or not row:
                continue
            try:
                values.append(float(row[column]))
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"invalid value in column {column} at CSV line {line_number}"
                ) from error
    if len(values) < 2:
        raise ValueError("the CSV signal must contain at least two samples")
    return np.asarray(values)


def build_figure(
    signal: np.ndarray, sample_rate: float, hlength: int, iterations: list[int]
) -> go.Figure:
    """Build a signal/instantaneous-frequency comparison figure."""

    time = np.arange(signal.size) / sample_rate
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Signal mesuré", "Fréquence instantanée estimée"),
    )
    figure.add_trace(
        go.Scatter(x=time, y=signal, name="Signal", line={"color": "#334155"}),
        row=1,
        col=1,
    )

    _, stft = msst_y(signal, hlength, 1)
    figure.add_trace(
        go.Scatter(
            x=time,
            y=ridge_frequency(stft, sample_rate),
            name="STFT (itération 0)",
        ),
        row=2,
        col=1,
    )
    for iteration in iterations:
        transform, _ = msst_y(signal, hlength, iteration)
        figure.add_trace(
            go.Scatter(
                x=time,
                y=ridge_frequency(transform, sample_rate),
                name=f"MSST – {iteration} itération(s)",
            ),
            row=2,
            col=1,
        )

    figure.update_xaxes(title_text="Temps (s)", row=2, col=1)
    figure.update_yaxes(title_text="Amplitude", row=1, col=1)
    figure.update_yaxes(title_text="Fréquence (Hz)", row=2, col=1)
    figure.update_layout(
        title="Influence du nombre d’itérations MSST",
        template="plotly_white",
        hovermode="x unified",
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace la fréquence instantanée d'un signal CSV avec Plotly."
    )
    parser.add_argument("csv", type=Path, help="fichier CSV contenant le signal")
    parser.add_argument(
        "--sample-rate",
        type=float,
        required=True,
        help="fréquence d'échantillonnage en Hz",
    )
    parser.add_argument(
        "--column", type=int, default=0, help="indice de colonne, à partir de zéro"
    )
    parser.add_argument(
        "--skip-rows", type=int, default=0, help="nombre de lignes d'en-tête"
    )
    parser.add_argument("--hlength", type=int, default=31, help="longueur de fenêtre")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--output", type=Path, default=Path("frequence_instantanee.html")
    )
    parser.add_argument(
        "--show", action="store_true", help="ouvre aussi la figure dans le navigateur"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(iteration < 1 for iteration in args.iterations):
        raise ValueError("iterations must contain only positive integers")
    signal = load_csv(args.csv, args.column, args.skip_rows)
    figure = build_figure(signal, args.sample_rate, args.hlength, args.iterations)
    figure.write_html(args.output, include_plotlyjs=True)
    print(f"Figure écrite dans {args.output}")
    if args.show:
        figure.show()


if __name__ == "__main__":
    main()
