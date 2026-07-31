"""Dashboard Streamlit para inspecionar os resultados dos experimentos.

Lê exclusivamente os artefatos já gravados em ``reports/`` — não treina, não
avalia e não recalcula nada. A separação é deliberada: um dashboard que dispara
processamento pesado a cada interação vira uma segunda implementação do
pipeline, com risco de divergir dele.

Executar::

    uv sync --extra app --dev
    make app
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# `src/` é a raiz de importação do projeto (ver docs/guides/architecture.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import streamlit as st

from config.paths import get_paths
from constants.labels import CLASS_DISPLAY_NAMES, CLASS_ORDER
from constants.metrics import METRIC_DISPLAY_NAMES
from utils.files import read_json

PATHS = get_paths()

AVISO = """
**Este sistema não é uma ferramenta de diagnóstico.** Produz um sinal
estatístico sobre padrões de linguagem, destinado à pesquisa acadêmica.

Se você está passando por sofrimento psíquico: **CVV — 188** (24 h, gratuito).
"""


@st.cache_data
def load_evaluation() -> dict[str, Any]:
    """Carrega o JSON de métricas da última avaliação.

    Returns
    -------
    dict
        Conteúdo de ``reports/metrics/evaluation.json`` (vazio se ausente).
    """
    target = PATHS.reports.metrics / "evaluation.json"
    return read_json(target) if target.is_file() else {}


@st.cache_data
def load_table(path_str: str) -> pl.DataFrame:
    """Carrega uma tabela CSV de resultados.

    Parameters
    ----------
    path_str : str
        Caminho do arquivo (string, para permitir o cache do Streamlit).

    Returns
    -------
    pl.DataFrame
        Conteúdo do arquivo (vazio se ausente).
    """
    target = Path(path_str)
    return pl.read_csv(target) if target.is_file() else pl.DataFrame()


def render_comparison() -> None:
    """Exibe a comparação entre modelos e o Ablation Study."""
    st.header("Comparação entre modelos")

    comparison = load_table(str(PATHS.reports.tables / "model_comparison.csv"))
    if comparison.is_empty():
        st.info("Nenhuma avaliação encontrada. Execute `make evaluate`.")
        return

    renamed = comparison.rename(
        {
            column: METRIC_DISPLAY_NAMES.get(column, column)
            for column in comparison.columns
            if column != "modelo"
        }
    )
    st.dataframe(renamed.to_pandas(), width="stretch")

    st.caption(
        "A ordenação **não** é um ranking de superioridade comprovada. "
        "Consulte a aba de testes estatísticos: intervalos sobrepostos "
        "significam que a diferença não está estabelecida."
    )

    ablation = load_table(str(PATHS.reports.ablation / "ablation_summary.csv"))
    if not ablation.is_empty():
        st.subheader("Ablation Study")
        st.dataframe(ablation.to_pandas(), width="stretch")
        st.caption(
            "**Contribuição marginal** = queda ao remover o grupo. Valor baixo "
            "não significa grupo inútil: grupos correlacionados se substituem, "
            "e a coluna *apenas o grupo* revela isso."
        )


def render_model_detail(evaluation: dict[str, Any]) -> None:
    """Exibe o detalhamento de um modelo selecionado."""
    st.header("Detalhamento por modelo")

    models = evaluation.get("models", {})
    if not models:
        st.info("Nenhuma avaliação encontrada. Execute `make evaluate`.")
        return

    selected = st.selectbox("Modelo", sorted(models))
    result = models[selected]

    columns = st.columns(4)
    for column, metric in zip(
        columns, ("f1_macro", "recall_macro", "roc_auc_ovr", "mcc"), strict=False
    ):
        value = result.get("metrics", {}).get(metric)
        column.metric(
            METRIC_DISPLAY_NAMES.get(metric, metric),
            f"{value:.4f}".replace(".", ",") if value is not None else "—",
        )

    interval = result.get("confidence_interval", {})
    if interval:
        st.caption(
            f"IC 95% do F1-macro: [{interval['lower']:.4f}; {interval['upper']:.4f}]".replace(
                ".", ","
            )
        )

    st.subheader("Desempenho por classe")
    per_class = result.get("per_class", {})
    if per_class:
        records = [
            {
                "Classe": CLASS_DISPLAY_NAMES.get(name, name),
                "Precisão": per_class.get(name, {}).get("precision"),
                "Revocação": per_class.get(name, {}).get("recall"),
                "F1": per_class.get(name, {}).get("f1"),
                "Suporte": per_class.get(name, {}).get("support"),
            }
            for name in CLASS_ORDER
        ]
        st.dataframe(pl.DataFrame(records).to_pandas(), width="stretch")
        st.warning(
            "A **revocação de Ideação Suicida** é a métrica de maior "
            "consequência: um falso negativo significa deixar de sinalizar "
            "alguém potencialmente em risco."
        )

    calibration = result.get("calibration", {})
    if calibration:
        st.subheader("Calibração")
        st.write(calibration.get("interpretation", ""))

    slices = result.get("slices", {})
    if slices:
        st.subheader("Desempenho por subgrupo comportamental")
        for name, data in slices.items():
            rows = data.get("slices", {})
            if not rows:
                continue
            st.markdown(f"**{name}**")
            st.dataframe(
                pl.DataFrame([{"faixa": key, **value} for key, value in rows.items()]).to_pandas(),
                width="stretch",
            )
            if data.get("exceeds_threshold"):
                st.error(
                    f"Disparidade de {data.get('gap', 0):.4f} entre a melhor e a "
                    "pior faixa, acima do limite aceitável.".replace(".", ",")
                )


def render_statistics(evaluation: dict[str, Any]) -> None:
    """Exibe os resultados dos testes de significância."""
    st.header("Testes estatísticos")

    statistics = evaluation.get("statistics", {})
    if not statistics:
        st.info("Nenhum teste estatístico encontrado. Execute `make evaluate`.")
        return

    if "friedman" in statistics:
        st.subheader("Teste de Friedman")
        st.write(statistics["friedman"].get("interpretation", ""))

    if "pairwise" in statistics:
        st.subheader("Comparações par a par (Wilcoxon + correção de Holm)")
        records = [
            {
                "Comparação": pair.replace("_vs_", " vs. "),
                "p-valor": entry["p_value"],
                "p corrigido": entry["p_value_corrected"],
                "Significativo": "sim" if entry["significant"] else "não",
                "Tamanho de efeito": entry["effect_size"],
            }
            for pair, entry in statistics["pairwise"].items()
        ]
        st.dataframe(pl.DataFrame(records).to_pandas(), width="stretch")
        st.caption(
            "A correção de Holm existe porque comparar 8 modelos par a par são "
            "28 testes: sem correção, esperar-se-ia ao menos um "
            "'significativo' por puro acaso."
        )


def render_figures() -> None:
    """Exibe as figuras geradas pela etapa de relatórios."""
    st.header("Figuras")

    figures = sorted(PATHS.reports.figures.glob("*.png"))
    if not figures:
        st.info("Nenhuma figura encontrada. Execute `make report`.")
        return

    selected = st.multiselect(
        "Figuras",
        [path.stem for path in figures],
        default=[path.stem for path in figures[:3]],
    )
    for path in figures:
        if path.stem in selected:
            st.subheader(path.stem.replace("_", " ").title())
            st.image(str(path), width="stretch")


def render_documents() -> None:
    """Exibe o Model Card e o Datasheet."""
    st.header("Documentação de IA responsável")

    documents = {
        "Model Card": PATHS.reports.model_cards / "model_card.md",
        "Datasheet": PATHS.reports.datasheets / "datasheet.md",
    }
    for title, path in documents.items():
        with st.expander(title, expanded=False):
            if path.is_file():
                st.markdown(path.read_text(encoding="utf-8"))
            else:
                st.info(f"{title} não encontrado. Execute `make report`.")


def main() -> None:
    """Monta o dashboard."""
    st.set_page_config(
        page_title="mental-health-nlp-ptbr",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 mental-health-nlp-ptbr")
    st.caption(
        "Detecção longitudinal de sinais de depressão e ideação suicida — "
        "resultados dos experimentos."
    )
    st.warning(AVISO)

    evaluation = load_evaluation()

    tabs = st.tabs(["Comparação", "Por modelo", "Estatística", "Figuras", "IA responsável"])
    with tabs[0]:
        render_comparison()
    with tabs[1]:
        render_model_detail(evaluation)
    with tabs[2]:
        render_statistics(evaluation)
    with tabs[3]:
        render_figures()
    with tabs[4]:
        render_documents()


if __name__ == "__main__":
    main()
