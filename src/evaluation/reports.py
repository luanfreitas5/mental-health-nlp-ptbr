"""Geração dos relatórios de avaliação em JSON, CSV e Markdown.

Três formatos, três públicos: JSON para consumo programático e para o MLflow,
CSV para as tabelas da dissertação e Markdown para leitura direta no
repositório.

O relatório sempre traz o intervalo de confiança ao lado do ponto e um aviso
explícito quando duas métricas próximas **não** são estatisticamente
distinguíveis — evitando que a tabela ordenada por desempenho seja lida como
um ranking de superioridade comprovada.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from config.logging import get_logger
from config.paths import ProjectPaths
from config.settings import Config
from config.version import describe_version
from constants.labels import CLASS_DISPLAY_NAMES, CLASS_ORDER
from constants.metrics import METRIC_DISPLAY_NAMES
from evaluation.evaluator import EvaluationResult
from evaluation.metrics import format_metric_with_ci
from utils.files import write_json, write_text

logger = get_logger(__name__)


def _format_number(value: Any, precision: int = 4) -> str:
    """Formata um número no padrão pt-BR, tolerando ``None``."""
    if value is None:
        return "—"
    if isinstance(value, (int | float)):
        return f"{value:.{precision}f}".replace(".", ",")
    return str(value)


def build_comparison_table(comparison: pl.DataFrame) -> str:
    """Converte a tabela comparativa em Markdown.

    Parameters
    ----------
    comparison : pl.DataFrame
        Saída de :meth:`evaluation.evaluator.Evaluator.compare`.

    Returns
    -------
    str
        Tabela em Markdown, com nomes de métrica em pt-BR.

    Examples
    --------
    >>> build_comparison_table(pl.DataFrame({"modelo": ["a"], "f1_macro": [0.5]}))
    '| Modelo | F1 (macro) |\\n| --- | --- |\\n| a | 0,5000 |'
    """
    if comparison.is_empty():
        return "_Nenhum modelo avaliado._"

    headers = [
        "Modelo" if column == "modelo" else METRIC_DISPLAY_NAMES.get(column, column)
        for column in comparison.columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in comparison.iter_rows(named=True):
        cells = [
            str(value) if column == "modelo" else _format_number(value)
            for column, value in row.items()
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_per_class_table(result: EvaluationResult) -> str:
    """Converte as métricas por classe em Markdown.

    Parameters
    ----------
    result : EvaluationResult
        Resultado de um modelo.

    Returns
    -------
    str
        Tabela em Markdown.

    Examples
    --------
    >>> build_per_class_table(resultado)  # doctest: +SKIP
    """
    lines = [
        "| Classe | Precisão | Revocação | F1 | PR-AUC | Suporte |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for class_name in CLASS_ORDER:
        metrics = result.per_class.get(class_name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    CLASS_DISPLAY_NAMES.get(class_name, class_name),
                    _format_number(metrics.get("precision")),
                    _format_number(metrics.get("recall")),
                    _format_number(metrics.get("f1")),
                    _format_number(metrics.get("pr_auc")),
                    _format_number(metrics.get("support"), precision=0),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_confusion_table(result: EvaluationResult) -> str:
    """Converte a matriz de confusão em Markdown.

    Parameters
    ----------
    result : EvaluationResult
        Resultado de um modelo.

    Returns
    -------
    str
        Matriz em Markdown, com linhas = verdade e colunas = predição.

    Examples
    --------
    >>> build_confusion_table(resultado)  # doctest: +SKIP
    """
    if not result.confusion_matrix:
        return "_Matriz de confusão indisponível._"

    names = [CLASS_DISPLAY_NAMES.get(name, name) for name in CLASS_ORDER]
    lines = [
        "| Verdade \\ Predição | " + " | ".join(names) + " |",
        "| " + " | ".join("---" for _ in range(len(names) + 1)) + " |",
    ]
    for index, row in enumerate(result.confusion_matrix):
        cells = [_format_number(value, precision=0) for value in row]
        lines.append(f"| {names[index]} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_slices_section(result: EvaluationResult) -> str:
    """Descreve o desempenho por fatia em Markdown.

    Parameters
    ----------
    result : EvaluationResult
        Resultado de um modelo.

    Returns
    -------
    str
        Seção em Markdown, com alerta quando a disparidade excede o limite.

    Examples
    --------
    >>> build_slices_section(resultado)  # doctest: +SKIP
    """
    if not result.slices:
        return "_Avaliação por fatias não executada._"

    blocks: list[str] = []
    for name, data in result.slices.items():
        blocks.append(f"#### Fatia: `{name}`\n")
        rows = data.get("slices", {})
        if not rows:
            blocks.append("_Nenhuma fatia com amostras suficientes._\n")
            continue

        blocks.extend(
            ("| Faixa | n | F1 (macro) | Revocação (macro) |", "| --- | --- | --- | --- |")
        )
        for label, metrics in rows.items():
            blocks.append(
                f"| {label} | {int(metrics['n'])} | "
                f"{_format_number(metrics.get('f1_macro'))} | "
                f"{_format_number(metrics.get('recall_macro'))} |"
            )

        if data.get("exceeds_threshold"):
            blocks.append(
                f"\n> **Atenção:** disparidade de {_format_number(data.get('gap'))} entre a "
                f"melhor e a pior faixa, acima do limite aceitável. Documentar no model card."
            )
        blocks.append("")

    return "\n".join(blocks)


def _build_header_section(
    config: Config,
    comparison: pl.DataFrame,
    primary: str,
    version: dict[str, str],
) -> list[str]:
    """Monta o cabeçalho, o aviso de incerteza e a tabela comparativa (seção 1)."""
    return [
        "# Relatório de Avaliação",
        "",
        f"**Projeto:** {config.general.project.name}  ",
        f"**Versão:** {version['version']} (`{version['git_sha'][:8]}`)  ",
        f"**Gerado em:** {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M')} UTC  ",
        f"**Métrica principal:** {METRIC_DISPLAY_NAMES.get(primary, primary)}",
        "",
        "> Todas as métricas são reportadas com intervalo de confiança de "
        f"{config.evaluation.uncertainty.confidence_level:.0%} (bootstrap, "
        f"{config.evaluation.uncertainty.n_bootstrap} reamostragens). Diferenças entre "
        "modelos só devem ser consideradas reais quando confirmadas pelos testes "
        "estatísticos da seção correspondente.",
        "",
        "## 1. Comparação entre modelos",
        "",
        build_comparison_table(comparison),
        "",
    ]


def _build_best_model_section(results: dict[str, EvaluationResult], primary: str) -> list[str]:
    """Destaca o melhor modelo pela métrica principal, se houver resultados."""
    if not results:
        return []

    best_name = max(results, key=lambda name: results[name].metrics.get(primary, float("-inf")))
    best = results[best_name]
    return [
        f"**Melhor modelo:** `{best_name}` — "
        f"{METRIC_DISPLAY_NAMES.get(primary, primary)} = "
        f"{format_metric_with_ci(best.confidence_interval)}",
        "",
    ]


def _build_single_model_section(name: str, result: EvaluationResult) -> list[str]:
    """Monta a subseção detalhada (classes, matriz, calibração, fatias) de um modelo."""
    lines = [
        f"### {name}",
        "",
        "#### Métricas por classe",
        "",
        build_per_class_table(result),
        "",
        "#### Matriz de confusão",
        "",
        build_confusion_table(result),
        "",
    ]
    if result.calibration:
        lines.extend(
            [
                "#### Calibração",
                "",
                result.calibration.get("interpretation", ""),
                "",
            ]
        )
    if result.slices:
        lines.extend(["#### Desempenho por fatia", "", build_slices_section(result), ""])
    return lines


def _build_model_details_section(results: dict[str, EvaluationResult]) -> list[str]:
    """Monta a seção 2 (desempenho detalhado por modelo)."""
    lines = ["## 2. Desempenho detalhado por modelo", ""]
    for name, result in results.items():
        lines.extend(_build_single_model_section(name, result))
    return lines


def _build_friedman_subsection(statistics: dict[str, Any]) -> list[str]:
    """Monta a subseção do teste de Friedman, se ele tiver sido executado."""
    if "friedman" not in statistics:
        return []

    return [
        "### Teste de Friedman",
        "",
        statistics["friedman"].get("interpretation", ""),
        "",
    ]


def _build_pairwise_subsection(statistics: dict[str, Any]) -> list[str]:
    """Monta a subseção de comparações par a par (Wilcoxon + Holm), se disponível."""
    if "pairwise" not in statistics:
        return []

    lines = [
        "### Comparações par a par (Wilcoxon com correção de Holm)",
        "",
        "| Comparação | p-valor | p corrigido | Significativo | Efeito |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pair, entry in statistics["pairwise"].items():
        lines.append(
            f"| {pair.replace('_vs_', ' vs. ')} | "
            f"{_format_number(entry['p_value'])} | "
            f"{_format_number(entry['p_value_corrected'])} | "
            f"{'sim' if entry['significant'] else 'não'} | "
            f"{_format_number(entry['effect_size'], precision=3)} |"
        )
    lines.append("")
    return lines


def _build_statistics_section(statistics: dict[str, Any] | None) -> list[str]:
    """Monta a seção 3 (testes estatísticos), se houver resultados a reportar."""
    if not statistics:
        return []

    lines = ["## 3. Testes estatísticos", ""]
    lines.extend(_build_friedman_subsection(statistics))
    lines.extend(_build_pairwise_subsection(statistics))
    return lines


def _build_ablation_section(ablation: pl.DataFrame | None) -> list[str]:
    """Monta a seção 4 (Ablation Study), se houver resumo disponível."""
    if ablation is None or ablation.is_empty():
        return []

    lines = [
        "## 4. Ablation Study",
        "",
        "Contribuição marginal de cada grupo de atributos (queda na métrica "
        "principal ao remover o grupo do modelo híbrido).",
        "",
        "| Grupo | Contribuição marginal | Sem o grupo | Apenas o grupo | Atributos |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['grupo']} | {_format_number(row['contribuicao_marginal'])} | "
        f"{_format_number(row['score_sem_grupo'])} | "
        f"{_format_number(row['score_apenas_grupo'])} | "
        f"{row['n_atributos']} |"
        for row in ablation.iter_rows(named=True)
    )
    lines.append("")
    return lines


def _build_limitations_section() -> list[str]:
    """Monta a seção fixa de limitações do sistema."""
    return [
        "## Limitações",
        "",
        "- O rótulo `controle` significa *sem sinais detectados no recorte coletado*, "
        "e não ausência clínica confirmada (viés de seleção documentado no datasheet).",
        "- Os rótulos vêm de supervisão fraca; a concordância com a revisão manual "
        "(kappa de Cohen) delimita o teto de desempenho alcançável.",
        "- Não há atributos demográficos coletados (minimização de dados / LGPD), "
        "então a auditoria de disparidade é comportamental, não demográfica.",
        "- Este sistema é uma ferramenta de **triagem para pesquisa**. Não constitui "
        "diagnóstico clínico nem substitui avaliação profissional.",
        "",
    ]


def build_markdown_report(
    results: dict[str, EvaluationResult],
    comparison: pl.DataFrame,
    config: Config,
    statistics: dict[str, Any] | None = None,
    ablation: pl.DataFrame | None = None,
) -> str:
    """Monta o relatório completo de avaliação em Markdown.

    Parameters
    ----------
    results : dict of str to EvaluationResult
        Resultados por modelo.
    comparison : pl.DataFrame
        Tabela comparativa.
    config : Config
        Configuração completa do projeto.
    statistics : dict, optional
        Resultado dos testes de significância.
    ablation : pl.DataFrame, optional
        Resumo do Ablation Study.

    Returns
    -------
    str
        Relatório em Markdown.

    Examples
    --------
    >>> build_markdown_report(resultados, comparacao, config)  # doctest: +SKIP
    """
    version = describe_version()
    primary = config.evaluation.metrics.primary

    sections: list[str] = []
    sections.extend(_build_header_section(config, comparison, primary, version))
    sections.extend(_build_best_model_section(results, primary))
    sections.extend(_build_model_details_section(results))
    sections.extend(_build_statistics_section(statistics))
    sections.extend(_build_ablation_section(ablation))
    sections.extend(_build_limitations_section())

    return "\n".join(sections)


def _write_json_report(
    results: dict[str, EvaluationResult],
    statistics: dict[str, Any] | None,
    config: Config,
    paths: ProjectPaths,
) -> Path:
    """Grava o relatório JSON com métricas por modelo e testes estatísticos."""
    payload = {
        **describe_version(),
        "primary_metric": config.evaluation.metrics.primary,
        "models": {name: result.to_dict() for name, result in results.items()},
        "statistics": statistics or {},
    }
    return write_json(paths.reports.metrics / "evaluation.json", payload)


def _write_csv_reports(
    comparison: pl.DataFrame,
    ablation: pl.DataFrame | None,
    paths: ProjectPaths,
) -> dict[str, Path]:
    """Grava a tabela comparativa e, se disponível, o resumo do Ablation Study."""
    written: dict[str, Path] = {}
    if comparison.is_empty():
        return written

    target = paths.reports.tables / "model_comparison.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(target)
    written["comparison_csv"] = target

    if ablation is not None and not ablation.is_empty():
        ablation_path = paths.reports.ablation / "ablation_summary.csv"
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation.write_csv(ablation_path)
        written["ablation_csv"] = ablation_path

    return written


def _collect_prediction_records(results: dict[str, EvaluationResult]) -> list[dict[str, Any]]:
    """Achata as predições de todos os modelos em registros linha a linha."""
    return [
        {
            "modelo": name,
            "user_id": user_id,
            "y_true": result.predictions["y_true"][index],
            "y_pred": result.predictions["y_pred"][index],
        }
        for name, result in results.items()
        if result.predictions
        for index, user_id in enumerate(result.predictions["user_ids"])
    ]


def _write_predictions_report(
    results: dict[str, EvaluationResult],
    paths: ProjectPaths,
) -> dict[str, Path]:
    """Grava as predições individuais em CSV, se houver registros disponíveis."""
    records = _collect_prediction_records(results)
    if not records:
        return {}

    target = paths.reports.metrics / "predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_csv(target)
    return {"predictions_csv": target}


def save_reports(
    results: dict[str, EvaluationResult],
    comparison: pl.DataFrame,
    config: Config,
    paths: ProjectPaths,
    statistics: dict[str, Any] | None = None,
    ablation: pl.DataFrame | None = None,
) -> dict[str, Path]:
    """Grava todos os relatórios de avaliação nos formatos configurados.

    Parameters
    ----------
    results : dict of str to EvaluationResult
        Resultados por modelo.
    comparison : pl.DataFrame
        Tabela comparativa.
    config : Config
        Configuração completa do projeto.
    paths : ProjectPaths
        Caminhos do projeto.
    statistics : dict, optional
        Resultado dos testes estatísticos.
    ablation : pl.DataFrame, optional
        Resumo do Ablation Study.

    Returns
    -------
    dict of str to Path
        Nome lógico -> caminho do arquivo gravado.

    Examples
    --------
    >>> save_reports(resultados, comparacao, config, paths)  # doctest: +SKIP
    """
    formats = set(config.evaluation.reporting.formats)
    written: dict[str, Path] = {}

    if "json" in formats:
        written["metrics_json"] = _write_json_report(results, statistics, config, paths)

    if "csv" in formats:
        written.update(_write_csv_reports(comparison, ablation, paths))

    if "md" in formats:
        content = build_markdown_report(results, comparison, config, statistics, ablation)
        written["report_md"] = write_text(paths.reports.root / "evaluation_report.md", content)

    if config.evaluation.reporting.save_predictions:
        written.update(_write_predictions_report(results, paths))

    logger.info("Relatórios gravados: %s.", ", ".join(sorted(written)))
    return written
