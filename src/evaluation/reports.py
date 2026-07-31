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
    if isinstance(value, (int, float)):
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

    sections: list[str] = [
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

    if results:
        best_name = max(results, key=lambda name: results[name].metrics.get(primary, float("-inf")))
        best = results[best_name]
        sections.extend(
            [
                f"**Melhor modelo:** `{best_name}` — "
                f"{METRIC_DISPLAY_NAMES.get(primary, primary)} = "
                f"{format_metric_with_ci(best.confidence_interval)}",
                "",
            ]
        )

    sections.extend(["## 2. Desempenho detalhado por modelo", ""])
    for name, result in results.items():
        sections.extend(
            [
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
        )
        if result.calibration:
            sections.extend(
                [
                    "#### Calibração",
                    "",
                    result.calibration.get("interpretation", ""),
                    "",
                ]
            )
        if result.slices:
            sections.extend(["#### Desempenho por fatia", "", build_slices_section(result), ""])

    if statistics:
        sections.extend(["## 3. Testes estatísticos", ""])
        if "friedman" in statistics:
            sections.extend(
                [
                    "### Teste de Friedman",
                    "",
                    statistics["friedman"].get("interpretation", ""),
                    "",
                ]
            )
        if "pairwise" in statistics:
            sections.extend(
                [
                    "### Comparações par a par (Wilcoxon com correção de Holm)",
                    "",
                    "| Comparação | p-valor | p corrigido | Significativo | Efeito |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for pair, entry in statistics["pairwise"].items():
                sections.append(
                    f"| {pair.replace('_vs_', ' vs. ')} | "
                    f"{_format_number(entry['p_value'])} | "
                    f"{_format_number(entry['p_value_corrected'])} | "
                    f"{'sim' if entry['significant'] else 'não'} | "
                    f"{_format_number(entry['effect_size'], precision=3)} |"
                )
            sections.append("")

    if ablation is not None and not ablation.is_empty():
        sections.extend(
            [
                "## 4. Ablation Study",
                "",
                "Contribuição marginal de cada grupo de atributos (queda na métrica "
                "principal ao remover o grupo do modelo híbrido).",
                "",
                "| Grupo | Contribuição marginal | Sem o grupo | Apenas o grupo | Atributos |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        sections.extend(
            f"| {row['grupo']} | {_format_number(row['contribuicao_marginal'])} | "
            f"{_format_number(row['score_sem_grupo'])} | "
            f"{_format_number(row['score_apenas_grupo'])} | "
            f"{row['n_atributos']} |"
            for row in ablation.iter_rows(named=True)
        )
        sections.append("")

    sections.extend(
        [
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
    )

    return "\n".join(sections)


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
        payload = {
            **describe_version(),
            "primary_metric": config.evaluation.metrics.primary,
            "models": {name: result.to_dict() for name, result in results.items()},
            "statistics": statistics or {},
        }
        written["metrics_json"] = write_json(paths.reports.metrics / "evaluation.json", payload)

    if "csv" in formats and not comparison.is_empty():
        target = paths.reports.tables / "model_comparison.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_csv(target)
        written["comparison_csv"] = target

        if ablation is not None and not ablation.is_empty():
            ablation_path = paths.reports.ablation / "ablation_summary.csv"
            ablation_path.parent.mkdir(parents=True, exist_ok=True)
            ablation.write_csv(ablation_path)
            written["ablation_csv"] = ablation_path

    if "md" in formats:
        content = build_markdown_report(results, comparison, config, statistics, ablation)
        written["report_md"] = write_text(paths.reports.root / "evaluation_report.md", content)

    if config.evaluation.reporting.save_predictions:
        records = [
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
        if records:
            target = paths.reports.metrics / "predictions.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(records).write_csv(target)
            written["predictions_csv"] = target

    logger.info("Relatórios gravados: %s.", ", ".join(sorted(written)))
    return written
