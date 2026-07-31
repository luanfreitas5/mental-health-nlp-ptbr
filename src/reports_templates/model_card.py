"""Geração do Model Card do modelo selecionado.

O Model Card documenta uso pretendido, uso fora de escopo, desempenho **por
subgrupo**, limitações e considerações éticas. Num sistema que produz sinais
sobre a saúde mental de pessoas, publicar métricas sem esse contexto é o que
permite que o modelo seja usado de formas que os dados não sustentam.

Segue a estrutura proposta por Mitchell et al. (2019), adaptada ao domínio.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config.logging import get_logger
from config.paths import ProjectPaths
from config.settings import Config
from config.version import describe_version
from constants.labels import CLASS_DISPLAY_NAMES, CLASS_ORDER
from constants.metrics import METRIC_DISPLAY_NAMES
from utils.files import read_json

logger = get_logger(__name__)


def _format(value: Any, precision: int = 4) -> str:
    """Formata um número no padrão pt-BR."""
    if value is None:
        return "—"
    if isinstance(value, (int | float)):
        return f"{value:.{precision}f}".replace(".", ",")
    return str(value)


def build_model_card(
    evaluation: dict[str, Any],
    config: Config,
    paths: ProjectPaths,
) -> str:
    """Monta o Model Card em Markdown.

    Parameters
    ----------
    evaluation : dict
        Conteúdo de ``reports/metrics/evaluation.json``.
    config : Config
        Configuração completa do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    str
        Model Card em Markdown.

    Examples
    --------
    >>> build_model_card(metricas, config, paths)[:14]  # doctest: +SKIP
    '# Model Card'
    """
    version = describe_version()
    primary = config.evaluation.metrics.primary
    models = evaluation.get("models", {})

    best_name = (
        max(models, key=lambda name: models[name]["metrics"].get(primary, float("-inf")))
        if models
        else "—"
    )
    best = models.get(best_name, {})
    metrics = best.get("metrics", {})
    per_class = best.get("per_class", {})
    calibration = best.get("calibration", {})
    interval = best.get("confidence_interval", {})

    labeling_quality = {}
    quality_path = paths.reports.metrics / "labeling_quality.json"
    if quality_path.is_file():
        labeling_quality = read_json(quality_path)

    lines: list[str] = [
        f"# Model Card — {config.general.project.name}",
        "",
        f"**Modelo:** `{best_name}`  ",
        f"**Versão do projeto:** {version['version']} (`{version['git_sha'][:8]}`)  ",
        f"**Gerado em:** {datetime.now(timezone.utc).strftime('%d/%m/%Y')}  ",
        "**Licença:** MIT (código) — os dados **não** são redistribuídos",
        "",
        "---",
        "",
        "## 1. Detalhes do modelo",
        "",
        f"- **Tarefa:** classificação multiclasse no nível do **usuário** "
        f"({', '.join(CLASS_DISPLAY_NAMES[name] for name in CLASS_ORDER)}).",
        "- **Unidade de decisão:** o usuário, a partir do histórico de publicações — "
        "não o tweet individual.",
        f"- **Arquitetura:** ver `configs/model_params.yaml`, entrada `{best_name}`.",
        f"- **Métrica principal:** {METRIC_DISPLAY_NAMES.get(primary, primary)}, escolhida "
        "porque o custo de um falso negativo (não detectar risco) supera o de um falso "
        "positivo (encaminhar para triagem humana), e as classes são desbalanceadas.",
        "- **Idioma:** português brasileiro.",
        "",
        "## 2. Uso pretendido",
        "",
        "**Uso pretendido:** pesquisa acadêmica em saúde mental computacional; estudo de "
        "viabilidade de detecção longitudinal de sinais de risco em texto em português.",
        "",
        "**Usuários pretendidos:** pesquisadores de PLN e de saúde digital, com aprovação "
        "ética para trabalhar com dados sensíveis.",
        "",
        "### Usos fora de escopo",
        "",
        "As aplicações abaixo **não** são suportadas por este modelo:",
        "",
        "- **Diagnóstico clínico.** O modelo não diagnostica depressão nem qualquer "
        "transtorno; produz um sinal estatístico sobre padrões de linguagem.",
        "- **Decisão automatizada sobre pessoas.** Qualquer uso operacional exige revisão "
        "humana; o modelo não deve acionar intervenção sozinho.",
        "- **Vigilância ou triagem sem consentimento** de indivíduos identificáveis.",
        "- **Decisões de emprego, seguro, crédito** ou qualquer contexto de alocação de "
        "recursos com impacto na vida das pessoas.",
        "- **Generalização para outros idiomas, plataformas ou populações** não observadas "
        "nos dados de treino.",
        "",
        "## 3. Dados de treinamento",
        "",
        "Ver o [Datasheet do dataset](../datasheets/datasheet.md) para a descrição completa. "
        "Em resumo:",
        "",
        "- Tweets públicos em português, coletados via `twscrape` por palavras-chave e "
        "hashtags, com histórico retrospectivo por usuário.",
        f"- Janela de observação: {config.collection.user_history.window_days} dias; "
        f"mínimo de {config.collection.user_history.min_tweets_per_user} tweets por usuário.",
        "- Rótulos por **supervisão fraca** (grupo de coleta + evidência léxica + "
        "persistência temporal), com amostra revisada manualmente.",
        "",
    ]

    if labeling_quality:
        agreement = labeling_quality.get("concordancia_revisao_manual", {})
        lines.extend(
            [
                "### Qualidade dos rótulos",
                "",
                f"- Usuários rotulados: {labeling_quality.get('n_usuarios_rotulados', '—')}",
                f"- Concordância média entre fontes: "
                f"{_format(labeling_quality.get('concordancia_media_fontes'))}",
                f"- Revisão manual: {int(agreement.get('n_revisados', 0))} usuários, "
                f"kappa de Cohen = {_format(agreement.get('kappa_cohen'))}",
                "",
                "> O kappa delimita o teto de desempenho alcançável: nenhum modelo pode "
                "superar consistentemente a qualidade do rótulo com que foi treinado.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Desempenho",
            "",
            "### Métricas agregadas (conjunto de teste)",
            "",
            "| Métrica | Valor |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| {METRIC_DISPLAY_NAMES.get(metric, metric)} | {_format(metrics[metric])} |"
        for metric in config.evaluation.metrics.compute
        if metric in metrics
    )

    if interval:
        lines.extend(
            [
                "",
                f"**{METRIC_DISPLAY_NAMES.get(primary, primary)} com IC "
                f"{config.evaluation.uncertainty.confidence_level:.0%}:** "
                f"{_format(interval.get('point'))} "
                f"[{_format(interval.get('lower'))}; {_format(interval.get('upper'))}]",
                "",
            ]
        )

    lines.extend(
        [
            "### Desempenho por classe",
            "",
            "| Classe | Precisão | Revocação | F1 | Suporte |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for class_name in CLASS_ORDER:
        entry = per_class.get(class_name, {})
        lines.append(
            f"| {CLASS_DISPLAY_NAMES[class_name]} | {_format(entry.get('precision'))} | "
            f"{_format(entry.get('recall'))} | {_format(entry.get('f1'))} | "
            f"{_format(entry.get('support'), precision=0)} |"
        )

    lines.extend(
        [
            "",
            "> A revocação da classe **Ideação Suicida** é a métrica de maior consequência: "
            "um falso negativo significa deixar de sinalizar alguém potencialmente em risco.",
            "",
        ]
    )

    if calibration:
        lines.extend(
            [
                "### Calibração",
                "",
                f"- Brier score: {_format(calibration.get('brier_score'))}",
                f"- Erro de calibração esperado (ECE): "
                f"{_format(calibration.get('expected_calibration_error'))}",
                "",
                calibration.get("interpretation", ""),
                "",
            ]
        )

    slices = best.get("slices", {})
    if slices:
        lines.extend(["### Desempenho por subgrupo", ""])
        for name, data in slices.items():
            gap = data.get("gap", 0.0)
            marker = " ⚠️" if data.get("exceeds_threshold") else ""
            lines.append(f"- **{name}:** disparidade de {_format(gap)}{marker}")
        lines.extend(
            [
                "",
                "> **Sobre fairness demográfica:** o projeto não coleta sexo, idade, raça ou "
                "região, por minimização de dados (LGPD). Uma auditoria demográfica exigiria "
                "coletar exatamente a informação sensível que se optou por não coletar. As "
                "fatias acima são **comportamentais** — é a auditoria que os dados "
                "disponíveis permitem fazer honestamente, e essa é uma limitação declarada.",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. Limitações",
            "",
            "- **Viés de seleção na classe controle.** O rótulo `controle` significa *sem "
            "sinais detectados no recorte coletado*, não ausência clínica confirmada. "
            "Postar sobre temas neutros não garante ausência de sofrimento psíquico.",
            "- **Rótulos ruidosos.** A supervisão fraca introduz erro; parte dos erros do "
            "modelo é, na verdade, erro do rótulo.",
            "- **Autosseleção da plataforma.** Usuários do X/Twitter que escrevem "
            "publicamente sobre sofrimento não representam a população geral.",
            "- **Ironia e sarcasmo.** Frequentes na plataforma e sistematicamente difíceis "
            "para modelos de linguagem; letras de música e citações produzem falsos positivos.",
            "- **Deriva temporal.** Vocabulário de rede social muda rápido; o desempenho "
            "tende a degradar em dados posteriores à janela de coleta.",
            "- **Sem validação clínica.** Nenhum rótulo foi confirmado por profissional de "
            "saúde mental nem por instrumento psicométrico validado.",
            "",
            "## 6. Considerações éticas",
            "",
            "- Todos os identificadores diretos são pseudonimizados por SHA-256 com salt na "
            "ingestão; nenhum handle é persistido.",
            "- Menções, URLs, e-mails e telefones são removidos do texto e filtrados também "
            "nos logs.",
            "- O processamento por LLM é **local** (Ollama): nenhum texto sensível é enviado "
            "a serviços de terceiros.",
            "- A coleta é bloqueada por barreira técnica enquanto não houver aprovação "
            "CEP/CONEP registrada.",
            "- **Risco de uso indevido:** um modelo que sinaliza risco de suicídio pode ser "
            "usado para vigilância ou estigmatização. Por isso o escopo de uso é restrito à "
            "pesquisa, e os dados não são redistribuídos.",
            "",
            "## 7. Recomendações",
            "",
            "- Nunca use a saída do modelo isoladamente para tomar decisão sobre uma pessoa.",
            "- Em qualquer aplicação, mantenha revisão humana e um canal de encaminhamento "
            "para atendimento profissional (no Brasil, CVV — 188, 24 h, gratuito).",
            "- Reavalie o desempenho antes de aplicar o modelo a dados de período, "
            "plataforma ou população diferentes.",
            "- Monitore deriva de dados e reavalie periodicamente com rótulos novos.",
            "",
            "---",
            "",
            f"*Gerado automaticamente por `{__name__}` em "
            f"{datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M')} UTC.*",
            "",
        ]
    )

    return "\n".join(lines)
