"""Geração do Datasheet for Datasets.

Segue a estrutura proposta por Gebru et al. (2021): motivação, composição,
processo de coleta, pré-processamento, usos, distribuição e manutenção.

Num dataset de saúde mental construído a partir de publicações públicas de
pessoas reais, o datasheet é o documento que registra o que foi coletado, sob
qual base legal, com quais limitações — e por que o dataset **não** é
redistribuído.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config.logging import get_logger
from config.paths import ProjectPaths
from config.settings import Config
from config.version import describe_version
from utils.files import read_json

logger = get_logger(__name__)


def _load_features_summary(paths: ProjectPaths) -> dict[str, Any]:
    """Carrega o resumo de atributos do dataset, se existir."""
    features_path = paths.reports.metrics / "features_summary.json"
    if features_path.is_file():
        return read_json(features_path)
    return {}


def _load_labeling_quality(paths: ProjectPaths) -> dict[str, Any]:
    """Carrega o relatório de qualidade dos rótulos, se existir."""
    labeling_path = paths.reports.metrics / "labeling_quality.json"
    if labeling_path.is_file():
        return read_json(labeling_path)
    return {}


def _resolve_total_users(summary: dict[str, Any], distribution: dict[str, Any]) -> Any:
    """Resolve o total de usuários a partir do resumo ou, na ausência, da distribuição."""
    return summary.get("n_usuarios", sum(distribution.values()) if distribution else "—")


def _describe_pseudonymization_status(privacy: Any) -> str:
    """Descreve se a pseudonimização de identificadores diretos está ativa."""
    return "ativa" if privacy.pseudonymize_user_ids else "inativa"


def _describe_distribution(distribution: dict[str, Any]) -> Any:
    """Retorna a distribuição de classes para exibição, ou um marcador de ausência."""
    return distribution or "—"


def build_datasheet(config: Config, paths: ProjectPaths) -> str:
    """Monta o Datasheet do dataset em Markdown.

    Parameters
    ----------
    config : Config
        Configuração completa do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    str
        Datasheet em Markdown.

    Examples
    --------
    >>> build_datasheet(config, paths)[:12]  # doctest: +SKIP
    '# Datasheet'
    """
    version = describe_version()
    collection = config.collection
    privacy = config.general.privacy

    summary = _load_features_summary(paths)

    distribution = _load_labeling_quality(paths).get("distribuicao_classes", {})
    total_users = _resolve_total_users(summary, distribution)

    lines: list[str] = [
        f"# Datasheet — Base Longitudinal de {config.general.project.name}",
        "",
        f"**Versão do projeto:** {version['version']} (`{version['git_sha'][:8]}`)  ",
        f"**Gerado em:** {datetime.now(timezone.utc).strftime('%d/%m/%Y')}",
        "",
        "---",
        "",
        "## 1. Motivação",
        "",
        "**Para que o dataset foi criado?**  ",
        "Para viabilizar o estudo de detecção **longitudinal** de sinais de depressão e "
        "ideação suicida em português. A literatura concentra-se na classificação de "
        "publicações isoladas, mas transtornos mentais são condições persistentes: uma "
        "única publicação pode refletir ironia, sarcasmo ou um evento pontual. O dataset "
        "foi construído para permitir a modelagem centrada no **usuário**, com histórico e "
        "evolução temporal.",
        "",
        "**Quem criou e com qual financiamento?**  ",
        "Pesquisa de mestrado em Ciência de Dados. Sem financiamento comercial.",
        "",
        "## 2. Composição",
        "",
        "**O que representa cada instância?**  ",
        "Um **usuário** do X/Twitter, descrito pelo histórico de publicações, por atributos "
        "agregados (linguísticos, emocionais, semânticos, temporais, comportamentais e "
        "psicológicos) e por um rótulo de classe.",
        "",
        f"- **Número de usuários:** {total_users}",
        f"- **Número de atributos:** {summary.get('n_atributos', '—')}",
        f"- **Distribuição das classes:** {_describe_distribution(distribution)}",
        "",
        "**Há informação faltante?**  ",
        "Sim, por construção: features de tendência temporal exigem histórico mínimo e ficam "
        "ausentes para usuários com janela curta. A ausência é sinalizada por colunas "
        f"indicadoras (`*_is_missing`) antes da imputação por "
        f"`{config.features.aggregation.missing_strategy}` — a ausência aqui não é "
        "aleatória e carrega informação.",
        "",
        "**O dataset contém dados confidenciais ou sensíveis?**  ",
        "Sim. Trata-se de dado sensível de saúde na acepção da LGPD (art. 5º, II). Todo o "
        "tratamento segue as salvaguardas da Seção 5.",
        "",
        "## 3. Processo de coleta",
        "",
        "**Como os dados foram obtidos?**  ",
        "Coleta de publicações **públicas** do X/Twitter via `twscrape`, em duas fases:",
        "",
        "1. **Busca semente** por palavras-chave e hashtags relacionadas a depressão, "
        "ideação suicida e (para a classe controle) temas neutros. Os termos estão "
        "versionados em `configs/queries/*.txt`.",
        "2. **Coleta retrospectiva** do histórico de cada autor identificado, formando o "
        "perfil temporal.",
        "",
        f"- **Janela de busca:** {collection.seed_search.since} a {collection.seed_search.until}",
        f"- **Profundidade retrospectiva:** {collection.user_history.window_days} dias",
        f"- **Tweets por usuário:** mínimo {collection.user_history.min_tweets_per_user}, "
        f"máximo {collection.user_history.max_tweets_per_user}",
        f"- **Idioma:** {collection.seed_search.language}",
        "",
        "**Amostragem.**  ",
        "Não probabilística, por conveniência e por palavras-chave. Isso implica viés de "
        "seleção conhecido e não corrigível (ver Seção 6).",
        "",
        "**Houve revisão ética?**  ",
        "A coleta só é executada mediante aprovação CEP/CONEP, verificada por barreira "
        "técnica no pipeline (`ETHICS_APPROVAL_ID` no `.env`).",
        "",
        "## 4. Pré-processamento e rotulação",
        "",
        "**Limpeza aplicada:**",
        "",
        "- Remoção de duplicatas por identificador e por texto dentro do mesmo usuário.",
        "- Normalização Unicode, colapso de repetições expressivas, desempacotamento de hashtags.",
        "- Substituição de PII por placeholders (menções, URLs, e-mails, telefones).",
        "- Filtros de comprimento, idioma e de contas com padrão automatizado.",
        "- Duas versões do texto são mantidas: `text_normalized` (para Transformers/LLM) e "
        "`text_clean` (para TF-IDF e léxicos).",
        "",
        "**Rotulação:**",
        "",
        "- **Por tweet:** sentimento (positivo/negativo/neutro) por encoder Transformer "
        f"(`{config.labeling.sentiment.model_name}`), usado como *feature*, **não** como "
        "proxy de risco clínico.",
        "- **Por usuário:** supervisão fraca por voto ponderado entre grupo de coleta, "
        "evidência léxica e persistência temporal, com descarte dos casos sem consenso "
        f"mínimo de {config.labeling.user_labeling.consensus.min_agreement:.0%}.",
        f"- **Revisão manual:** amostra estratificada de "
        f"{config.labeling.user_labeling.consensus.manual_review_sample_size} usuários, com "
        "concordância medida por kappa de Cohen.",
        "",
        "**Os dados brutos foram preservados?**  ",
        "Sim: `data/raw/` nunca é modificado. Todas as transformações escrevem em "
        "`data/interim/` e `data/processed/`.",
        "",
        "## 5. Privacidade e base legal (LGPD)",
        "",
        f"- **Pseudonimização:** {_describe_pseudonymization_status(privacy)} — "
        "identificadores diretos são convertidos em SHA-256 com salt secreto **antes** de "
        "qualquer gravação em disco. Sem o salt, o hash não é reversível por força bruta "
        "sobre handles públicos.",
        "- **Minimização:** nome de exibição, biografia, foto e localização não são "
        "coletados. Nenhum atributo demográfico é coletado.",
        "- **PII em logs:** filtro aplicado a todos os handlers de logging.",
        "- **Processamento local:** a extração por LLM roda em Ollama na própria máquina; "
        "nenhum texto é enviado a serviços de terceiros (evita transferência internacional "
        "de dado sensível, arts. 11 e 33).",
        "- **Base legal:** tratamento para finalidade de **pesquisa acadêmica** "
        "(art. 7º, IV e art. 11, II, 'c'), com aprovação do comitê de ética.",
        "- **Direito à eliminação:** como cada usuário tem um arquivo próprio em "
        "`data/raw/user_histories/<user_id>.parquet`, a remoção de um titular é uma operação "
        "localizada, seguida da reexecução do pipeline.",
        "",
        "## 6. Limitações e vieses conhecidos",
        "",
        "- **Viés de seleção da classe controle.** Coletada por temas neutros, ela significa "
        "*sem sinais detectados no recorte coletado* — nunca ausência clínica confirmada. "
        "Uma pessoa com depressão pode postar apenas sobre futebol.",
        "- **Viés de plataforma.** Quem escreve publicamente sobre sofrimento psíquico no "
        "X/Twitter não representa a população brasileira.",
        "- **Viés de palavra-chave.** Os termos das consultas determinam quem entra na "
        "amostra; formas de expressar sofrimento fora desse vocabulário ficam invisíveis.",
        "- **Ausência de validação clínica.** Nenhum rótulo foi confirmado por profissional "
        "de saúde nem por instrumento psicométrico.",
        "- **Coocorrência colapsada.** Depressão e ideação suicida coocorrem clinicamente; a "
        "formulação multiclasse atribui o rótulo dominante, e a coocorrência é preservada "
        "apenas em `user_label_multilabel` para a análise de sensibilidade.",
        "",
        "## 7. Distribuição",
        "",
        "**O dataset será distribuído?**  ",
        "**Não.** O conteúdo textual de publicações de pessoas identificáveis na origem não "
        "é redistribuído, mesmo pseudonimizado — a reidentificação por busca do texto exato "
        "é trivial em rede social pública.",
        "",
        "O que é publicado: o **código** que reproduz a coleta e o processamento, os termos "
        "das consultas, os léxicos, as configurações e os resultados agregados. Terceiros "
        "com aprovação ética própria podem reconstruir uma base equivalente.",
        "",
        "## 8. Manutenção",
        "",
        "- **Responsável:** autor da dissertação (contato no `README.md`).",
        "- **Versionamento:** os artefatos de dados são versionados por hash em "
        "`data/processed/dataset_manifest.json`; alterações silenciosas são detectadas na "
        "comparação de manifestos.",
        "- **Retenção:** os dados brutos são mantidos apenas durante a pesquisa e "
        "eliminados ao término, conforme o protocolo aprovado.",
        "",
        "---",
        "",
        "> **Se você está passando por sofrimento psíquico:** no Brasil, o CVV atende pelo "
        "**188** (24 h, gratuito) e em cvv.org.br.",
        "",
        f"*Gerado automaticamente por `{__name__}` em "
        f"{datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M')} UTC.*",
        "",
    ]

    return "\n".join(lines)
