"""Construção e versionamento dos prompts enviados ao LLM.

Os prompts vivem em ``configs/llm.yaml`` e carregam um número de versão. Isso
não é burocracia: mudar um prompt muda o método de extração, e resultados
produzidos com prompts diferentes não são comparáveis. A versão é gravada
junto de cada score extraído, de modo que a análise possa detectar (em vez de
misturar silenciosamente) dados gerados por métodos distintos.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.logging import get_logger
from config.settings import LLMConfig
from exceptions.model import LLMError
from preprocessing.text import contains_pii

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prompt:
    """Par de mensagens pronto para envio ao LLM.

    Attributes
    ----------
    system : str
        Instruções de sistema (papel, regras, formato de saída).
    user : str
        Conteúdo a analisar.
    version : str
        Versão do template usado.
    """

    system: str
    user: str
    version: str

    @property
    def total_chars(self) -> int:
        """Comprimento total do prompt, em caracteres."""
        return len(self.system) + len(self.user)


def format_tweets(tweets: list[str], max_chars: int) -> str:
    """Formata uma lista de tweets como bloco numerado para o prompt.

    A numeração ajuda o modelo a tratar as publicações como uma sequência
    temporal, e não como um texto único — que é justamente a informação que a
    abordagem centrada no usuário quer explorar.

    Parameters
    ----------
    tweets : list of str
        Textos normalizados, em ordem cronológica.
    max_chars : int
        Orçamento máximo de caracteres; o excedente é truncado pelo fim.

    Returns
    -------
    str
        Bloco formatado.

    Examples
    --------
    >>> format_tweets(["primeiro", "segundo"], max_chars=1000)
    '1. primeiro\\n2. segundo'
    """
    lines: list[str] = []
    used = 0

    for index, tweet in enumerate(tweets, start=1):
        line = f"{index}. {tweet}"
        if used + len(line) > max_chars:
            logger.debug("Prompt truncado em %d de %d tweets.", index - 1, len(tweets))
            break
        lines.append(line)
        used += len(line) + 1

    return "\n".join(lines)


def build_psychological_prompt(tweets: list[str], config: LLMConfig) -> Prompt:
    """Monta o prompt de extração do vetor psicológico.

    Parameters
    ----------
    tweets : list of str
        Tweets do usuário (um lote), em ordem cronológica.
    config : LLMConfig
        Configuração de ``configs/llm.yaml``.

    Returns
    -------
    Prompt
        Prompt pronto para envio.

    Raises
    ------
    LLMError
        Se a lista estiver vazia, ou se a salvaguarda de PII estiver ativa e
        algum texto ainda contiver informação identificável.

    Examples
    --------
    >>> build_psychological_prompt(["hoje foi difícil"], config.llm).version  # doctest: +SKIP
    '1.0.0'
    """
    if not tweets:
        raise LLMError("Não é possível montar o prompt psicológico sem tweets.")

    if config.safeguards.require_pii_scrubbed_input:
        offenders = sum(1 for tweet in tweets if contains_pii(tweet))
        if offenders:
            raise LLMError(
                f"{offenders} texto(s) ainda contêm PII. O LLM só recebe conteúdo já "
                "higienizado pela etapa de preprocessing.",
                context={"n_tweets": len(tweets)},
            )

    budget = config.safeguards.max_prompt_chars - len(config.prompts.psychological_system)
    body = format_tweets(tweets, max_chars=max(budget, 0))

    return Prompt(
        system=config.prompts.psychological_system,
        user=config.prompts.psychological_user.format(tweets=body),
        version=config.prompts.version,
    )


def build_classifier_prompt(
    tweets: list[str],
    config: LLMConfig,
    examples: list[tuple[list[str], str]] | None = None,
) -> Prompt:
    """Monta o prompt de classificação do usuário pelo LLM.

    Parameters
    ----------
    tweets : list of str
        Histórico do usuário a classificar.
    config : LLMConfig
        Configuração de ``configs/llm.yaml``.
    examples : list of tuple, optional
        Exemplos ``(tweets, classe)`` para o modo few-shot. Devem vir
        **sempre** do split de treino: exemplos do teste seriam vazamento
        direto e inflariam a métrica reportada.

    Returns
    -------
    Prompt
        Prompt pronto para envio.

    Raises
    ------
    LLMError
        Se a lista de tweets estiver vazia.

    Examples
    --------
    >>> build_classifier_prompt(["dia difícil"], config.llm)  # doctest: +SKIP
    """
    if not tweets:
        raise LLMError("Não é possível classificar um usuário sem tweets.")

    sections: list[str] = []
    if config.classifier.mode == "few_shot" and examples:
        sections.append("Exemplos anotados:")
        for index, (example_tweets, label) in enumerate(examples, start=1):
            body = format_tweets(example_tweets[:5], max_chars=1200)
            sections.append(f"[Exemplo {index}]\n{body}\nClasse: {label}\n")

    limited = tweets[: config.classifier.max_tweets_per_prompt]
    budget = config.safeguards.max_prompt_chars - len(config.prompts.classifier_system)
    budget -= sum(len(section) for section in sections)

    sections.append(
        config.prompts.classifier_user.format(tweets=format_tweets(limited, max(budget, 0)))
    )

    return Prompt(
        system=config.prompts.classifier_system,
        user="\n".join(sections),
        version=config.prompts.version,
    )
