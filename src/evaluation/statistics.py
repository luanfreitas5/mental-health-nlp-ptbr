"""Testes de significância para comparação entre modelos.

Afirmar que "o modelo híbrido é melhor que o XGBoost" a partir de dois
números — 0,76 contra 0,74 — não é uma conclusão científica. Este módulo
implementa os testes que a proposta exige e adiciona duas salvaguardas que
costumam faltar:

* **Correção para múltiplas comparações.** Comparando oito modelos par a par
  são 28 testes; ao nível de 5%, esperar-se-ia ao menos um "significativo"
  por puro acaso. A correção de Holm é aplicada por padrão.
* **Tamanho de efeito.** O p-valor responde "a diferença é real?", não "a
  diferença importa?". Com amostra grande, uma diferença de 0,002 pode ser
  significativa e irrelevante. O delta de Cliff acompanha cada comparação.

Escolha dos testes:

* **McNemar** — dois modelos no *mesmo* conjunto de teste. Usa apenas os
  casos em que discordam, que é exatamente onde está a informação.
* **Wilcoxon pareado** — dois modelos ao longo dos folds da validação
  cruzada; não assume normalidade, adequado a 5–10 folds.
* **Friedman + Nemenyi** — três ou mais modelos ao longo dos folds.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
from scipy import stats

from config.logging import get_logger
from config.settings import StatisticsSection

logger = get_logger(__name__)


@dataclass(frozen=True)
class TestResult:
    """Resultado de um teste de significância.

    Attributes
    ----------
    test : str
        Nome do teste aplicado.
    statistic : float
        Estatística do teste.
    p_value : float
        p-valor bruto.
    p_value_corrected : float
        p-valor após correção para múltiplas comparações.
    significant : bool
        Se a diferença é significativa ao nível ``alpha`` (após correção).
    effect_size : float
        Tamanho de efeito (delta de Cliff), quando aplicável.
    interpretation : str
        Leitura do resultado em pt-BR.
    """

    test: str
    statistic: float
    p_value: float
    p_value_corrected: float
    significant: bool
    effect_size: float
    interpretation: str


def cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    """Calcula o delta de Cliff entre duas amostras.

    Não paramétrico e interpretável: é a diferença entre a probabilidade de
    um valor de ``first`` superar um de ``second`` e a probabilidade inversa.
    Varia em ``[-1, 1]``.

    Parameters
    ----------
    first, second : np.ndarray
        Amostras a comparar.

    Returns
    -------
    float
        Delta de Cliff.

    Examples
    --------
    >>> cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0]))
    1.0
    """
    if 0 in (len(first), len(second)):
        return 0.0

    comparisons = np.sign(first[:, None] - second[None, :])
    return float(comparisons.mean())


def interpret_effect_size(delta: float) -> str:
    """Traduz o delta de Cliff nas faixas convencionais da literatura.

    Parameters
    ----------
    delta : float
        Delta de Cliff.

    Returns
    -------
    str
        ``"desprezível"``, ``"pequeno"``, ``"médio"`` ou ``"grande"``.

    Examples
    --------
    >>> interpret_effect_size(0.5)
    'grande'
    """
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "desprezível"
    if magnitude < 0.33:
        return "pequeno"
    if magnitude < 0.474:
        return "médio"
    return "grande"


def holm_correction(p_values: list[float]) -> list[float]:
    """Aplica a correção de Holm-Bonferroni a uma lista de p-valores.

    Preferida à Bonferroni simples: controla o mesmo erro familiar, mas é
    uniformemente mais potente — rejeita ao menos tantas hipóteses quanto ela.

    Parameters
    ----------
    p_values : list of float
        p-valores brutos.

    Returns
    -------
    list of float
        p-valores corrigidos, na ordem original.

    Examples
    --------
    >>> holm_correction([0.01, 0.04])
    [0.02, 0.04]
    """
    if not p_values:
        return []

    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(n, dtype=float)

    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (n - rank) * p_values[index])
        running_max = max(running_max, adjusted)
        corrected[index] = running_max

    return corrected.tolist()


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    correction: bool = True,
    alpha: float = 0.05,
) -> TestResult:
    """Compara dois modelos no mesmo conjunto de teste (teste de McNemar).

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    pred_a, pred_b : np.ndarray
        Predições dos dois modelos.
    correction : bool, optional
        Aplica a correção de continuidade de Yates, by default True.
    alpha : float, optional
        Nível de significância, by default 0.05.

    Returns
    -------
    TestResult
        Resultado do teste.

    Examples
    --------
    >>> resultado = mcnemar_test(
    ...     np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 1])
    ... )
    >>> resultado.test
    'mcnemar'
    """
    correct_a = y_true == pred_a
    correct_b = y_true == pred_b

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b

    if discordant == 0:
        return TestResult(
            test="mcnemar",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            significant=False,
            effect_size=0.0,
            interpretation="Os modelos acertam e erram exatamente nos mesmos casos.",
        )

    if discordant < 25:
        # Amostra pequena: o teste binomial exato é preferível à aproximação
        # qui-quadrado, que só vale assintoticamente.
        p_value = float(stats.binomtest(only_a, discordant, 0.5).pvalue)
        statistic = float(min(only_a, only_b))
        method = "binomial exato"
    else:
        numerator = abs(only_a - only_b) - (1 if correction else 0)
        statistic = numerator**2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "qui-quadrado"

    effect = (only_a - only_b) / discordant
    significant = p_value < alpha
    better = "o primeiro" if only_a > only_b else "o segundo"

    interpretation = (
        f"Discordâncias: {only_a} a favor do primeiro, {only_b} do segundo ({method}). "
        + (
            f"Diferença significativa (p={p_value:.4f}); {better} modelo acerta mais."
            if significant
            else f"Sem diferença significativa (p={p_value:.4f})."
        )
    )

    return TestResult(
        test="mcnemar",
        statistic=statistic,
        p_value=p_value,
        p_value_corrected=p_value,
        significant=significant,
        effect_size=effect,
        interpretation=interpretation,
    )


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """Compara dois modelos ao longo dos folds (Wilcoxon pareado).

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Métrica de cada modelo por fold, na mesma ordem.
    alternative : str, optional
        Hipótese alternativa, by default ``"two-sided"``.
    alpha : float, optional
        Nível de significância, by default 0.05.

    Returns
    -------
    TestResult
        Resultado do teste.

    Raises
    ------
    ValueError
        Se as amostras tiverem tamanhos diferentes.

    Examples
    --------
    >>> wilcoxon_test(np.array([0.7, 0.72, 0.71]), np.array([0.6, 0.62, 0.61])).test
    'wilcoxon'
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Wilcoxon exige amostras pareadas: {len(scores_a)} vs. {len(scores_b)} folds."
        )

    if np.allclose(scores_a, scores_b):
        return TestResult(
            test="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            significant=False,
            effect_size=0.0,
            interpretation="Os modelos obtiveram desempenho idêntico em todos os folds.",
        )

    # `scipy.stats.wilcoxon` devolve uma tupla sem tipagem precisa nos stubs;
    # o cast declara o tipo real de retorno em tempo de execução.
    statistic, p_value = cast(
        "tuple[float, float]", stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    )
    delta = cliffs_delta(scores_a, scores_b)
    significant = bool(p_value < alpha)

    interpretation = (
        f"Diferença mediana de {np.median(scores_a - scores_b):+.4f} entre os folds. "
        f"{'Significativa' if significant else 'Não significativa'} (p={p_value:.4f}), "
        f"efeito {interpret_effect_size(delta)} (delta de Cliff={delta:+.3f})."
    )

    return TestResult(
        test="wilcoxon",
        statistic=statistic,
        p_value=p_value,
        p_value_corrected=p_value,
        significant=significant,
        effect_size=delta,
        interpretation=interpretation,
    )


def friedman_test(scores_by_model: dict[str, np.ndarray], alpha: float = 0.05) -> TestResult:
    """Compara três ou mais modelos ao longo dos folds (teste de Friedman).

    Parameters
    ----------
    scores_by_model : dict of str to np.ndarray
        Métrica por fold de cada modelo.
    alpha : float, optional
        Nível de significância, by default 0.05.

    Returns
    -------
    TestResult
        Resultado do teste, com o ranking médio na interpretação.

    Raises
    ------
    ValueError
        Se houver menos de três modelos.

    Examples
    --------
    >>> friedman_test(
    ...     {"a": np.array([0.7, 0.8]), "b": np.array([0.6, 0.7]), "c": np.array([0.5, 0.6])}
    ... ).test
    'friedman'
    """
    if len(scores_by_model) < 3:
        raise ValueError(
            f"O teste de Friedman exige ao menos 3 modelos; recebidos {len(scores_by_model)}. "
            "Para dois modelos, use o teste de Wilcoxon."
        )

    names = sorted(scores_by_model)
    samples = [np.asarray(scores_by_model[name]) for name in names]
    # `scipy.stats.friedmanchisquare` devolve uma tupla sem tipagem precisa
    # nos stubs; convertida logo após a chamada (mesmo motivo do Wilcoxon).
    statistic_raw, p_value_raw = stats.friedmanchisquare(*samples)
    statistic, p_value = float(statistic_raw), float(p_value_raw)

    # Ranking médio: 1 = melhor. `-matrix` inverte porque rankdata ordena de
    # forma crescente e a métrica é "maior é melhor".
    matrix = np.vstack(samples).T
    ranks = np.apply_along_axis(stats.rankdata, 1, -matrix)
    mean_ranks = dict(zip(names, ranks.mean(axis=0).tolist(), strict=True))
    ordered = sorted(mean_ranks.items(), key=operator.itemgetter(1))

    significant = bool(p_value < alpha)
    interpretation = (
        "Ranking médio (1 = melhor): "
        + ", ".join(f"{name}={rank:.2f}" for name, rank in ordered)
        + ". "
        + (
            f"Há diferença significativa entre os modelos (p={p_value:.4f}); "
            "aplique o pós-teste de Nemenyi para identificar os pares."
            if significant
            else f"Nenhuma diferença significativa entre os modelos (p={p_value:.4f})."
        )
    )

    return TestResult(
        test="friedman",
        statistic=statistic,
        p_value=p_value,
        p_value_corrected=p_value,
        significant=significant,
        effect_size=0.0,
        interpretation=interpretation,
    )


def nemenyi_critical_difference(n_models: int, n_folds: int, alpha: float = 0.05) -> float:
    """Calcula a diferença crítica do pós-teste de Nemenyi.

    Dois modelos diferem significativamente quando a distância entre seus
    rankings médios excede este valor.

    Parameters
    ----------
    n_models : int
        Número de modelos comparados.
    n_folds : int
        Número de folds (blocos).
    alpha : float, optional
        Nível de significância, by default 0.05.

    Returns
    -------
    float
        Diferença crítica.

    Examples
    --------
    >>> round(nemenyi_critical_difference(3, 5), 2) > 0
    True
    """
    # Valores críticos q_alpha da distribuição do range studentizado
    # (dividida por raiz de 2), tabelados para alpha = 0,05.
    critical_values = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    q_alpha = critical_values.get(n_models, 3.164)
    if alpha != 0.05:
        logger.warning("Valores críticos de Nemenyi tabelados apenas para alpha=0,05.")

    return float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_folds)))


def _run_friedman_comparison(
    scores_by_model: dict[str, np.ndarray],
    config: StatisticsSection,
    names: list[str],
) -> dict[str, Any]:
    """Executa o teste de Friedman e a diferença crítica de Nemenyi, se aplicável."""
    if not config.friedman.enabled or len(names) < 3:
        return {}

    friedman = friedman_test(scores_by_model, alpha=config.alpha)
    n_folds = len(next(iter(scores_by_model.values())))
    return {
        "friedman": friedman.__dict__,
        "nemenyi_critical_difference": nemenyi_critical_difference(
            len(names), n_folds, config.alpha
        ),
    }


def _apply_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Aplica a correção de Bonferroni simples a uma lista de p-valores brutos."""
    return [min(1.0, p_value * len(p_values)) for p_value in p_values]


def _correct_pairwise_p_values(tests: list[TestResult], method: str) -> list[float]:
    """Aplica a correção para múltiplas comparações configurada aos p-valores brutos."""
    p_values = [test.p_value for test in tests]
    if method == "holm":
        return holm_correction(p_values)
    if method == "bonferroni":
        return _apply_bonferroni_correction(p_values)
    return p_values


def _compute_pairwise_tests(
    scores_by_model: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    config: StatisticsSection,
) -> list[TestResult]:
    """Executa o teste de Wilcoxon para cada par de modelos."""
    return [
        wilcoxon_test(
            scores_by_model[first],
            scores_by_model[second],
            alternative=config.wilcoxon.alternative or "two-sided",
            alpha=config.alpha,
        )
        for first, second in pairs
    ]


def _build_pairwise_results(
    pairs: list[tuple[str, str]],
    tests: list[TestResult],
    corrected: list[float],
    alpha: float,
) -> dict[str, Any]:
    """Monta o dicionário de resultados pareados, com o p-valor corrigido."""
    return {
        f"{first}_vs_{second}": test.__dict__
        | {
            "p_value_corrected": adjusted,
            "significant": adjusted < alpha,
        }
        for (first, second), test, adjusted in zip(pairs, tests, corrected, strict=True)
    }


def _log_pairwise_summary(pairwise: dict[str, Any], n_pairs: int, correction_method: str) -> None:
    """Registra quantas comparações par a par foram significativas após a correção."""
    n_significant = sum(1 for entry in pairwise.values() if entry["significant"])
    logger.info(
        "Comparações par a par: %d de %d significativas após correção %s.",
        n_significant,
        n_pairs,
        correction_method,
    )


def _run_pairwise_comparison(
    scores_by_model: dict[str, np.ndarray],
    config: StatisticsSection,
    names: list[str],
) -> dict[str, Any]:
    """Compara todos os pares de modelos via Wilcoxon, com correção e log do resumo."""
    if not config.wilcoxon.enabled or len(names) < 2:
        return {}

    pairs = list(combinations(names, 2))
    tests = _compute_pairwise_tests(scores_by_model, pairs, config)
    corrected = _correct_pairwise_p_values(tests, config.multiple_comparison_correction)
    pairwise = _build_pairwise_results(pairs, tests, corrected, config.alpha)

    _log_pairwise_summary(pairwise, len(pairs), config.multiple_comparison_correction)
    return {"pairwise": pairwise}


def compare_all_models(
    scores_by_model: dict[str, np.ndarray],
    config: StatisticsSection,
) -> dict[str, Any]:
    """Executa a bateria completa de comparações entre modelos.

    Parameters
    ----------
    scores_by_model : dict of str to np.ndarray
        Métrica por fold de cada modelo.
    config : StatisticsSection
        Seção ``statistics`` de ``configs/evaluation.yaml``.

    Returns
    -------
    dict
        ``friedman`` (se aplicável), ``pairwise`` (Wilcoxon corrigido) e
        ``nemenyi_critical_difference``.

    Examples
    --------
    >>> compare_all_models(scores, config.evaluation.statistics)  # doctest: +SKIP
    """
    names = sorted(scores_by_model)

    results: dict[str, Any] = {}
    results.update(_run_friedman_comparison(scores_by_model, config, names))
    results.update(_run_pairwise_comparison(scores_by_model, config, names))
    return results
