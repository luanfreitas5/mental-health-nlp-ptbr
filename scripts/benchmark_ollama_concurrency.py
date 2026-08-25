"""Mede o teto real de concorrência que o servidor Ollama/GPU aguenta.

Contexto
--------
``configs/llm.yaml`` fixa ``psychological_features.max_concurrency`` (etapa 4
— extração do vetor psicológico). O valor não pode ser "chutado": depende da
GPU, do modelo e de ``OLLAMA_NUM_PARALLEL`` na máquina que realmente roda o
pipeline — não da máquina de desenvolvimento. Este script sobe o número de
chamadas simultâneas ao servidor Ollama configurado (o mesmo caminho de
código de :class:`labeling.llm.OllamaClient`, com ``use_cache=False`` para
medir inferência real), em níveis crescentes, e reporta vazão, latência e
taxa de erro por nível — a base empírica para escolher ``max_concurrency``.

Como usar
---------
Rodar na máquina com o servidor Ollama e a GPU de produção (não localmente,
se o desenvolvimento for numa máquina diferente)::

    uv run python scripts/benchmark_ollama_concurrency.py

Customizar os níveis testados e o número de requisições por nível::

    uv run python scripts/benchmark_ollama_concurrency.py \\
        --levels 2 4 8 12 16 24 32 --requests-per-level 15

O script para de subir os níveis assim que um deles apresenta erro (timeout,
conexão recusada) — sinal de que o teto foi ultrapassado — a menos que
``--no-stop-on-error`` seja passado. Ao final, imprime uma recomendação e
grava o detalhe de cada nível em ``reports/metrics/ollama_concurrency_benchmark.json``.

O número recomendado deve ser copiado manualmente para
``configs/llm.yaml`` (``psychological_features.max_concurrency``) — o script
não edita configuração.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.table import Table  # noqa: E402

from config.logging import CONSOLE, configure_logging, get_logger  # noqa: E402
from config.paths import get_paths  # noqa: E402
from config.settings import LLMConfig, load_config  # noqa: E402
from exceptions.model import LLMError  # noqa: E402
from labeling.llm import OllamaClient  # noqa: E402
from labeling.prompt import build_psychological_prompt  # noqa: E402
from utils.files import write_json  # noqa: E402

logger = get_logger(__name__)

#: Frases sintéticas em pt-BR, sem PII, usadas só para gerar carga de
#: inferência realista (mesmo tamanho médio de um tweet normalizado).
SYNTHETIC_SENTENCES: tuple[str, ...] = (
    "hoje o dia começou meio devagar mas melhorou depois do almoço",
    "não consegui dormir direito essa semana, muita coisa na cabeça",
    "fui caminhar de manhã e o sol estava ótimo, ajudou bastante",
    "cansado de novo, parece que o sono não repõe nada ultimamente",
    "conversei com um amigo hoje e me senti mais leve depois",
    "trabalho puxado essa semana, mal deu tempo de parar",
    "assisti um filme bom ontem à noite, distraiu um pouco",
    "acordei ansioso sem motivo claro, só uma sensação ruim",
    "consegui terminar uma tarefa que estava adiando há dias",
    "silêncio em casa hoje, meio estranho mas tranquilo",
)


@dataclass
class LevelResult:
    """Resultado agregado de um nível de concorrência testado.

    Attributes
    ----------
    level : int
        Número de chamadas simultâneas (``max_workers``) testado.
    n_requests : int
        Total de requisições disparadas neste nível.
    n_ok, n_error : int
        Requisições concluídas com sucesso / com erro.
    wall_seconds : float
        Tempo total decorrido para concluir o lote.
    latencies_seconds : list of float
        Latência individual de cada requisição bem-sucedida.
    """

    level: int
    n_requests: int
    n_ok: int
    n_error: int
    wall_seconds: float
    latencies_seconds: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        """Fração de requisições que falharam neste nível."""
        return self.n_error / self.n_requests if self.n_requests else 0.0

    @property
    def throughput_req_per_s(self) -> float:
        """Requisições bem-sucedidas por segundo (vazão)."""
        return self.n_ok / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def p95_latency_seconds(self) -> float:
        """Percentil 95 da latência (0 se não houver amostras)."""
        if not self.latencies_seconds:
            return 0.0
        ordered = sorted(self.latencies_seconds)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return ordered[index]

    def to_dict(self) -> dict[str, float | int]:
        """Serializa o resultado para gravação em JSON."""
        return {
            "level": self.level,
            "n_requests": self.n_requests,
            "n_ok": self.n_ok,
            "n_error": self.n_error,
            "error_rate": round(self.error_rate, 4),
            "wall_seconds": round(self.wall_seconds, 3),
            "throughput_req_per_s": round(self.throughput_req_per_s, 3),
            "mean_latency_seconds": round(statistics.fmean(self.latencies_seconds), 3)
            if self.latencies_seconds
            else 0.0,
            "p95_latency_seconds": round(self.p95_latency_seconds, 3),
        }


def build_synthetic_batch(n_tweets: int, request_index: int) -> list[str]:
    """Monta um lote de tweets sintéticos, distinto por requisição.

    O índice da requisição entra no texto para que cada prompt gere um hash
    diferente — sem isso, o cache de :meth:`OllamaClient.generate` (mesmo
    com ``use_cache=False`` na chamada) tornaria requisições repetidas
    triviais de identificar em logs, dificultando a leitura dos resultados.

    Parameters
    ----------
    n_tweets : int
        Tweets por lote (equivalente a ``batch_size_tweets``).
    request_index : int
        Índice da requisição, usado para variar o conteúdo.

    Returns
    -------
    list of str
        Tweets sintéticos, sem PII.
    """
    n_sentences = len(SYNTHETIC_SENTENCES)
    return [
        f"[{request_index}.{i}] {SYNTHETIC_SENTENCES[(request_index + i) % n_sentences]}"
        for i in range(n_tweets)
    ]


def run_level(
    client: OllamaClient,
    config: LLMConfig,
    level: int,
    n_requests: int,
    n_tweets: int,
) -> LevelResult:
    """Dispara ``n_requests`` chamadas ao LLM com até ``level`` em paralelo.

    Parameters
    ----------
    client : OllamaClient
        Cliente já apontado para o servidor configurado.
    config : LLMConfig
        Configuração de ``configs/llm.yaml`` (fonte do modelo e dos prompts).
    level : int
        Concorrência testada (``ThreadPoolExecutor(max_workers=level)``).
    n_requests : int
        Total de requisições disparadas neste nível.
    n_tweets : int
        Tweets por lote em cada requisição.

    Returns
    -------
    LevelResult
        Métricas agregadas do nível.
    """
    settings = config.psychological_features

    def call(request_index: int) -> float:
        prompt = build_psychological_prompt(build_synthetic_batch(n_tweets, request_index), config)
        start = time.perf_counter()
        client.generate(
            prompt,
            model=settings.model,
            temperature=settings.temperature,
            seed=settings.seed,
            num_ctx=settings.num_ctx,
            use_cache=False,
        )
        return time.perf_counter() - start

    n_ok = 0
    n_error = 0
    latencies: list[float] = []

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as executor:
        futures = {executor.submit(call, i): i for i in range(n_requests)}
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
                n_ok += 1
            except Exception as error:  # nível de teste tolera qualquer falha (rede, LLM, timeout)
                n_error += 1
                logger.warning("Requisição falhou no nível %d: %s", level, error)
    wall_seconds = time.perf_counter() - wall_start

    return LevelResult(
        level=level,
        n_requests=n_requests,
        n_ok=n_ok,
        n_error=n_error,
        wall_seconds=wall_seconds,
        latencies_seconds=latencies,
    )


def recommend_level(results: list[LevelResult]) -> int | None:
    """Recomenda o nível de concorrência a partir dos resultados medidos.

    Escolhe o **menor** nível cuja vazão está a até 5% do pico observado
    entre os níveis sem nenhum erro — preferir o menor nível equivalente
    evita saturar a GPU sem ganho real de vazão.

    Parameters
    ----------
    results : list of LevelResult
        Resultados de :func:`run_level`, um por nível testado.

    Returns
    -------
    int or None
        Nível recomendado, ou ``None`` se todos os níveis tiveram erro.
    """
    clean = [r for r in results if r.error_rate == 0.0 and r.n_ok > 0]
    if not clean:
        return None

    best_throughput = max(r.throughput_req_per_s for r in clean)
    near_peak = [r for r in clean if r.throughput_req_per_s >= 0.95 * best_throughput]
    return min(near_peak, key=lambda r: r.level).level


def print_results_table(results: list[LevelResult]) -> None:
    """Imprime uma tabela ``rich`` com as métricas de cada nível testado."""
    table = Table(title="Benchmark de concorrência — Ollama")
    table.add_column("Nível", justify="right")
    table.add_column("OK/Total", justify="right")
    table.add_column("Erros", justify="right")
    table.add_column("Vazão (req/s)", justify="right")
    table.add_column("Latência média (s)", justify="right")
    table.add_column("Latência p95 (s)", justify="right")

    for result in results:
        table.add_row(
            str(result.level),
            f"{result.n_ok}/{result.n_requests}",
            f"{result.error_rate:.0%}",
            f"{result.throughput_req_per_s:.2f}",
            f"{statistics.fmean(result.latencies_seconds):.2f}"
            if result.latencies_seconds
            else "-",
            f"{result.p95_latency_seconds:.2f}",
        )

    CONSOLE.print(table)


def build_parser() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Testa níveis crescentes de concorrência contra o servidor Ollama "
            "configurado em configs/llm.yaml e recomenda um valor para "
            "psychological_features.max_concurrency."
        )
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[2, 4, 8, 12, 16, 24, 32],
        help="Níveis de concorrência a testar, em ordem crescente (padrão: 2 4 8 12 16 24 32).",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=16,
        help="Requisições disparadas em cada nível (padrão: 16).",
    )
    parser.add_argument(
        "--n-tweets",
        type=int,
        default=None,
        help=(
            "Tweets por lote em cada requisição (padrão: "
            "psychological_features.batch_size_tweets do configs/llm.yaml)."
        ),
    )
    parser.add_argument(
        "--no-stop-on-error",
        action="store_true",
        help="Continua testando níveis mais altos mesmo após um nível com erro.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Caminho do JSON de saída (padrão: reports/metrics/ollama_concurrency_benchmark.json)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Sobrescreve o nível de log configurado em configs/logging.yaml.",
    )
    return parser


def main() -> int:
    """Ponto de entrada do script.

    Returns
    -------
    int
        0 em sucesso (mesmo que nenhum nível limpo tenha sido encontrado —
        o resultado bruto ainda é útil); 1 se o servidor Ollama estiver
        inacessível ou o modelo não estiver disponível.
    """
    args = build_parser().parse_args()
    configure_logging(level=args.log_level)

    config = load_config().llm
    settings = config.psychological_features
    n_tweets = args.n_tweets or settings.batch_size_tweets

    client = OllamaClient(config, cache_dir=None)  # sem cache: mede inferência real
    try:
        client.ensure_model(settings.model)
    except LLMError:
        logger.exception("Não foi possível iniciar o benchmark")
        return 1

    logger.info(
        "Benchmark iniciado: modelo=%s, host=%s, %d tweets/lote, níveis=%s.",
        settings.model,
        config.ollama.host,
        n_tweets,
        args.levels,
    )

    results: list[LevelResult] = []
    for level in sorted(args.levels):
        logger.info("Testando nível de concorrência %d...", level)
        result = run_level(client, config, level, args.requests_per_level, n_tweets)
        results.append(result)
        logger.info(
            "Nível %d: %d/%d ok, vazão=%.2f req/s, p95=%.2fs.",
            level,
            result.n_ok,
            result.n_requests,
            result.throughput_req_per_s,
            result.p95_latency_seconds,
        )
        if result.error_rate > 0 and not args.no_stop_on_error:
            logger.warning(
                "Nível %d apresentou erros — parando (use --no-stop-on-error para continuar).",
                level,
            )
            break

    print_results_table(results)

    recommended = recommend_level(results)
    output_path = args.output or (get_paths().reports.metrics / "ollama_concurrency_benchmark.json")
    write_json(
        output_path,
        {
            "model": settings.model,
            "host": config.ollama.host,
            "n_tweets_per_batch": n_tweets,
            "requests_per_level": args.requests_per_level,
            "current_max_concurrency": settings.max_concurrency,
            "recommended_max_concurrency": recommended,
            "levels": [r.to_dict() for r in results],
        },
    )
    logger.info("Resultado detalhado gravado em %s.", output_path)

    if recommended is None:
        logger.error(
            "Nenhum nível ficou livre de erro — mantenha max_concurrency=%d e investigue "
            "OLLAMA_NUM_PARALLEL/VRAM antes de subir a concorrência.",
            settings.max_concurrency,
        )
        return 0

    logger.info(
        "Recomendação: psychological_features.max_concurrency = %d "
        "(atual em configs/llm.yaml: %d). Atualize o YAML manualmente.",
        recommended,
        settings.max_concurrency,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
