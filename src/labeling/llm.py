"""Cliente Ollama e extração do vetor psicológico por LLM.

Todo o processamento é **local**. Enviar textos de pessoas em sofrimento
psíquico a uma API de terceiros seria transferência internacional de dados
pessoais sensíveis sem base legal adequada (LGPD, arts. 11 e 33) — por isso o
provedor é o Ollama, rodando na própria máquina.

Três salvaguardas de engenharia merecem destaque:

* **Saída estruturada validada.** A resposta é validada por Pydantic; se vier
  fora do schema, é reparada e revalidada até ``max_repairs`` e só então
  descartada. Um valor inventado seria pior do que um ausente.
* **Cache por hash do prompt.** Reexecutar a etapa não gasta inferência de
  novo, o que torna o experimento retomável — a extração leva horas.
* **Prompt nunca vai para o log.** Apenas o hash e os metadados são
  registrados (``llm.safeguards.log_prompt_content``).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field, ValidationError

from config.logging import get_logger
from config.settings import LLMConfig
from exceptions.model import LLMResponseError, LLMUnavailableError, MissingDependencyError
from labeling.prompt import Prompt, build_psychological_prompt
from utils.files import read_json, write_json
from utils.hashing import hash_payload
from utils.progress import build_progress

logger = get_logger(__name__)


class PsychologicalVector(BaseModel):
    """Vetor psicológico devolvido pelo LLM, com faixas validadas.

    Attributes
    ----------
    tristeza, isolamento, esperanca, ansiedade, risco_suicida : float
        Intensidade de cada dimensão, em ``[0, 1]``. ``esperanca`` é a única
        dimensão positiva: valores altos indicam perspectiva de futuro.
    """

    tristeza: float = Field(ge=0.0, le=1.0)
    isolamento: float = Field(ge=0.0, le=1.0)
    esperanca: float = Field(ge=0.0, le=1.0)
    ansiedade: float = Field(ge=0.0, le=1.0)
    risco_suicida: float = Field(ge=0.0, le=1.0)


class UserClassification(BaseModel):
    """Classificação de um usuário feita pelo LLM."""

    classe: str
    confianca: float = Field(ge=0.0, le=1.0, default=0.5)
    justificativa: str = ""


def _import_ollama() -> Any:
    """Importa o cliente ``ollama`` sob demanda."""
    try:
        import ollama
    except ImportError as error:
        raise MissingDependencyError(
            "O pacote 'ollama' não está instalado. Rode 'uv sync --extra llm' e garanta "
            "que o servidor Ollama esteja ativo (docs/guides/llm.md)."
        ) from error
    return ollama


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extrai o primeiro objeto JSON de uma resposta possivelmente verborrágica.

    Mesmo com ``format="json"`` e temperatura zero, modelos menores às vezes
    envolvem o objeto em texto explicativo ou em cerca de código. Recortar o
    trecho entre as chaves externas recupera a maior parte desses casos sem
    precisar de uma nova chamada.

    Parameters
    ----------
    raw : str
        Resposta bruta do modelo.

    Returns
    -------
    dict
        Objeto JSON desserializado.

    Raises
    ------
    LLMResponseError
        Se não houver objeto JSON válido na resposta.

    Examples
    --------
    >>> extract_json_object('Claro! {"a": 1} espero ter ajudado')
    {'a': 1}
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if -1 in (start, end) or end < start:
        raise LLMResponseError("A resposta do LLM não contém nenhum objeto JSON.")

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as error:
        raise LLMResponseError(f"JSON inválido na resposta do LLM: {error}") from error

    if not isinstance(parsed, dict):
        raise LLMResponseError(f"Esperado um objeto JSON, veio {type(parsed).__name__}.")
    return parsed


class OllamaClient:
    """Cliente do Ollama com repetição, cache e registro seguro.

    Parameters
    ----------
    config : LLMConfig
        Configuração de ``configs/llm.yaml``.
    cache_dir : Path, optional
        Diretório do cache; ``None`` desativa o cache.

    Examples
    --------
    >>> client = OllamaClient(config.llm)  # doctest: +SKIP
    >>> client.generate(prompt, model="llama3.2")  # doctest: +SKIP
    """

    def __init__(self, config: LLMConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Instancia o cliente do Ollama apontando para o host configurado."""
        if self._client is not None:
            return self._client

        ollama = _import_ollama()
        host = self.config.ollama.host
        self._client = ollama.Client(host=host, timeout=self.config.ollama.timeout_seconds)
        return self._client

    def ensure_model(self, model: str) -> None:
        """Verifica se o modelo está disponível localmente.

        Falhar aqui é intencional: baixar vários GB no meio de um pipeline
        longo, sem que o usuário perceba, é pior do que interromper com uma
        instrução clara.

        Parameters
        ----------
        model : str
            Nome do modelo (ex.: ``"llama3.2"``).

        Raises
        ------
        LLMUnavailableError
            Se o servidor estiver inacessível ou o modelo não estiver baixado.

        Examples
        --------
        >>> client.ensure_model("llama3.2")  # doctest: +SKIP
        """
        client = self._get_client()
        try:
            response = client.list()
        except Exception as error:
            raise LLMUnavailableError(
                f"Servidor Ollama inacessível em {self.config.ollama.host}: {error}. "
                "Inicie o servidor com 'ollama serve'."
            ) from error

        available = {
            str(item.get("model", item.get("name", ""))).split(":")[0]
            for item in response.get("models", [])
        }
        if model.split(":", maxsplit=1)[0] not in available:
            if self.config.ollama.auto_pull:
                logger.info("Baixando o modelo '%s' (auto_pull ativo).", model)
                client.pull(model)
                return
            raise LLMUnavailableError(
                f"Modelo '{model}' não está disponível localmente. "
                f"Rode 'ollama pull {model}'. Disponíveis: {sorted(available)}"
            )

    def _cache_path(self, key: str) -> Path | None:
        """Retorna o caminho do arquivo de cache de uma chave."""
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def generate(
        self,
        prompt: Prompt,
        model: str,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Envia um prompt e devolve a resposta já desserializada.

        Parameters
        ----------
        prompt : Prompt
            Prompt construído por :mod:`labeling.prompt`.
        model : str
            Modelo do Ollama.
        temperature : float, optional
            Temperatura da amostragem, by default 0 (determinismo).
        seed : int, optional
            Semente de geração, by default 42.
        num_ctx : int, optional
            Tamanho da janela de contexto, by default 8192.
        use_cache : bool, optional
            Reaproveita respostas já obtidas, by default True.

        Returns
        -------
        dict
            Objeto JSON da resposta.

        Raises
        ------
        LLMUnavailableError
            Se todas as tentativas de comunicação falharem.
        LLMResponseError
            Se a resposta não contiver JSON válido.

        Examples
        --------
        >>> client.generate(prompt, model="llama3.2")  # doctest: +SKIP
        {'tristeza': 0.9, ...}
        """
        cache_key = hash_payload(
            {
                "system": prompt.system,
                "user": prompt.user,
                "version": prompt.version,
                "model": model,
                "temperature": temperature,
                "seed": seed,
            }
        )

        cache_path = self._cache_path(cache_key)
        if use_cache and cache_path and cache_path.is_file():
            logger.debug("Cache do LLM utilizado (hash=%s).", cache_key[:12])
            return read_json(cache_path)

        client = self._get_client()
        options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        delay = self.config.ollama.backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.config.ollama.retry_attempts + 1):
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user", "content": prompt.user},
                    ],
                    options=options,
                    format="json"
                    if self.config.psychological_features.response_format == "json"
                    else "",
                    keep_alive=self.config.ollama.keep_alive,
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    "Falha na chamada ao LLM (tentativa %d/%d): %s",
                    attempt,
                    self.config.ollama.retry_attempts,
                    error,
                )
                time.sleep(delay)
                delay *= 2
                continue

            content = response["message"]["content"]
            parsed = extract_json_object(content)

            if cache_path:
                write_json(cache_path, parsed)
            return parsed

        raise LLMUnavailableError(
            f"O LLM não respondeu após {self.config.ollama.retry_attempts} tentativas: {last_error}"
        )


class PsychologicalExtractor:
    """Extrai o vetor psicológico dos usuários usando o LLM.

    Parameters
    ----------
    config : LLMConfig
        Configuração de ``configs/llm.yaml``.
    cache_dir : Path, optional
        Diretório do cache de respostas.

    Examples
    --------
    >>> extractor = PsychologicalExtractor(config.llm)  # doctest: +SKIP
    >>> extractor.extract_frame(tweets)  # doctest: +SKIP
    """

    def __init__(self, config: LLMConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        cache = cache_dir
        if cache is None and config.psychological_features.cache.enabled:
            cache = Path(config.psychological_features.cache.path)
        self.client = OllamaClient(config, cache_dir=cache)

    def extract_batch(self, tweets: list[str]) -> PsychologicalVector | None:
        """Extrai o vetor psicológico de um lote de tweets de um usuário.

        Parameters
        ----------
        tweets : list of str
            Tweets normalizados, em ordem cronológica.

        Returns
        -------
        PsychologicalVector or None
            Vetor validado, ou ``None`` se a resposta não pôde ser reparada —
            é preferível perder um lote a inventar valores.

        Examples
        --------
        >>> extractor.extract_batch(["hoje foi difícil"])  # doctest: +SKIP
        PsychologicalVector(tristeza=0.9, ...)
        """
        settings = self.config.psychological_features
        prompt = build_psychological_prompt(tweets, self.config)

        for attempt in range(settings.max_repairs + 1):
            try:
                payload = self.client.generate(
                    prompt,
                    model=settings.model,
                    temperature=settings.temperature,
                    seed=settings.seed,
                    num_ctx=settings.num_ctx,
                    use_cache=attempt == 0,
                )
                return PsychologicalVector(**payload)
            except (LLMResponseError, ValidationError) as error:
                logger.warning(
                    "Resposta do LLM fora do schema (tentativa %d/%d): %s",
                    attempt + 1,
                    settings.max_repairs + 1,
                    error,
                )

        logger.error("Lote descartado: o LLM não produziu um vetor psicológico válido.")
        return None

    def extract_frame(self, tweets: pl.DataFrame) -> pl.DataFrame:
        """Extrai o vetor psicológico de todos os usuários de um DataFrame.

        Os lotes são montados sequencialmente (para preservar ``batch_index``
        por usuário), mas as chamadas ao LLM rodam em paralelo — até
        ``max_concurrency`` simultâneas — já que cada uma é uma chamada de
        rede bloqueante ao servidor Ollama, e não trabalho de CPU.

        Parameters
        ----------
        tweets : pl.DataFrame
            Tweets limpos, com ``user_id``, ``text_normalized`` e ``created_at``.

        Returns
        -------
        pl.DataFrame
            Um registro por lote, conforme
            :class:`schemas.tweets.PsychologicalScoreSchema`.

        Examples
        --------
        >>> extractor.extract_frame(tweets)  # doctest: +SKIP
        """
        from constants.columns import CREATED_AT, TEXT_NORMALIZED, USER_ID

        settings = self.config.psychological_features
        self.client.ensure_model(settings.model)

        groups = tweets.sort([USER_ID, CREATED_AT]).partition_by(
            USER_ID, as_dict=True, maintain_order=True
        )

        pending: list[dict[str, Any]] = []
        for (user_id,), user_frame in groups.items():
            texts = user_frame[TEXT_NORMALIZED].to_list()
            timestamps = user_frame[CREATED_AT].to_list()

            for batch_index, start in enumerate(range(0, len(texts), settings.batch_size_tweets)):
                batch = texts[start : start + settings.batch_size_tweets]
                window = timestamps[start : start + settings.batch_size_tweets]
                pending.append(
                    {
                        "user_id": user_id,
                        "batch_index": batch_index,
                        "batch": batch,
                        "window_start": min(window),
                        "window_end": max(window),
                    }
                )

        records: list[dict[str, Any]] = []

        with build_progress() as progress:
            task = progress.add_task("Extraindo vetor psicológico", total=len(pending))
            with ThreadPoolExecutor(max_workers=settings.max_concurrency) as executor:
                futures = {
                    executor.submit(self.extract_batch, item["batch"]): item for item in pending
                }
                try:
                    for future in as_completed(futures):
                        item = futures[future]
                        vector = future.result()
                        progress.advance(task)
                        if vector is None:
                            continue

                        records.append(
                            {
                                "user_id": item["user_id"],
                                "batch_index": item["batch_index"],
                                "n_tweets": len(item["batch"]),
                                "window_start": item["window_start"],
                                "window_end": item["window_end"],
                                **vector.model_dump(),
                                "model": settings.model,
                                "prompt_version": self.config.prompts.version,
                            }
                        )
                except BaseException:
                    # Interrompe lotes ainda não iniciados (ex.: LLMUnavailableError);
                    # os que já estão em execução terminam antes do 'with' fechar.
                    for other in futures:
                        other.cancel()
                    raise

        records.sort(key=lambda record: (record["user_id"], record["batch_index"]))

        logger.info(
            "Vetor psicológico extraído: %d lotes de %d usuários.", len(records), len(groups)
        )
        return pl.DataFrame(records) if records else pl.DataFrame()
