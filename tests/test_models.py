"""Testes dos modelos, da fábrica e da persistência."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from constants.labels import CLASS_ORDER
from exceptions.model import ModelNotFittedError, ModelPersistenceError, UnknownModelError
from models.base import UserDataset
from models.factory import create_model, create_models
from models.hybrid import HybridClassifier, split_feature_blocks
from models.persistence import load_metadata, load_model, save_model
from models.traditional import TabularClassifier, build_estimator
from models.transformer import aggregate_user_probabilities, flatten_user_texts
from training.trainer import build_dataset, split_features


@pytest.fixture
def dataset(feature_matrix: pl.DataFrame) -> UserDataset:
    """Conjunto completo montado a partir da matriz sintética."""
    return build_dataset(feature_matrix)


@pytest.fixture
def train_test(
    feature_matrix: pl.DataFrame, splits: pl.DataFrame
) -> tuple[UserDataset, UserDataset]:
    """Conjuntos de treino e teste, sem sobreposição de usuários."""
    train = build_dataset(split_features(feature_matrix, splits, "train"))
    test = build_dataset(split_features(feature_matrix, splits, "test"))
    return train, test


class TestUserDataset:
    """Testes do contêiner de dados dos modelos."""

    def test_tamanho(self, dataset: UserDataset) -> None:
        """O tamanho corresponde ao número de usuários."""
        assert len(dataset) == dataset.features.shape[0]

    def test_tem_rotulos(self, dataset: UserDataset) -> None:
        """A matriz sintética inclui rótulos."""
        assert dataset.has_labels

    def test_exige_textos_com_erro_claro(self, dataset: UserDataset) -> None:
        """Modelos textuais falham com mensagem que aponta a montagem."""
        with pytest.raises(ValueError, match="textos"):
            dataset.require_texts()

    def test_exige_sequencias_com_erro_claro(self, dataset: UserDataset) -> None:
        """Modelos recorrentes falham apontando a etapa 'embed'."""
        with pytest.raises(ValueError, match="embed"):
            dataset.require_sequences()


class TestFabricaDeEstimadores:
    """Testes da construção de estimadores do scikit-learn."""

    def test_cria_dummy(self) -> None:
        """O baseline trivial é sempre construível."""
        assert type(build_estimator("dummy", {"strategy": "stratified"}, 3)).__name__ == (
            "DummyClassifier"
        )

    def test_cria_regressao_logistica(self) -> None:
        """A regressão logística é construída com seus hiperparâmetros."""
        estimator = cast(LogisticRegression, build_estimator("logistic_regression", {"C": 0.5}, 3))
        assert estimator.C == 0.5

    def test_estimador_desconhecido_e_rejeitado(self) -> None:
        """Um nome não registrado falha listando os disponíveis."""
        with pytest.raises(UnknownModelError, match="desconhecido"):
            build_estimator("modelo_inexistente", {}, 3)


class TestClassificadorTabular:
    """Testes do classificador sobre a matriz de atributos."""

    def test_treina_e_preve(self, train_test: tuple[UserDataset, UserDataset]) -> None:
        """O ciclo completo de treino e predição funciona."""
        train, test = train_test
        model = TabularClassifier(
            name="teste",
            params={"strategy": "stratified", "random_state": 42},
            estimator_name="dummy",
        )
        model.fit(train)
        predictions = model.predict(test)

        assert len(predictions) == len(test)
        assert model.is_fitted

    def test_probabilidades_somam_um(self, train_test: tuple[UserDataset, UserDataset]) -> None:
        """As probabilidades formam uma distribuição válida."""
        train, test = train_test
        model = TabularClassifier(
            name="lr",
            params={"max_iter": 500, "random_state": 42},
            estimator_name="logistic_regression",
        )
        model.fit(train)
        proba = model.predict_proba(test)

        assert proba.shape == (len(test), len(CLASS_ORDER))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predizer_antes_de_treinar_falha(self, dataset: UserDataset) -> None:
        """Prever sem treinar é erro de uso e falha alto."""
        model = TabularClassifier(name="teste", estimator_name="dummy")

        with pytest.raises(ModelNotFittedError, match="não foi treinado"):
            model.predict(dataset)

    def test_aprende_sinal_real(self, train_test: tuple[UserDataset, UserDataset]) -> None:
        """Teste de funcionalidade mínima: o modelo supera o acaso nos dados sintéticos."""
        train, test = train_test
        model = TabularClassifier(
            name="lr",
            params={"max_iter": 1000, "random_state": 42, "class_weight": "balanced"},
            estimator_name="logistic_regression",
        )
        model.fit(train)

        assert test.labels is not None
        accuracy = float((model.predict(test) == test.labels).mean())
        assert accuracy > 1.0 / len(CLASS_ORDER)

    def test_importancia_de_atributos(self, train_test: tuple[UserDataset, UserDataset]) -> None:
        """A importância é devolvida com os nomes reais das colunas."""
        train, _ = train_test
        model = TabularClassifier(
            name="lr",
            params={"max_iter": 500, "random_state": 42},
            estimator_name="logistic_regression",
        )
        model.fit(train)
        importance = model.feature_importances()

        assert importance is not None
        assert set(importance).issubset(set(train.feature_names))

    def test_treino_exige_rotulos(self, feature_matrix: pl.DataFrame) -> None:
        """Sem rótulos, o treino falha com mensagem clara."""
        sem_rotulo = build_dataset(feature_matrix.drop("user_label"))
        model = TabularClassifier(name="teste", estimator_name="dummy")

        with pytest.raises(ValueError, match="rótulos"):
            model.fit(sem_rotulo)

    def test_treina_com_classe_ausente_no_fold(self) -> None:
        """Um fold sem nenhum usuário de uma classe não deve quebrar o XGBoost.

        Reproduz o cenário em que a partição de treino de um fold, por
        composição dos dados, fica sem nenhum representante de uma classe
        intermediária (ex.: 'depressao'). Sem o reindexamento local dos
        rótulos, `unique(y) = [0, 2]` faz o XGBoost levantar
        ``ValueError: Invalid classes inferred from unique values of `y``.
        """
        rng = np.random.default_rng(42)
        features = rng.normal(size=(20, 4))
        # Apenas as classes 0 (controle) e 2 (ideacao_suicida) presentes;
        # a classe 1 (depressao) está totalmente ausente deste treino.
        labels = np.array([0] * 10 + [2] * 10)
        train = UserDataset(
            user_ids=[f"u{i}" for i in range(20)],
            features=features,
            feature_names=[f"f{i}" for i in range(4)],
            labels=labels,
        )

        model = TabularClassifier(
            name="xgb", params={"n_estimators": 10, "random_state": 42}, estimator_name="xgboost"
        )
        model.fit(train)
        proba = model.predict_proba(train)

        assert proba.shape == (len(train), len(CLASS_ORDER))
        assert np.allclose(proba.sum(axis=1), 1.0)
        # A classe nunca vista no treino não pode receber probabilidade.
        assert np.allclose(proba[:, 1], 0.0)

    def test_descricao_para_rastreamento(self) -> None:
        """A descrição alimenta o MLflow e o model card."""
        model = TabularClassifier(name="xgb", params={"max_depth": 6}, estimator_name="xgboost")
        description = model.describe()

        assert description["name"] == "xgb"
        assert description["params"]["max_depth"] == 6


class TestModeloHibrido:
    """Testes do modelo híbrido — a contribuição metodológica central."""

    def test_separa_blocos(self) -> None:
        """As colunas semânticas são separadas das estruturadas pelo prefixo."""
        semantic, structured = split_feature_blocks(["sem_mean_000", "emo_polarity_mean"])

        assert semantic == [0]
        assert structured == [1]

    def test_treina_com_pca_nos_embeddings(
        self, train_test: tuple[UserDataset, UserDataset]
    ) -> None:
        """O bloco semântico passa por PCA; o estruturado entra inteiro."""
        train, test = train_test
        model = HybridClassifier(
            name="hibrido",
            params={"head": "logistic_regression", "max_iter": 500, "random_state": 42},
            n_components=2,
        )
        model.fit(train)
        proba = model.predict_proba(test)

        assert proba.shape == (len(test), len(CLASS_ORDER))
        assert model.n_semantic_ > 0

    def test_nomes_apos_transformacao(self, train_test: tuple[UserDataset, UserDataset]) -> None:
        """As componentes do PCA recebem nome próprio, não o da coluna original."""
        train, _ = train_test
        model = HybridClassifier(
            name="hibrido",
            params={"head": "logistic_regression", "max_iter": 500, "random_state": 42},
            n_components=2,
        )
        model.fit(train)
        names = model.transformed_feature_names()

        assert any(name.startswith("sem_pca_") for name in names)


class TestAgregacaoPorUsuario:
    """Testes da agregação tweet -> usuário usada pelo Transformer."""

    def test_achata_textos_propagando_rotulo(self) -> None:
        """Cada tweet herda o rótulo do usuário para o fine-tuning."""
        _, _, labels = flatten_user_texts({"u_a": ["x", "y"]}, ["u_a"], np.array([1]))
        assert labels is not None
        assert labels.tolist() == [1, 1]

    def test_media_das_probabilidades(self) -> None:
        """A média é mais robusta a rótulos individuais ruidosos que o voto."""
        result = aggregate_user_probabilities(
            np.array([[0.2, 0.8], [0.6, 0.4]]), ["u_a", "u_a"], ["u_a"], 2
        )
        assert result[0].tolist() == pytest.approx([0.4, 0.6])

    def test_usuario_sem_tweets_recebe_uniforme(self) -> None:
        """Distribuição uniforme, e não zeros — zerar tornaria o argmax arbitrário."""
        result = aggregate_user_probabilities(np.zeros((0, 2)), [], ["u_a"], 2)
        assert result[0].tolist() == pytest.approx([0.5, 0.5])

    def test_voto_majoritario(self) -> None:
        """A agregação por voto majoritário é suportada como alternativa."""
        result = aggregate_user_probabilities(
            np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]),
            ["u_a"] * 3,
            ["u_a"],
            2,
            strategy="majority",
        )
        assert result[0][0] > result[0][1]


class TestFabricaDeModelos:
    """Testes da fábrica que instancia os modelos do YAML."""

    def test_cria_todos_os_modelos_principais(self, config) -> None:
        """A comparação principal é instanciável sem exceção."""
        models = create_models(config)
        assert "dummy" in models
        assert all(model.name == name for name, model in models.items())

    def test_restringe_a_modelos_especificos(self, config) -> None:
        """A opção --models restringe a execução."""
        models = create_models(config, only=["dummy"])
        assert set(models) == {"dummy"}

    def test_modelo_inexistente_e_rejeitado(self, config) -> None:
        """Um nome não declarado no YAML falha com a lista de disponíveis."""
        with pytest.raises(UnknownModelError, match="não declarados"):
            create_models(config, only=["modelo_que_nao_existe"])

    def test_exploratorios_ficam_de_fora_por_padrao(self, config) -> None:
        """Sem a flag, a extensão exploratória não é instanciada."""
        assert len(create_models(config)) < len(create_models(config, include_exploratory=True))

    def test_estimador_desconhecido_e_rejeitado(self, config) -> None:
        """Um estimador fora do registro falha na criação."""
        from config.settings import ModelSpec

        spec = ModelSpec(scope="comparison", estimator="inexistente", params={})
        with pytest.raises(UnknownModelError):
            create_model("x", spec, config)


class TestPersistencia:
    """Testes da persistência de modelos com metadados."""

    def test_salva_e_carrega(
        self, train_test: tuple[UserDataset, UserDataset], tmp_path: Path
    ) -> None:
        """O modelo carregado produz as mesmas predições do original."""
        train, test = train_test
        model = TabularClassifier(
            name="lr",
            params={"max_iter": 500, "random_state": 42},
            estimator_name="logistic_regression",
        )
        model.fit(train)

        save_model(model, tmp_path, dataset_hash="abc123")
        loaded = load_model(tmp_path / "lr.joblib")

        assert np.array_equal(model.predict(test), loaded.predict(test))

    def test_metadados_incluem_rastreabilidade(
        self, train_test: tuple[UserDataset, UserDataset], tmp_path: Path
    ) -> None:
        """Os metadados fecham a tríade código + ambiente + dados."""
        train, _ = train_test
        model = TabularClassifier(name="dummy", estimator_name="dummy")
        model.fit(train)
        save_model(model, tmp_path, dataset_hash="abc123")

        metadata = load_metadata(tmp_path / "dummy.joblib")
        assert metadata["dataset_hash"] == "abc123"
        assert "git_sha" in metadata
        assert "environment" in metadata

    def test_salvar_modelo_nao_treinado_falha(self, tmp_path: Path) -> None:
        """Persistir um modelo não treinado é erro de uso."""
        model = TabularClassifier(name="x", estimator_name="dummy")

        with pytest.raises(ModelPersistenceError, match="não foi treinado"):
            save_model(model, tmp_path)

    def test_carregar_inexistente_falha(self, tmp_path: Path) -> None:
        """Carregar um arquivo ausente falha com erro tipado."""
        with pytest.raises(ModelPersistenceError, match="não encontrado"):
            load_model(tmp_path / "inexistente.joblib")
