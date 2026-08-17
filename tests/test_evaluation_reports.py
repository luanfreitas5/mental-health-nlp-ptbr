"""Testes da montagem do relatório Markdown de avaliação."""

from __future__ import annotations

from evaluation.reports import _build_statistics_section


class TestSecaoEstatisticas:
    """Testes da seção 3 (testes estatísticos) do relatório Markdown."""

    def test_secao_vazia_sem_estatisticas(self) -> None:
        """Sem estatísticas calculadas, a seção inteira é omitida."""
        assert _build_statistics_section(None) == []
        assert _build_statistics_section({}) == []

    def test_secao_inclui_mcnemar(self) -> None:
        """Resultados de McNemar aparecem na seção, mesmo sem Friedman/Wilcoxon."""
        statistics = {
            "mcnemar": {
                "xgboost_vs_dummy": {
                    "p_value": 2.02e-79,
                    "significant": True,
                    "interpretation": "Diferença significativa; o primeiro modelo acerta mais.",
                },
            },
        }

        lines = _build_statistics_section(statistics)
        content = "\n".join(lines)

        assert "## 3. Testes estatísticos" in content
        assert "### Comparações par a par (McNemar)" in content
        assert "xgboost vs. dummy" in content
        assert "sim" in content
        assert "o primeiro modelo acerta mais" in content

    def test_secao_combina_pairwise_e_mcnemar(self) -> None:
        """Wilcoxon (por fold) e McNemar (no mesmo teste) podem coexistir na mesma seção."""
        statistics = {
            "pairwise": {
                "a_vs_b": {
                    "p_value": 0.01,
                    "p_value_corrected": 0.02,
                    "significant": True,
                    "effect_size": 0.5,
                },
            },
            "mcnemar": {
                "a_vs_b": {
                    "p_value": 0.03,
                    "significant": True,
                    "interpretation": "Diferença significativa.",
                },
            },
        }

        content = "\n".join(_build_statistics_section(statistics))

        assert "Comparações par a par (Wilcoxon com correção de Holm)" in content
        assert "Comparações par a par (McNemar)" in content
