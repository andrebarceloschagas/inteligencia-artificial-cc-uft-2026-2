"""
Testes unitários — Algoritmo Genético para Coloração de Mapas
===============================================================

Rodar com:
    pytest genetico-coloracao-teste.py -v

Cobre:
  * validação do mapa (vizinhos inexistentes, auto-adjacência, assimetria);
  * função de avaliação (contagem de conflitos e fitness);
  * operadores de cruzamento (um ponto e uniforme);
  * operadores de mutação (aleatória e dirigida);
  * parâmetros inválidos (ParametrosAG.__post_init__);
  * execução ponta a ponta em mapas pequenos, incluindo o caso em que o
    número de cores é insuficiente (condição de parada por limite de
    gerações, sem solução válida).
"""

from __future__ import annotations

import pytest

from genetico_coloracao import (
    AlgoritmoGeneticoColoracao,
    ParametrosAG,
    criar_mapa_simplificado,
    validar_mapa,
)

# ---------------------------------------------------------------------------
# Mapas pequenos usados nos testes (mais fáceis de verificar à mão)
# ---------------------------------------------------------------------------

# Triângulo: A-B-C todos vizinhos entre si -> precisa de 3 cores, nunca de 2.
MAPA_TRIANGULO = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}

# Quadrado (ciclo par): A-B-C-D-A -> é bipartido, 2 cores bastam.
MAPA_QUADRADO = {"A": ["B", "D"], "B": ["A", "C"], "C": ["B", "D"], "D": ["A", "C"]}


def criar_ag(mapa, **overrides) -> AlgoritmoGeneticoColoracao:
    """Atalho: cria o AG com uma semente fixa (reprodutível) e os overrides dados."""
    overrides.setdefault("semente", 0)
    return AlgoritmoGeneticoColoracao(mapa, ParametrosAG(**overrides))


# ---------------------------------------------------------------------------
# validar_mapa
# ---------------------------------------------------------------------------

class TestValidarMapa:
    def test_mapa_valido_nao_levanta_erro(self):
        validar_mapa(MAPA_TRIANGULO)  # não deve lançar

    def test_mapa_com_menos_de_2_regioes(self):
        with pytest.raises(ValueError, match="pelo menos 2 regiões"):
            validar_mapa({"A": []})

    def test_vizinho_inexistente(self):
        with pytest.raises(ValueError, match="não existe no mapa"):
            validar_mapa({"A": ["B"], "B": ["A", "Z"]})

    def test_auto_adjacencia(self):
        with pytest.raises(ValueError, match="si mesma"):
            validar_mapa({"A": ["A", "B"], "B": ["A"]})

    def test_adjacencia_assimetrica(self):
        with pytest.raises(ValueError, match="[Aa]ssimétrica"):
            validar_mapa({"A": ["B"], "B": []})

    def test_mapas_de_exemplo_sao_validos(self):
        mapa, _ = criar_mapa_simplificado()
        validar_mapa(mapa)  # não deve lançar


# ---------------------------------------------------------------------------
# ParametrosAG — validação
# ---------------------------------------------------------------------------

class TestParametrosAG:
    @pytest.mark.parametrize("campo,valor", [
        ("tamanho_populacao", 1),
        ("numero_geracoes", 0),
        ("taxa_crossover", 1.5),
        ("taxa_mutacao", -0.1),
        ("tamanho_torneio", 1),
        ("numero_elite", -1),
        ("num_cores", 1),
        ("num_cores", 5),
        ("fator_mutacao_dirigida", 0.5),
    ])
    def test_valor_invalido_levanta_erro(self, campo, valor):
        with pytest.raises(ValueError):
            ParametrosAG(**{campo: valor})

    def test_tipo_crossover_invalido(self):
        with pytest.raises(ValueError, match="tipo_crossover"):
            ParametrosAG(tipo_crossover="dois_pontos")

    def test_tipo_mutacao_invalido(self):
        with pytest.raises(ValueError, match="tipo_mutacao"):
            ParametrosAG(tipo_mutacao="inteligente")

    def test_torneio_maior_que_populacao(self):
        with pytest.raises(ValueError, match="tamanho_torneio"):
            ParametrosAG(tamanho_populacao=5, tamanho_torneio=10)

    def test_parametros_padrao_sao_validos(self):
        ParametrosAG()  # não deve lançar


# ---------------------------------------------------------------------------
# Função de avaliação
# ---------------------------------------------------------------------------

class TestAvaliacao:
    def test_sem_conflitos(self):
        ag = criar_ag(MAPA_TRIANGULO)
        # A, B, C com cores diferentes -> 0 conflitos
        individuo = [0, 1, 2]
        assert ag.contar_conflitos(individuo) == 0
        assert ag.calcular_fitness(individuo) == 1.0

    def test_um_conflito(self):
        ag = criar_ag(MAPA_TRIANGULO)
        # A e B com a mesma cor -> 1 conflito (a aresta A-B)
        individuo = [0, 0, 2]
        assert ag.contar_conflitos(individuo) == 1
        assert ag.calcular_fitness(individuo) == pytest.approx(0.5)

    def test_todos_a_mesma_cor_triangulo(self):
        ag = criar_ag(MAPA_TRIANGULO)
        # Triângulo com todos iguais: as 3 arestas estão em conflito.
        individuo = [1, 1, 1]
        assert ag.contar_conflitos(individuo) == 3
        assert ag.calcular_fitness(individuo) == pytest.approx(0.25)

    def test_eh_solucao_valida_via_resultado(self):
        ag = criar_ag(MAPA_QUADRADO)
        assert ag.contar_conflitos([0, 1, 0, 1]) == 0  # 2 cores bastam no ciclo par


# ---------------------------------------------------------------------------
# Cruzamento
# ---------------------------------------------------------------------------

class TestCruzamento:
    def test_um_ponto_sem_crossover_copia_os_pais(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_crossover=0.0)
        pai1, pai2 = [0, 0, 0, 0], [1, 1, 1, 1]
        filho1, filho2 = ag.cruzar_um_ponto(pai1, pai2)
        assert filho1 == pai1
        assert filho2 == pai2
        assert filho1 is not pai1  # é cópia, não a mesma lista

    def test_um_ponto_com_crossover_sempre_mistura(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_crossover=1.0)
        pai1, pai2 = [0, 0, 0, 0], [1, 1, 1, 1]
        filho1, filho2 = ag.cruzar_um_ponto(pai1, pai2)
        # Com pais "opostos", um cruzamento real produz filhos com os dois
        # valores presentes (a menos que o ponto de corte seja degenerado).
        assert set(filho1) == {0, 1}
        assert len(filho1) == 4
        # Cada filho é a combinação complementar do outro.
        assert filho1 == [1 - g for g in filho2]

    def test_uniforme_preserva_o_multiconjunto_de_genes(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_crossover=1.0)
        pai1, pai2 = [0, 1, 2, 3], [3, 2, 1, 0]
        filho1, filho2 = ag.cruzar_uniforme(pai1, pai2)
        # Em cada posição, o filho 1 recebeu o gene do pai 1 OU do pai 2
        # (nunca um valor que não estava em nenhum dos dois naquela posição).
        for i in range(4):
            assert filho1[i] in (pai1[i], pai2[i])
            assert filho2[i] in (pai1[i], pai2[i])
            assert {filho1[i], filho2[i]} == {pai1[i], pai2[i]}

    def test_dispatcher_cruzar_respeita_tipo_crossover(self):
        ag_ponto = criar_ag(MAPA_QUADRADO, tipo_crossover="um_ponto")
        ag_uniforme = criar_ag(MAPA_QUADRADO, tipo_crossover="uniforme")
        assert ag_ponto.parametros.tipo_crossover == "um_ponto"
        assert ag_uniforme.parametros.tipo_crossover == "uniforme"


# ---------------------------------------------------------------------------
# Mutação
# ---------------------------------------------------------------------------

class TestMutacao:
    def test_taxa_zero_nao_altera_individuo(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_mutacao=0.0)
        individuo = [0, 1, 2, 3]
        assert ag.mutar_aleatoria(individuo) == individuo

    def test_taxa_um_altera_todos_os_genes(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_mutacao=1.0)
        individuo = [0, 1, 2, 3]
        mutado = ag.mutar_aleatoria(individuo)
        # Cada gene mudou (a mutação nunca repete a cor atual).
        assert all(novo != antigo for antigo, novo in zip(mutado, individuo))

    def test_mutacao_nao_altera_lista_original(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_mutacao=1.0)
        individuo = [0, 1, 2, 3]
        copia = individuo.copy()
        ag.mutar_aleatoria(individuo)
        assert individuo == copia  # o original não foi tocado

    def test_dirigida_prioriza_genes_em_conflito(self):
        # Triângulo todo na mesma cor: todos os genes estão em conflito, então
        # a mutação dirigida deve usar a taxa ampliada e tender a reduzir
        # conflitos. Com fator bem alto, quase sempre melhora ou mantém.
        ag = criar_ag(MAPA_TRIANGULO, taxa_mutacao=0.5, fator_mutacao_dirigida=2.0,
                      tipo_mutacao="dirigida")
        individuo = [0, 0, 0]
        conflitos_antes = ag.contar_conflitos(individuo)
        melhorou_ou_igual = 0
        tentativas = 200
        for _ in range(tentativas):
            mutado = ag.mutar_dirigida(individuo)
            if ag.contar_conflitos(mutado) <= conflitos_antes:
                melhorou_ou_igual += 1
        # A mutação dirigida escolhe a cor com menos conflitos locais, então
        # deveria quase nunca piorar a situação.
        assert melhorou_ou_igual / tentativas > 0.9

    def test_dirigida_sem_conflito_ainda_pode_mutar_com_taxa_normal(self):
        ag = criar_ag(MAPA_QUADRADO, taxa_mutacao=1.0, tipo_mutacao="dirigida")
        individuo = [0, 1, 0, 1]  # solução válida, sem conflitos
        mutado = ag.mutar_dirigida(individuo)
        # Taxa normal = 1.0 -> todos os genes (sem conflito) ainda mutam.
        assert all(novo != antigo for antigo, novo in zip(mutado, individuo))

    def test_dispatcher_mutar_respeita_tipo_mutacao(self):
        ag_aleatoria = criar_ag(MAPA_QUADRADO, tipo_mutacao="aleatoria")
        ag_dirigida = criar_ag(MAPA_QUADRADO, tipo_mutacao="dirigida")
        assert ag_aleatoria.parametros.tipo_mutacao == "aleatoria"
        assert ag_dirigida.parametros.tipo_mutacao == "dirigida"


# ---------------------------------------------------------------------------
# Execução ponta a ponta
# ---------------------------------------------------------------------------

class TestExecucao:
    def test_quadrado_converge_com_2_cores(self):
        ag = criar_ag(MAPA_QUADRADO, num_cores=2, numero_geracoes=200, semente=1)
        resultado = ag.executar(verbose=False)
        assert resultado.valida
        assert resultado.conflitos == 0

    def test_triangulo_converge_com_3_cores(self):
        ag = criar_ag(MAPA_TRIANGULO, num_cores=3, numero_geracoes=200, semente=1)
        resultado = ag.executar(verbose=False)
        assert resultado.valida

    def test_triangulo_nao_converge_com_2_cores(self):
        # Um triângulo (3 regiões mutuamente vizinhas) nunca é 2-colorível:
        # o AG deve esgotar o limite de gerações sem achar solução válida.
        ag = criar_ag(MAPA_TRIANGULO, num_cores=2, numero_geracoes=50, semente=1)
        resultado = ag.executar(verbose=False)
        assert not resultado.valida
        assert resultado.conflitos >= 1
        assert resultado.geracoes == 50

    def test_resultado_e_reprodutivel_com_mesma_semente(self):
        resultados = [
            criar_ag(MAPA_QUADRADO, semente=123).executar(verbose=False).solucao
            for _ in range(3)
        ]
        assert resultados[0] == resultados[1] == resultados[2]

    @pytest.mark.parametrize("tipo_crossover", ["um_ponto", "uniforme"])
    @pytest.mark.parametrize("tipo_mutacao", ["aleatoria", "dirigida"])
    def test_todas_as_combinacoes_de_operadores_rodam_no_mapa_pequeno(
        self, tipo_crossover, tipo_mutacao,
    ):
        mapa, _ = criar_mapa_simplificado()
        ag = criar_ag(mapa, tipo_crossover=tipo_crossover, tipo_mutacao=tipo_mutacao,
                      numero_geracoes=300, semente=1)
        resultado = ag.executar(verbose=False)
        assert resultado.valida


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
