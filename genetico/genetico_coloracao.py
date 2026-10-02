"""
Algoritmo Genético para o Problema de Coloração de Mapas
=========================================================

Objetivo: atribuir uma cor a cada região de um mapa de forma que nenhuma
par de regiões vizinhas tenha a mesma cor, usando no máximo 4 cores
(Teorema das Quatro Cores).

Resumo da solução genética (itens exigidos na apresentação)
-----------------------------------------------------------
- Codificação (alfabeto genético): inteiros {0, 1, 2, 3}; cada valor é uma
  cor (0 = Vermelho, 1 = Azul, 2 = Verde, 3 = Amarelo).
- Cromossomo: um gene por região do mapa (27 genes no mapa do Brasil,
  9 no mapa simplificado). O gene i guarda a cor da região i.
- Função de avaliação: fitness = 1 / (1 + conflitos), em que um conflito é
  um par de regiões vizinhas com a mesma cor. Fitness 1.0 = solução válida.
- Seleção: torneio de tamanho k (padrão k = 3).
- Cruzamento: um ponto OU uniforme (escolha via --crossover), taxa de 85%.
- Mutação: aleatória OU dirigida a conflitos (escolha via --mutacao); taxa
  de 2% por gene (a dirigida usa uma taxa maior apenas nos genes em conflito).
- Substituição: geracional com elitismo (os 2 melhores da geração anterior
  entram no lugar dos 2 piores filhos).
- Parâmetros: população de 100 indivíduos, população inicial aleatória
  uniforme; parada ao encontrar uma solução sem conflitos ou após
  1000 gerações.

Uso
---
    python genetico_coloracao.py                              # pergunta qual mapa usar
    python genetico_coloracao.py --mapa brasil                # executa direto no mapa do Brasil
    python genetico_coloracao.py --mapa simplificado --semente 42       # resultado reprodutível
    python genetico_coloracao.py --mapa brasil --crossover uniforme     # cruzamento uniforme
    python genetico_coloracao.py --mapa brasil --mutacao dirigida       # mutação dirigida a conflitos
    python genetico_coloracao.py --mapa brasil --cores 3                # testa com apenas 3 cores

Para o experimento comparativo (várias sementes x parâmetros), veja
`genetico-coloracao-experimento-comparativo.py`. Para os testes automatizados, veja
`genetico-coloracao-teste.py` (rodar com `pytest genetico-coloracao-teste.py`).
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# TIPOS E CONSTANTES
# =============================================================================

# Apelidos de tipo: deixam as assinaturas das funções mais legíveis.
Cromossomo = list[int]                       # lista de cores, uma por região
Mapa = dict[str, list[str]]                  # região -> lista de regiões vizinhas
Coordenadas = dict[str, tuple[float, float]]  # região -> posição (x, y) no gráfico

# Paleta usada na visualização. A ordem precisa casar com o alfabeto genético:
# o gene de valor 0 é desenhado com CORES_HEX[0] e chamado NOMES_CORES[0], etc.
# As cores foram escolhidas para corresponder de fato aos nomes e serem bem
# distinguíveis entre si (inclusive em projetor/vídeo).
CORES_HEX = ("#E63946", "#1D70B8", "#2A9D8F", "#F4C430")
NOMES_CORES = ("Vermelho", "Azul", "Verde", "Amarelo")

# Pasta onde as imagens de resultado são salvas (a mesma deste script).
PASTA_SAIDA = Path(__file__).resolve().parent


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass(frozen=True)
class ParametrosAG:
    """Parâmetros genéticos do algoritmo.

    `frozen=True` torna o objeto imutável: os parâmetros não podem ser
    alterados por engano durante a execução.
    """

    tamanho_populacao: int = 100   # indivíduos por geração
    numero_geracoes: int = 1000    # limite de gerações (condição de parada)
    taxa_crossover: float = 0.85   # probabilidade de um par de pais cruzar
    taxa_mutacao: float = 0.02     # probabilidade de CADA gene sofrer mutação
    tamanho_torneio: int = 3       # competidores sorteados em cada torneio
    numero_elite: int = 2          # melhores preservados; 0 desativa o elitismo
    num_cores: int = 4             # tamanho do alfabeto genético
    intervalo_log: int = 100       # de quantas em quantas gerações imprimir progresso
    semente: int | None = None     # semente aleatória (None = execução não reprodutível)

    # Operadores selecionáveis (ver AlgoritmoGeneticoColoracao.cruzar / .mutar).
    tipo_crossover: str = "um_ponto"   # "um_ponto" ou "uniforme"
    tipo_mutacao: str = "aleatoria"    # "aleatoria" ou "dirigida"
    fator_mutacao_dirigida: float = 8.0
    # Na mutação dirigida, um gene em conflito usa taxa_mutacao * fator_mutacao_dirigida
    # (limitada a 1.0) em vez de taxa_mutacao. Só tem efeito com tipo_mutacao="dirigida".

    def __post_init__(self) -> None:
        """Valida os parâmetros logo na criação (falha cedo, com mensagem clara)."""
        if self.tamanho_populacao < 2:
            raise ValueError("tamanho_populacao deve ser >= 2.")
        if self.numero_geracoes < 1:
            raise ValueError("numero_geracoes deve ser >= 1.")
        for nome in ("taxa_crossover", "taxa_mutacao"):
            if not 0.0 <= getattr(self, nome) <= 1.0:
                raise ValueError(f"{nome} deve estar entre 0 e 1.")
        if not 2 <= self.tamanho_torneio <= self.tamanho_populacao:
            raise ValueError("tamanho_torneio deve estar entre 2 e tamanho_populacao.")
        if not 0 <= self.numero_elite < self.tamanho_populacao:
            raise ValueError("numero_elite deve estar entre 0 e tamanho_populacao - 1.")
        if not 2 <= self.num_cores <= len(NOMES_CORES):
            raise ValueError(f"num_cores deve estar entre 2 e {len(NOMES_CORES)}.")
        if self.tipo_crossover not in ("um_ponto", "uniforme"):
            raise ValueError("tipo_crossover deve ser 'um_ponto' ou 'uniforme'.")
        if self.tipo_mutacao not in ("aleatoria", "dirigida"):
            raise ValueError("tipo_mutacao deve ser 'aleatoria' ou 'dirigida'.")
        if self.fator_mutacao_dirigida < 1.0:
            raise ValueError("fator_mutacao_dirigida deve ser >= 1.0.")


@dataclass
class Estatisticas:
    """Histórico por geração, usado nos gráficos de evolução."""

    melhor_fitness: list[float] = field(default_factory=list)
    fitness_medio: list[float] = field(default_factory=list)
    pior_fitness: list[float] = field(default_factory=list)
    conflitos_melhor: list[int] = field(default_factory=list)


@dataclass
class Resultado:
    """Resultado final de uma execução do algoritmo."""

    solucao: Cromossomo
    fitness: float
    conflitos: int
    geracoes: int            # quantas gerações foram produzidas após a inicial
    tempo_execucao: float    # em segundos
    estatisticas: Estatisticas

    @property
    def valida(self) -> bool:
        """Uma solução é válida quando não há nenhum conflito."""
        return self.conflitos == 0


# =============================================================================
# VALIDAÇÃO DO MAPA
# =============================================================================

def validar_mapa(mapa: Mapa) -> None:
    """Confere se o mapa é consistente antes de rodar o algoritmo.

    Regras verificadas:
      * todo vizinho citado precisa existir como região do mapa;
      * uma região não pode ser vizinha de si mesma;
      * a vizinhança precisa ser simétrica (se A faz fronteira com B,
        B também precisa listar A).

    Erros de digitação nas adjacências mudam o problema silenciosamente,
    então é melhor interromper com uma mensagem clara.
    """
    if len(mapa) < 2:
        raise ValueError("O mapa precisa ter pelo menos 2 regiões.")

    for regiao, vizinhos in mapa.items():
        for vizinho in vizinhos:
            if vizinho not in mapa:
                raise ValueError(f"'{vizinho}' (vizinho de '{regiao}') não existe no mapa.")
            if vizinho == regiao:
                raise ValueError(f"'{regiao}' está listada como vizinha de si mesma.")
            if regiao not in mapa[vizinho]:
                raise ValueError(
                    f"Adjacência assimétrica: '{regiao}' lista '{vizinho}', "
                    f"mas '{vizinho}' não lista '{regiao}'."
                )


# =============================================================================
# ALGORITMO GENÉTICO
# =============================================================================

class AlgoritmoGeneticoColoracao:
    """Algoritmo Genético para o problema de coloração de mapas."""

    def __init__(self, mapa: Mapa, parametros: ParametrosAG | None = None) -> None:
        validar_mapa(mapa)

        self.parametros = parametros or ParametrosAG()
        self.regioes = list(mapa)          # ordem das regiões = ordem dos genes
        self.num_genes = len(self.regioes)

        # Gerador aleatório próprio: com uma semente fixa, a execução inteira
        # é reprodutível (útil para gravar o vídeo e repetir o resultado).
        self.rng = random.Random(self.parametros.semente)

        # Alfabeto genético: 0, 1, ..., num_cores - 1.
        self.alfabeto = list(range(self.parametros.num_cores))

        # Converte as adjacências em uma lista de ARESTAS ÚNICAS (i, j) com i < j.
        # Assim cada fronteira é verificada uma única vez na função de avaliação,
        # sem precisar contar duas vezes e dividir por 2.
        indice = {regiao: i for i, regiao in enumerate(self.regioes)}
        self.arestas: list[tuple[int, int]] = sorted(
            {
                (min(indice[a], indice[b]), max(indice[a], indice[b]))
                for a, vizinhos in mapa.items()
                for b in vizinhos
            }
        )

        # Lista de adjacência por índice (quem é vizinho de quem). Usada pela
        # mutação dirigida para localizar, em O(1) por gene, quais vizinhos
        # têm a mesma cor.
        self.adjacencia: list[list[int]] = [[] for _ in range(self.num_genes)]
        for a, b in self.arestas:
            self.adjacencia[a].append(b)
            self.adjacencia[b].append(a)

    # ------------------------------------------------------------------
    # Representação e população inicial
    # ------------------------------------------------------------------

    def criar_individuo(self) -> Cromossomo:
        """Cria um cromossomo aleatório: uma cor sorteada para cada região."""
        return [self.rng.choice(self.alfabeto) for _ in range(self.num_genes)]

    def criar_populacao_inicial(self) -> list[Cromossomo]:
        """Gera a população inicial de forma totalmente aleatória (uniforme)."""
        return [self.criar_individuo() for _ in range(self.parametros.tamanho_populacao)]

    # ------------------------------------------------------------------
    # Função de avaliação
    # ------------------------------------------------------------------

    def contar_conflitos(self, individuo: Cromossomo) -> int:
        """Conta as fronteiras cujas duas regiões receberam a mesma cor."""
        return sum(individuo[i] == individuo[j] for i, j in self.arestas)

    def calcular_fitness(self, individuo: Cromossomo) -> float:
        """Fitness = 1 / (1 + conflitos).

        * 0 conflitos  -> fitness 1.0 (solução ótima);
        * mais conflitos -> fitness tende a 0.
        O "+1" evita divisão por zero e mantém o valor em (0, 1].
        """
        return 1.0 / (1.0 + self.contar_conflitos(individuo))

    def avaliar_populacao(self, populacao: list[Cromossomo]) -> list[float]:
        """Calcula o fitness de todos os indivíduos UMA única vez por geração.

        Os valores são reaproveitados na seleção, no elitismo e nas
        estatísticas, evitando recalcular o fitness dezenas de vezes.
        """
        return [self.calcular_fitness(ind) for ind in populacao]

    # ------------------------------------------------------------------
    # Operadores genéticos
    # ------------------------------------------------------------------

    def selecionar_por_torneio(self, populacao: list[Cromossomo],
                               fitness: list[float]) -> Cromossomo:
        """Seleção por torneio.

        Sorteia `tamanho_torneio` indivíduos distintos e devolve o de maior
        fitness. Torneios maiores aumentam a pressão seletiva (convergem mais
        rápido, porém com maior risco de convergência prematura).
        """
        competidores = self.rng.sample(range(len(populacao)), self.parametros.tamanho_torneio)
        vencedor = max(competidores, key=lambda idx: fitness[idx])
        return populacao[vencedor]

    def cruzar_um_ponto(self, pai1: Cromossomo,
                        pai2: Cromossomo) -> tuple[Cromossomo, Cromossomo]:
        """Cruzamento de um ponto, aplicado com probabilidade `taxa_crossover`.

        Exemplo com ponto de corte 3:
            pai1 = [0 1 2 | 3 0 1]      filho1 = [0 1 2 | 2 3 0]
            pai2 = [1 3 0 | 2 3 0]  ->  filho2 = [1 3 0 | 3 0 1]

        Se o cruzamento não ocorrer, os filhos são cópias dos pais. Como a
        ordem dos genes é arbitrária (ordem das regiões no dicionário do
        mapa), o ponto de corte não separa "blocos" de regiões vizinhas com
        significado geográfico — ver `cruzar_uniforme` para uma alternativa
        que não depende dessa ordem.
        """
        if self.rng.random() < self.parametros.taxa_crossover:
            ponto = self.rng.randint(1, self.num_genes - 1)  # nunca corta nas pontas
            return pai1[:ponto] + pai2[ponto:], pai2[:ponto] + pai1[ponto:]
        return pai1.copy(), pai2.copy()

    def cruzar_uniforme(self, pai1: Cromossomo,
                        pai2: Cromossomo) -> tuple[Cromossomo, Cromossomo]:
        """Cruzamento uniforme, aplicado com probabilidade `taxa_crossover`.

        Em vez de um único ponto de corte, CADA gene é sorteado
        independentemente: com 50% de chance o filho 1 herda o gene do pai 1
        (e o filho 2 herda do pai 2); caso contrário, o inverso. Isso evita
        o viés posicional do cruzamento de um ponto — não importa em que
        ordem as regiões aparecem no cromossomo, qualquer combinação de
        genes dos dois pais é igualmente alcançável.
        """
        if self.rng.random() < self.parametros.taxa_crossover:
            filho1, filho2 = [], []
            for g1, g2 in zip(pai1, pai2):
                if self.rng.random() < 0.5:
                    filho1.append(g1)
                    filho2.append(g2)
                else:
                    filho1.append(g2)
                    filho2.append(g1)
            return filho1, filho2
        return pai1.copy(), pai2.copy()

    def cruzar(self, pai1: Cromossomo, pai2: Cromossomo) -> tuple[Cromossomo, Cromossomo]:
        """Despacha para o operador de cruzamento escolhido em `tipo_crossover`."""
        if self.parametros.tipo_crossover == "uniforme":
            return self.cruzar_uniforme(pai1, pai2)
        return self.cruzar_um_ponto(pai1, pai2)

    def mutar_aleatoria(self, individuo: Cromossomo) -> Cromossomo:
        """Mutação por reinicialização aleatória (operador original).

        Cada gene, de forma independente, tem probabilidade `taxa_mutacao` de
        ser trocado por uma cor DIFERENTE da atual (garante que a mutação
        realmente altera o gene). Com 27 genes e taxa 0,02, ocorre em média
        ~0,5 mutação por indivíduo.
        """
        mutado = individuo.copy()  # não altera o indivíduo original
        for i, cor_atual in enumerate(mutado):
            if self.rng.random() < self.parametros.taxa_mutacao:
                outras_cores = [cor for cor in self.alfabeto if cor != cor_atual]
                mutado[i] = self.rng.choice(outras_cores)
        return mutado

    def mutar_dirigida(self, individuo: Cromossomo) -> Cromossomo:
        """Mutação dirigida a conflitos.

        Ideia: um gene que já está em conflito com algum vizinho é um alvo
        muito melhor para mutação do que um gene sorteado ao acaso — mutar
        genes sem conflito só adiciona ruído. Por isso:

        * genes SEM conflito sofrem mutação aleatória com a taxa normal
          (`taxa_mutacao`), como no operador original — mantém diversidade;
        * genes EM conflito sofrem mutação com taxa ampliada
          (`taxa_mutacao * fator_mutacao_dirigida`, limitada a 100%) e, ao
          mutar, a nova cor é escolhida de forma gulosa: entre as cores que
          NÃO são a atual, prefere a que aparece menos vezes entre os
          vizinhos daquele gene (reduz o número de conflitos locais em vez
          de apenas trocar de cor às cegas).

        Isso é uma heurística de reparo local inserida dentro do operador de
        mutação — útil porque reduz bastante o número de gerações em mapas
        com poucos conflitos residuais (ver `experimento_comparativo.py`).
        """
        p = self.parametros
        mutado = individuo.copy()
        taxa_normal = p.taxa_mutacao
        taxa_conflito = min(1.0, p.taxa_mutacao * p.fator_mutacao_dirigida)

        for i in range(self.num_genes):
            vizinhos = self.adjacencia[i]
            cor_atual = mutado[i]
            em_conflito = any(mutado[v] == cor_atual for v in vizinhos)
            taxa = taxa_conflito if em_conflito else taxa_normal

            if self.rng.random() >= taxa:
                continue

            if em_conflito and vizinhos:
                # Conta quantas vezes cada cor aparece entre os vizinhos
                # atuais e escolhe (entre as cores diferentes da atual) uma
                # das que geram o menor número de conflitos.
                contagem = [0] * len(self.alfabeto)
                for v in vizinhos:
                    contagem[mutado[v]] += 1
                minimo = min(contagem[c] for c in self.alfabeto if c != cor_atual)
                candidatas = [c for c in self.alfabeto
                             if c != cor_atual and contagem[c] == minimo]
                mutado[i] = self.rng.choice(candidatas)
            else:
                outras_cores = [cor for cor in self.alfabeto if cor != cor_atual]
                mutado[i] = self.rng.choice(outras_cores)

        return mutado

    def mutar(self, individuo: Cromossomo) -> Cromossomo:
        """Despacha para o operador de mutação escolhido em `tipo_mutacao`."""
        if self.parametros.tipo_mutacao == "dirigida":
            return self.mutar_dirigida(individuo)
        return self.mutar_aleatoria(individuo)

    def gerar_descendentes(self, populacao: list[Cromossomo],
                           fitness: list[float]) -> list[Cromossomo]:
        """Produz uma nova população do mesmo tamanho: seleção -> cruzamento -> mutação."""
        tamanho = self.parametros.tamanho_populacao
        filhos: list[Cromossomo] = []

        while len(filhos) < tamanho:
            pai1 = self.selecionar_por_torneio(populacao, fitness)
            pai2 = self.selecionar_por_torneio(populacao, fitness)
            filho1, filho2 = self.cruzar(pai1, pai2)

            filhos.append(self.mutar(filho1))
            if len(filhos) < tamanho:  # trata populações de tamanho ímpar
                filhos.append(self.mutar(filho2))

        return filhos

    def substituir_com_elitismo(self, populacao: list[Cromossomo], fitness: list[float],
                                filhos: list[Cromossomo],
                                fitness_filhos: list[float]) -> tuple[list[Cromossomo], list[float]]:
        """Estratégia de substituição: geracional com elitismo.

        Os filhos substituem toda a geração anterior, exceto pelos
        `numero_elite` melhores pais, que entram no lugar dos PIORES filhos.
        Isso garante que o melhor fitness nunca piora de uma geração para outra.
        """
        n_elite = self.parametros.numero_elite
        if n_elite == 0:
            return filhos, fitness_filhos

        melhores_pais = sorted(range(len(populacao)), key=lambda i: fitness[i], reverse=True)[:n_elite]
        piores_filhos = sorted(range(len(filhos)), key=lambda i: fitness_filhos[i])[:n_elite]

        nova_populacao = filhos.copy()
        novo_fitness = fitness_filhos.copy()
        for idx_pai, idx_filho in zip(melhores_pais, piores_filhos):
            nova_populacao[idx_filho] = populacao[idx_pai]
            novo_fitness[idx_filho] = fitness[idx_pai]

        return nova_populacao, novo_fitness

    # ------------------------------------------------------------------
    # Laço principal
    # ------------------------------------------------------------------

    def executar(self, verbose: bool = True) -> Resultado:
        """Executa o AG até achar uma solução válida ou atingir o limite de gerações.

        `verbose=False` desliga os prints de progresso (cabeçalho, log
        periódico e mensagem final) — usado pelo experimento comparativo,
        que roda dezenas de execuções e só precisa do `Resultado` de volta.
        """
        p = self.parametros
        estatisticas = Estatisticas()  # novo a cada execução (não acumula entre execuções)
        inicio = time.perf_counter()   # relógio de alta precisão para medir tempo

        if verbose:
            self._imprimir_cabecalho()

        populacao = self.criar_populacao_inicial()
        fitness = self.avaliar_populacao(populacao)

        melhor_global: Cromossomo = []
        fitness_global = -1.0
        geracao = 0  # geração 0 = população inicial

        while True:
            # 1) Melhor indivíduo da geração atual e registro das estatísticas.
            idx_melhor = max(range(len(populacao)), key=lambda i: fitness[i])
            if fitness[idx_melhor] > fitness_global:
                melhor_global, fitness_global = populacao[idx_melhor], fitness[idx_melhor]

            conflitos_melhor = self.contar_conflitos(melhor_global)
            estatisticas.melhor_fitness.append(fitness_global)
            estatisticas.fitness_medio.append(statistics.fmean(fitness))
            estatisticas.pior_fitness.append(min(fitness))
            estatisticas.conflitos_melhor.append(conflitos_melhor)

            # 2) Condição de parada: solução sem conflitos OU limite de gerações.
            if conflitos_melhor == 0:
                if verbose:
                    print(f"✓ Solução ótima encontrada na geração {geracao}!")
                break
            if geracao == p.numero_geracoes:
                if verbose:
                    print(f"✗ Limite de {p.numero_geracoes} gerações atingido sem solução ótima.")
                break

            # 3) Log periódico do progresso.
            if verbose and geracao % p.intervalo_log == 0:
                print(f"Geração {geracao:4d}: fitness = {fitness_global:.4f}, "
                      f"conflitos = {conflitos_melhor}")

            # 4) Reprodução e substituição.
            filhos = self.gerar_descendentes(populacao, fitness)
            fitness_filhos = self.avaliar_populacao(filhos)
            populacao, fitness = self.substituir_com_elitismo(populacao, fitness,
                                                              filhos, fitness_filhos)
            geracao += 1

        return Resultado(
            solucao=melhor_global,
            fitness=fitness_global,
            conflitos=self.contar_conflitos(melhor_global),
            geracoes=geracao,
            tempo_execucao=time.perf_counter() - inicio,
            estatisticas=estatisticas,
        )

    def _imprimir_cabecalho(self) -> None:
        """Mostra no console o problema e os parâmetros usados."""
        p = self.parametros
        print("=== ALGORITMO GENÉTICO PARA COLORAÇÃO DE MAPAS ===")
        print(f"Regiões (genes): {self.num_genes} | Fronteiras: {len(self.arestas)} | "
              f"Cores disponíveis: {p.num_cores}")
        print(f"População = {p.tamanho_populacao} | Gerações máx. = {p.numero_geracoes} | "
              f"Torneio = {p.tamanho_torneio} | Elite = {p.numero_elite} | Semente = {p.semente}")
        print(f"Crossover = {p.tipo_crossover} (taxa {p.taxa_crossover}) | "
              f"Mutação = {p.tipo_mutacao} (taxa {p.taxa_mutacao}"
              + (f", fator conflito {p.fator_mutacao_dirigida}x)" if p.tipo_mutacao == "dirigida" else ")"))
        print("-" * 70)


# =============================================================================
# MAPAS DE EXEMPLO (adjacências + coordenadas para desenho)
# =============================================================================

def criar_mapa_brasil() -> tuple[Mapa, Coordenadas]:
    """Os 27 estados brasileiros e suas fronteiras terrestres.

    As coordenadas são (longitude, latitude) aproximadas do centro de cada
    estado, servindo apenas para desenhar o grafo com aparência de mapa.
    Os estados pequenos do Nordeste foram levemente afastados para que os
    círculos não se sobreponham no gráfico.
    """
    mapa = {
        "AC": ["AM", "RO"],
        "AL": ["SE", "BA", "PE"],
        "AM": ["AC", "RO", "MT", "PA", "RR"],
        "AP": ["PA"],
        "BA": ["SE", "AL", "PE", "PI", "TO", "GO", "MG", "ES"],
        "CE": ["RN", "PB", "PE", "PI"],
        "DF": ["GO", "MG"],
        "ES": ["BA", "MG", "RJ"],
        "GO": ["BA", "TO", "MT", "MS", "MG", "DF"],
        "MA": ["PI", "TO", "PA"],
        "MG": ["BA", "GO", "DF", "MS", "SP", "RJ", "ES"],
        "MS": ["MT", "GO", "MG", "SP", "PR"],
        "MT": ["AM", "PA", "TO", "GO", "MS", "RO"],   
        "PA": ["AM", "MT", "TO", "MA", "AP", "RR"],
        "PB": ["RN", "CE", "PE"],
        "PE": ["PB", "CE", "PI", "BA", "AL"],
        "PI": ["CE", "PE", "BA", "TO", "MA"],
        "PR": ["MS", "SP", "SC"],
        "RJ": ["ES", "MG", "SP"],
        "RN": ["CE", "PB"],
        "RO": ["AC", "AM", "MT"],                     
        "RR": ["AM", "PA"],
        "RS": ["SC"],
        "SC": ["RS", "PR"],
        "SE": ["AL", "BA"],
        "SP": ["MG", "RJ", "MS", "PR"],
        "TO": ["MA", "PI", "BA", "GO", "MT", "PA"],
    }

    coordenadas = {
        "AC": (-70.5, -9.0), "AL": (-34.6, -10.0), "AM": (-64.5, -4.0), "AP": (-51.5, 1.4),
        "BA": (-41.7, -12.5), "CE": (-39.8, -4.6), "DF": (-47.8, -15.8), "ES": (-40.6, -19.6),
        "GO": (-49.8, -16.3), "MA": (-45.3, -5.0), "MG": (-44.6, -18.5), "MS": (-54.8, -20.5),
        "MT": (-55.9, -12.9), "PA": (-52.5, -4.0), "PB": (-34.0, -7.3), "PE": (-37.6, -8.6),
        "PI": (-42.8, -7.7), "PR": (-51.6, -24.6), "RJ": (-42.7, -22.3), "RN": (-35.8, -4.6),
        "RO": (-62.8, -10.9), "RR": (-61.4, 2.0), "RS": (-53.2, -29.7), "SC": (-50.4, -27.3),
        "SE": (-36.6, -11.8), "SP": (-48.6, -22.2), "TO": (-48.3, -10.2),
    }
    return mapa, coordenadas


def criar_mapa_simplificado() -> tuple[Mapa, Coordenadas]:
    """Mapa didático com 9 regiões dispostas em uma grade 3x3."""
    mapa = {
        "A": ["B", "C", "D"],
        "B": ["A", "C", "E"],
        "C": ["A", "B", "D", "E", "F"],
        "D": ["A", "C", "F", "G"],
        "E": ["B", "C", "F", "H"],
        "F": ["C", "D", "E", "G", "H", "I"],
        "G": ["D", "F", "I"],
        "H": ["E", "F", "I"],
        "I": ["F", "G", "H"],
    }
    coordenadas = {
        "A": (1, 3), "B": (2, 3), "C": (3, 3),
        "D": (1, 2), "E": (2, 2), "F": (3, 2),
        "G": (1, 1), "H": (2, 1), "I": (3, 1),
    }
    return mapa, coordenadas


# Catálogo de mapas disponíveis: chave -> (função que cria o mapa, título).
# Para adicionar um mapa novo basta incluir uma entrada aqui.
MAPAS: dict[str, tuple[Callable[[], tuple[Mapa, Coordenadas]], str]] = {
    "brasil": (criar_mapa_brasil, "Mapa do Brasil - 27 Estados"),
    "simplificado": (criar_mapa_simplificado, "Mapa Simplificado - 9 Regiões"),
}


# =============================================================================
# VISUALIZAÇÃO E RELATÓRIO
# =============================================================================

def _desenhar_mapa(ax, mapa: Mapa, coordenadas: Coordenadas,
                   solucao: Cromossomo, titulo: str) -> None:
    """Subplot 1: grafo do mapa com cada região pintada com sua cor."""
    regioes = list(mapa)
    cor_da_regiao = dict(zip(regioes, solucao))

    # Fronteiras: desenha cada uma uma única vez. Fronteiras em conflito
    # (mesma cor nas duas pontas) aparecem em vermelho tracejado.
    desenhadas: set[frozenset[str]] = set()
    for regiao, vizinhos in mapa.items():
        for vizinho in vizinhos:
            par = frozenset((regiao, vizinho))
            if par in desenhadas:
                continue
            desenhadas.add(par)
            (x1, y1), (x2, y2) = coordenadas[regiao], coordenadas[vizinho]
            em_conflito = cor_da_regiao[regiao] == cor_da_regiao[vizinho]
            ax.plot([x1, x2], [y1, y2],
                    color="red" if em_conflito else "gray",
                    linestyle="--" if em_conflito else "-",
                    linewidth=2 if em_conflito else 1,
                    alpha=0.9 if em_conflito else 0.5, zorder=1)

    # Regiões: círculos coloridos com a sigla no centro.
    for regiao, (x, y) in coordenadas.items():
        ax.scatter(x, y, s=500, c=CORES_HEX[cor_da_regiao[regiao]],
                   edgecolors="black", linewidths=1.5, zorder=3)
        ax.text(x, y, regiao, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4)

    # Legenda com apenas as cores realmente usadas.
    cores_usadas = sorted(set(solucao))
    ax.legend(handles=[Patch(color=CORES_HEX[c], label=NOMES_CORES[c]) for c in cores_usadas],
              loc="lower left", fontsize=8)

    ax.set_title(f"MAPA COLORIDO - {titulo}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (aprox.)")
    ax.set_ylabel("Latitude (aprox.)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.margins(0.1)


def _desenhar_fitness(ax, estatisticas: Estatisticas) -> None:
    """Subplot 2: evolução do melhor, médio e pior fitness por geração."""
    ax.plot(estatisticas.melhor_fitness, "b-", linewidth=2, label="Melhor")
    ax.plot(estatisticas.fitness_medio, "g-", linewidth=2, alpha=0.7, label="Médio")
    ax.plot(estatisticas.pior_fitness, "r-", linewidth=2, alpha=0.7, label="Pior")
    ax.set_title("EVOLUÇÃO DO FITNESS", fontsize=10, fontweight="bold")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_ylim(0, 1.05)
    ax.xaxis.get_major_locator().set_params(integer=True)  # gerações são inteiras
    ax.legend()
    ax.grid(True, alpha=0.3)


def _desenhar_conflitos(ax, estatisticas: Estatisticas) -> None:
    """Subplot 3: conflitos do melhor indivíduo (registrados diretamente,
    sem reconverter a partir do fitness, o que evitaria erros de arredondamento)."""
    ax.plot(estatisticas.conflitos_melhor, "r-", linewidth=2, drawstyle="steps-post")
    ax.set_title("CONFLITOS DO MELHOR INDIVÍDUO", fontsize=10, fontweight="bold")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Número de conflitos")
    ax.xaxis.get_major_locator().set_params(integer=True)  # eixos só com inteiros
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(True, alpha=0.3)


def _desenhar_relatorio(ax, resultado: Resultado, num_regioes: int,
                        parametros: ParametrosAG) -> None:
    """Subplot 4: painel de texto com parâmetros e resultados."""
    ax.axis("off")
    linhas = [
        ("RELATÓRIO FINAL", 13, "bold", "black"),
        ("", 6, "normal", "black"),
        (f"Regiões: {num_regioes}", 10, "normal", "black"),
        (f"População: {parametros.tamanho_populacao}  |  Torneio: {parametros.tamanho_torneio}", 10, "normal", "black"),
        (f"Crossover: {parametros.tipo_crossover} ({parametros.taxa_crossover})", 10, "normal", "black"),
        (f"Mutação: {parametros.tipo_mutacao} ({parametros.taxa_mutacao})", 10, "normal", "black"),
        (f"Elite: {parametros.numero_elite}  |  Cores: {parametros.num_cores}  |  Semente: {parametros.semente}", 10, "normal", "black"),
        ("", 6, "normal", "black"),
        (f"Gerações executadas: {resultado.geracoes}", 10, "normal", "black"),
        (f"Tempo de execução: {resultado.tempo_execucao:.3f} s", 10, "normal", "black"),
        (f"Conflitos: {resultado.conflitos}", 10, "normal", "black"),
        (f"Fitness final: {resultado.fitness:.4f}", 10, "normal", "black"),
        ("", 6, "normal", "black"),
    ]
    if resultado.valida:
        linhas.append((f"SOLUÇÃO VÁLIDA - {parametros.num_cores} CORES SEM CONFLITOS", 12, "bold", "green"))
    else:
        linhas.append((f"SOLUÇÃO COM CONFLITOS ({resultado.conflitos})", 12, "bold", "red"))

    y = 0.95
    for texto, tamanho, peso, cor in linhas:
        ax.text(0.05, y, texto, fontsize=tamanho, fontweight=peso, color=cor,
                transform=ax.transAxes, va="top")
        y -= 0.07


def salvar_figura(mapa: Mapa, coordenadas: Coordenadas, resultado: Resultado,
                  parametros: ParametrosAG, titulo: str, caminho: Path) -> Path:
    """Monta a figura 2x2 (mapa, fitness, conflitos, relatório) e salva em PNG."""
    fig, ((ax_mapa, ax_fit), (ax_conf, ax_rel)) = plt.subplots(2, 2, figsize=(16, 12))

    _desenhar_mapa(ax_mapa, mapa, coordenadas, resultado.solucao, titulo)
    _desenhar_fitness(ax_fit, resultado.estatisticas)
    _desenhar_conflitos(ax_conf, resultado.estatisticas)
    _desenhar_relatorio(ax_rel, resultado, len(mapa), parametros)

    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)  # libera a memória da figura
    return caminho


def imprimir_relatorio(regioes: list[str], resultado: Resultado) -> None:
    """Relatório final no console, incluindo a cor atribuída a cada região."""
    print(f"\n{'=' * 70}\nRELATÓRIO DE RESULTADOS\n{'=' * 70}")
    print(f"Tempo de execução : {resultado.tempo_execucao:.3f} s")
    print(f"Gerações          : {resultado.geracoes}")
    print(f"Fitness final     : {resultado.fitness:.4f}")
    print(f"Conflitos         : {resultado.conflitos}")
    print(f"Solução válida    : {'Sim' if resultado.valida else 'Não'}")
    print(f"Cromossomo        : {resultado.solucao}")
    print("\nColoração por região:")
    for regiao, cor in zip(regioes, resultado.solucao):
        print(f"  {regiao:>3} -> {NOMES_CORES[cor]}")


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

def escolher_mapa_interativamente() -> str:
    """Pergunta ao usuário qual mapa usar, repetindo até receber uma opção válida."""
    opcoes = {"1": "brasil", "2": "simplificado"}
    print("Escolha o mapa:")
    print("  1. Brasil (27 estados)")
    print("  2. Mapa simplificado (9 regiões)")
    while True:
        escolha = input("Digite 1 ou 2: ").strip()
        if escolha in opcoes:
            return opcoes[escolha]
        print("Opção inválida. Tente novamente.")


def main() -> None:
    """Lê os argumentos, executa o AG, imprime o relatório e salva a figura."""
    parser = argparse.ArgumentParser(description="Algoritmo Genético para coloração de mapas.")
    parser.add_argument("--mapa", choices=MAPAS, help="mapa a ser colorido (se omitido, pergunta)")
    parser.add_argument("--semente", type=int, default=None,
                        help="semente aleatória para resultados reprodutíveis")
    parser.add_argument("--crossover", choices=("um_ponto", "uniforme"), default="um_ponto",
                        help="tipo de cruzamento (padrão: um_ponto)")
    parser.add_argument("--mutacao", choices=("aleatoria", "dirigida"), default="aleatoria",
                        help="tipo de mutação (padrão: aleatoria)")
    parser.add_argument("--cores", type=int, default=4, choices=(2, 3, 4),
                        help="tamanho do alfabeto genético / nº de cores (padrão: 4)")
    parser.add_argument("--fator-mutacao-dirigida", type=float, default=8.0,
                        help="multiplicador da taxa de mutação nos genes em conflito "
                             "(só com --mutacao dirigida; padrão: 8.0)")
    args = parser.parse_args()

    nome_mapa = args.mapa or escolher_mapa_interativamente()
    criar_mapa, titulo = MAPAS[nome_mapa]
    mapa, coordenadas = criar_mapa()

    parametros = ParametrosAG(
        semente=args.semente,
        tipo_crossover=args.crossover,
        tipo_mutacao=args.mutacao,
        num_cores=args.cores,
        fator_mutacao_dirigida=args.fator_mutacao_dirigida,
    )  # demais valores: padrão da classe

    print(f"\nExecutando o algoritmo para: {titulo}\n")
    ag = AlgoritmoGeneticoColoracao(mapa, parametros)
    resultado = ag.executar()

    imprimir_relatorio(ag.regioes, resultado)

    caminho = salvar_figura(mapa, coordenadas, resultado, parametros, titulo,
                            PASTA_SAIDA / f"resultado_{nome_mapa}.png")
    print(f"\nImagem salva em: {caminho}")


if __name__ == "__main__":
    main()
