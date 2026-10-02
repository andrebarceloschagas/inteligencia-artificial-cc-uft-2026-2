"""
Experimento comparativo — Algoritmo Genético para Coloração de Mapas
=====================================================================

Roda o algoritmo várias vezes (uma por semente) para cada configuração de
parâmetro e resume quantas gerações foram necessárias até a solução válida,
além da taxa de sucesso. Serve para a seção "Testes e resultados" do
trabalho: mostra o efeito de cada operador/parâmetro de forma objetiva, em
vez de uma única execução isolada.

Três comparações são feitas (todas no mapa do Brasil, por padrão):

  1. Taxa de mutação:     0.01 vs 0.02 vs 0.05
  2. Tamanho do torneio:  2 vs 3 vs 5 vs 7
  3. Tipo de operador:    crossover um_ponto vs uniforme,
                          mutação aleatória vs dirigida

Saídas:
  - tabela impressa no console;
  - `experimento_resultados.csv` com os números completos;
  - `experimento_comparativo.png` com um gráfico de barras (gerações médias
    ± desvio padrão) para cada comparação.

Uso:
    python genetico-coloracao-experimento-comparativo.py
    python genetico-coloracao-experimento-comparativo.py --mapa simplificado --execucoes 50
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt

from genetico_coloracao import AlgoritmoGeneticoColoracao, MAPAS, ParametrosAG

PASTA_SAIDA = Path(__file__).resolve().parent


@dataclass
class ResumoConfiguracao:
    """Resumo estatístico de uma configuração, agregando várias sementes."""

    grupo: str              # nome da comparação (ex.: "Taxa de mutação")
    rotulo: str             # valor testado (ex.: "0.05")
    execucoes: int
    sucessos: int           # quantas execuções terminaram sem conflitos
    geracoes_media: float
    geracoes_desvio: float
    geracoes_max: int
    tempo_medio: float      # segundos por execução

    @property
    def taxa_sucesso(self) -> float:
        return self.sucessos / self.execucoes


def rodar_configuracao(mapa, parametros_base: ParametrosAG, overrides: dict,
                       sementes: list[int], grupo: str, rotulo: str) -> ResumoConfiguracao:
    """Executa o AG uma vez por semente com os mesmos `overrides` e resume o resultado."""
    geracoes: list[int] = []
    tempos: list[float] = []
    sucessos = 0

    for semente in sementes:
        parametros = replace(parametros_base, semente=semente, **overrides)
        resultado = AlgoritmoGeneticoColoracao(mapa, parametros).executar(verbose=False)
        geracoes.append(resultado.geracoes)
        tempos.append(resultado.tempo_execucao)
        sucessos += resultado.valida

    return ResumoConfiguracao(
        grupo=grupo,
        rotulo=rotulo,
        execucoes=len(sementes),
        sucessos=sucessos,
        geracoes_media=statistics.fmean(geracoes),
        geracoes_desvio=statistics.pstdev(geracoes) if len(geracoes) > 1 else 0.0,
        geracoes_max=max(geracoes),
        tempo_medio=statistics.fmean(tempos),
    )


def executar_experimento(nome_mapa: str, num_execucoes: int) -> list[ResumoConfiguracao]:
    """Roda as três comparações descritas no cabeçalho do arquivo."""
    criar_mapa, _ = MAPAS[nome_mapa]
    mapa, _ = criar_mapa()
    sementes = list(range(num_execucoes))
    base = ParametrosAG(numero_geracoes=1000)  # demais parâmetros: padrão da classe
    resumos: list[ResumoConfiguracao] = []

    print(f"Mapa: {nome_mapa} | execuções por configuração: {num_execucoes}")
    print("Isso pode levar alguns segundos...\n")

    # 1) Taxa de mutação -----------------------------------------------
    for taxa in (0.01, 0.02, 0.05):
        resumos.append(rodar_configuracao(
            mapa, base, {"taxa_mutacao": taxa}, sementes,
            grupo="Taxa de mutação", rotulo=f"{taxa:.2f}",
        ))

    # 2) Tamanho do torneio ---------------------------------------------
    for torneio in (2, 3, 5, 7):
        resumos.append(rodar_configuracao(
            mapa, base, {"tamanho_torneio": torneio}, sementes,
            grupo="Tamanho do torneio", rotulo=str(torneio),
        ))

    # 3) Operadores de cruzamento e mutação ------------------------------
    combinacoes = [
        ("um_ponto", "aleatoria", "1 ponto + aleatória (original)"),
        ("uniforme", "aleatoria", "uniforme + aleatória"),
        ("um_ponto", "dirigida", "1 ponto + dirigida"),
        ("uniforme", "dirigida", "uniforme + dirigida"),
    ]
    for tipo_crossover, tipo_mutacao, rotulo in combinacoes:
        resumos.append(rodar_configuracao(
            mapa, base, {"tipo_crossover": tipo_crossover, "tipo_mutacao": tipo_mutacao},
            sementes, grupo="Operadores", rotulo=rotulo,
        ))

    return resumos


def imprimir_tabela(resumos: list[ResumoConfiguracao]) -> None:
    """Mostra os resultados em formato de tabela simples no console."""
    cabecalho = (f"{'Grupo':<20} {'Configuração':<28} {'Sucesso':>8} "
                 f"{'Gerações (média±dp)':>22} {'Máx':>6} {'Tempo médio':>12}")
    print(cabecalho)
    print("-" * len(cabecalho))
    for r in resumos:
        media_dp = f"{r.geracoes_media:.1f} ± {r.geracoes_desvio:.1f}"
        print(f"{r.grupo:<20} {r.rotulo:<28} {r.taxa_sucesso:>7.0%} "
              f"{media_dp:>22} {r.geracoes_max:>6} {r.tempo_medio * 1000:>10.1f} ms")


def salvar_csv(resumos: list[ResumoConfiguracao], caminho: Path) -> None:
    """Grava os resultados completos em CSV, para quem quiser abrir no Excel/Sheets."""
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["grupo", "configuracao", "execucoes", "sucessos", "taxa_sucesso",
                           "geracoes_media", "geracoes_desvio", "geracoes_max", "tempo_medio_s"])
        for r in resumos:
            escritor.writerow([r.grupo, r.rotulo, r.execucoes, r.sucessos, f"{r.taxa_sucesso:.4f}",
                               f"{r.geracoes_media:.2f}", f"{r.geracoes_desvio:.2f}",
                               r.geracoes_max, f"{r.tempo_medio:.4f}"])


def _desenhar_grupo(ax, resumos: list[ResumoConfiguracao], titulo: str) -> None:
    """Um gráfico de barras (gerações médias ± desvio padrão) para um grupo de configurações."""
    rotulos = [r.rotulo for r in resumos]
    medias = [r.geracoes_media for r in resumos]
    desvios = [r.geracoes_desvio for r in resumos]

    barras = ax.bar(rotulos, medias, yerr=desvios, capsize=5,
                    color="#1D70B8", edgecolor="black", alpha=0.85)
    for barra, r in zip(barras, resumos):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + max(medias) * 0.03,
               f"{r.taxa_sucesso:.0%} válidas", ha="center", fontsize=8)

    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_ylabel("Gerações até a solução (média ± dp)")
    ax.set_ylim(bottom=0)  # gerações nunca são negativas; corta a barra de erro em 0
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20 if max(len(r) for r in rotulos) > 6 else 0)


def salvar_grafico(resumos: list[ResumoConfiguracao], caminho: Path) -> Path:
    """Monta um gráfico de barras por grupo de comparação e salva em PNG."""
    grupos = list(dict.fromkeys(r.grupo for r in resumos))  # ordem de inserção, sem repetir
    fig, eixos = plt.subplots(1, len(grupos), figsize=(6 * len(grupos), 5.5))
    if len(grupos) == 1:
        eixos = [eixos]

    for eixo, grupo in zip(eixos, grupos):
        _desenhar_grupo(eixo, [r for r in resumos if r.grupo == grupo], grupo)

    fig.suptitle("Experimento comparativo — gerações até a solução válida", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento comparativo do AG de coloração de mapas.")
    parser.add_argument("--mapa", choices=MAPAS, default="brasil", help="mapa usado no experimento")
    parser.add_argument("--execucoes", type=int, default=30,
                        help="execuções (sementes) por configuração (padrão: 30)")
    args = parser.parse_args()

    resumos = executar_experimento(args.mapa, args.execucoes)

    print()
    imprimir_tabela(resumos)

    caminho_csv = PASTA_SAIDA / "experimento_resultados.csv"
    salvar_csv(resumos, caminho_csv)
    print(f"\nTabela completa salva em: {caminho_csv}")

    caminho_png = salvar_grafico(resumos, PASTA_SAIDA / "experimento_comparativo.png")
    print(f"Gráfico salvo em: {caminho_png}")


if __name__ == "__main__":
    main()
