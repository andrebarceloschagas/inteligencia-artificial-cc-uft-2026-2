# Universidade Federal do Tocantins (UFT)

## Ciência da Computação

### Inteligência Artificial — 2026-2

**Professor:** Alexandre Rossini  
**Alunos:** [Antonio Andre](https://github.com/andrebarceloschagas) e Maria Eduarda

## Índice

- [Algoritmos Genéticos](#algoritmos-genéticos)
  - [Coloração de mapas](#coloração-de-mapas)

## Algoritmos Genéticos

Algoritmos genéticos são técnicas de inteligência artificial inspiradas no
processo de seleção natural. Uma população de soluções é evoluída por meio de
seleção, cruzamento e mutação, buscando soluções cada vez melhores para um
determinado problema.

### Coloração de mapas

Este projeto aplica um algoritmo genético ao problema de coloração de mapas.
O objetivo é atribuir uma cor a cada região de modo que regiões vizinhas não
tenham a mesma cor, utilizando o menor número possível de conflitos.

O projeto inclui:

- uma implementação do algoritmo genético;
- mapas para teste, incluindo um mapa simplificado e o mapa do Brasil;
- visualização gráfica dos resultados;
- experimento comparativo entre diferentes parâmetros;
- testes automatizados.

### Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o algoritmo:

```bash
python genetico/genetico_coloracao.py
```

Execute os testes:

```bash
pytest genetico/genetico-coloracao-teste.py -v
```
