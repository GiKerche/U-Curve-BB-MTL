# U-Curve BB-MTL

O repositório contém:

- o pacote `u_curve_atl`, com o seletor `UcurveATLSelectorGeneric` e os três comparadores MTL (`LogisticL21MTL`, `LogisticDirtyMTL` e `LogisticRMTFL`);
- os scripts utilizados na tese para executar experimentos, inclusive comparações e agregação final de resultados;
- bases sintéticas e reais já pré-formatadas em `base/`, prontas para serem usadas com os scripts.

## Estrutura principal

- `src/u_curve_atl/`: implementação do seletor ATL com busca em largura e heurística de poda na Curva-U, além dos modelos comparativos.
- `test_code_final.py`: pipeline principal de experimentos.
- `run_all.sh`:  shell para disparar múltimas execuções em paralelo (ajuste com variáveis `BASES`, `ESTIMATORS`, `DEPTHS` e limites de jobs).
- `run_mtl_comparators_only.py`: recalcula somente os três métodos MTL clássicos usando os mesmos splits gerados pelo seletor.
- `thesis_results_aggregator.py`: utilitário para reunir métricas de todos os `runs` e gerar tabelas/gráficos consolidados.
- `base/`: diretórios das bases. Cada base possui `info.yaml` com a lista das tarefas (ex.: `t0`, `t1`, ...) e cada tarefa contém um `data.csv` (última coluna é o rótulo), arquivos de índices e metadados.
- `results/`, `runs/`: alvos padrão onde os scripts escrevem saídas intermediárias (históricos, comparações, ablações, etc.).

## Ambiente e dependências

Este projeto foi validado com Python 3.10+. Para preparar o ambiente:

```bash
python -m venv .venv
source .venv/bin/activate         
pip install --upgrade pip
pip install -r requirements.txt
```

As principais bibliotecas usadas são NumPy, pandas, scikit-learn, Optuna, Matplotlib, SciPy e PyYAML.

## Preparando as bases

Cada base deve seguir o esquema usado em `base/`:

```
base/<nome_da_base>/
  info.yaml          # contém a chave "datasets" com a lista de tarefas
  t0/
    data.csv         # features + rótulo (coluna final)
    indexes_test.dat # índices dos splits (opcional)
    ...
  t1/
    ...
```

O script `test_code_final.py` usa `info.yaml` para saber quais diretórios (tarefas) carregar. Para adicionar novas bases basta seguir o mesmo padrão.

## Como executar

### Experimento único

```bash
python test_code_final.py \
  --path base/33gg \
  --estimator logreg \
  --depth 4 \
  --n_trials 120 \
  --compare_mtl \
  --mtl_folds 5 \
  --mtl_repeats 2
```

Parâmetros relevantes:

- `--estimator`: `logreg`, `rf_clf`, `gb_clf`, `rf_reg`, `gb_reg`. Cada opção possui espaço de hiperparâmetros definido em código.
- `--depth`: profundidade máxima da busca (número de fontes adicionadas ao alvo).
- `--n_trials`: número de _trials_ Optuna ao ajustar pesos/hiperparâmetros.
- `--improve_delta`: ganho mínimo exigido para expandir um nó na Curva-U.
- `--compare_mtl`: ao acionar, roda os três comparadores MTL nas tarefas binárias usando os splits salvos pelo seletor.
- `--abl_depths` / `--abl_eps`: listas separadas por vírgula para executar ablações de profundidade e epsilon; os resultados vão para `results/ablation/`.

Saídas diretas desse script:

- `summary_generic_atl.csv`: resumo por _target_ (baseline vs melhor combinação, métricas de teste e parâmetros).
- `history_generic_atl.csv`: histórico das combinações exploradas ao longo da busca.
- `comparisons_mtl.csv`: métricas dos comparadores quando `--compare_mtl` está ativo.
- `results/`: contém gráficos/CSVs adicionais (ablação, comparação de AUC/LogLoss, etc.).

### Execução em lote

Use `run_all.sh` para disparar várias bases/estimadores de uma só vez:

```bash
RUNS_DIR=runs_new \
N_TRIALS=150 \
N_FOLDS=5 \
N_REPEATS=2 \
MAX_JOBS=4 \
bash run_all.sh
```

O script lê as listas `BASES`, `ESTIMATORS` e `DEPTHS` definidas no próprio arquivo e cria subpastas dentro de `RUNS_DIR/<base>/<estimator>/depth_<d>` com os `logs` e CSVs correspondentes.

### Somente comparadores MTL

Para recalcular apenas os modelos MTL mantendo os splits já gerados:

```bash
python run_mtl_comparators_only.py \
  --path base/33gg \
  --base_name 33gg \
  --estimator logreg \
  --run_depth 4 \
  --n_trials 80 \
  --output comparisons_mtl_only.csv
```

Esse comando salva um CSV com as métricas por alvo para `Logistic_L21`, `Logistic_Dirty` e `Logistic_rMTFL`.

## Desenvolvimento

O pacote `u_curve_atl` foi organizado como um módulo instalável (modo `src/`). Caso deseje utilizá-lo em outros projetos python, execute `pip install -e .` na raiz após instalar os requisitos. O código possui _type hints_ e segue a API do `scikit-learn`, o que permite encaixá-lo em _pipelines_ existentes.

---
