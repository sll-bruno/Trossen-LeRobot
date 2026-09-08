# Trossen-LeRobot

Infraestrutura fail-closed para teleoperação, coleta humana, exportação ao
LeRobot/ACT e avaliação do experimento bimanual. O modo padrão é simulado; uma
configuração física incompleta é rejeitada antes de conectar o robô.

## Início rápido sem hardware

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/trossen-experiment preflight \
  --config configs/experiment.mock.json
```

Um episódio simulado pode ser gravado e validado assim:

```bash
.venv/bin/trossen-experiment record \
  --config configs/experiment.mock.json \
  --backend simulated \
  --duration 1 \
  --position train_01 \
  --split train \
  --pilot \
  --outcome indeterminate

.venv/bin/trossen-experiment validate-dataset data/raw
```

O fluxo físico exige uma cópia da configuração com limites medidos,
`calibration_id`, fábrica LeRobot, validador de trajetória e
`motion_enabled=true`. Não use os limites do arquivo `mock` no equipamento.
[`configs/experiment.physical.example.json`](configs/experiment.physical.example.json)
é somente um template bloqueado; seus limites mínimos também não são uma
calibração.
Execute primeiro o preflight isolado das câmeras; `preflight` não conecta os
braços a menos que outro comando de controle seja escolhido.
O wrapper exigido está especificado em
[`docs/HARDWARE_ADAPTER.md`](docs/HARDWARE_ADAPTER.md).

## Comandos

- `preflight`: valida contrato; opcionalmente mede câmeras sem importar o driver
  dos braços.
- `teleop`: leaders e followers pelo mesmo loop seguro, preservando um episódio
  de auditoria.
- `record`: coleta e rotula um episódio.
- `rollout`: executa uma fábrica de política em processo separado; o mesmo gate
  de segurança valida suas ações.
- `validate-dataset`: verifica frames, dimensões, timestamps e arquivos.
- `annotate-episode`: registra a revisão, o revisor e o frame de sucesso.
- `export-lerobot`: exporta somente sucessos revisados do split de treino.
- `make-subsets`: cria manifestos aninhados de 5/25, cobrindo todas as
  posições de treino antes de repetir uma posição.
- `evaluate`: cria uma ordem fixa e alternada de ensaios.
- `record-result`: acrescenta o resultado de um ensaio da matriz sem apagar os
  anteriores.
- `summarize`: calcula taxas, abortos, indeterminados e intervalo Wilson de 95%.

O protocolo científico e os gates de bancada estão em
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). A lacuna dos
quatro sinais de contato do Dreamer está registrada em
[`docs/CONTACT_ADAPTATION.md`](docs/CONTACT_ADAPTATION.md) e no Beads.

Durante `record` e `rollout`, um terminal interativo aceita `s`, `f`, `i` ou
`a` seguido de Enter para encerrar e marcar sucesso, falha, indeterminado ou
aborto. Sem anotação até o horizonte, o resultado é indeterminado. Use
`--cost demonstration=...`, `--cost reset=...` e `--cost supervision=...`
para registrar segundos de trabalho humano. Para ACT, `rollout` pode usar
`trossen_experiment.sources:act_policy_factory`; o JSON de argumentos precisa
informar checkpoint, device e o mapa das features visuais para câmeras físicas.
Primitivas futuras podem usar
`trossen_experiment.primitives:waypoint_policy_factory`; o arquivo de waypoints
deve permanecer versionado e conter alvos físicos calibrados. Isso não libera a
coleta automática na matriz científica atual.

Uma marcação ao vivo é provisória. Antes de criar os subconjuntos ACT, revise o
episódio e aprove-o; sucessos exigem o índice do primeiro frame que satisfaz
altura e contato:

```bash
trossen-experiment annotate-episode data/raw/ID.episode \
  --outcome success \
  --success-frame 123 \
  --reason 'altura e contato confirmados no vídeo' \
  --reviewer NOME
```
