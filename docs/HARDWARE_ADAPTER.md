# Contrato da fábrica física

A configuração física aponta `robot.factory` para uma função sem argumentos.
Ela deve construir um wrapper da revisão fixada do LeRobot, sem câmeras, e não
deve conectar nenhum dispositivo durante a construção.

O wrapper expõe:

```python
leader_arms = {"left": ..., "right": ...}
follower_arms = {"left": ..., "right": ...}

def connect(): ...
def abort_connection(): ...
def experiment_stop(reason: str): ...
def experiment_close(): ...
def experiment_abort_close(): ...
def experiment_resync(): ...
```

Cada braço fornece `read("Present_Position")`,
`read("External_Efforts")` e `write("Goal_Position", values)`. Cada vetor tem
sete valores, com a garra na última posição. O wrapper precisa provar na
calibração que esses valores correspondem às unidades internas do harness.

`abort_connection` desfaz somente uma conexão parcial. `experiment_stop`
recebe uma falha durante uma sessão e deve impedir movimentos posteriores sem
presumir comunicação saudável. `experiment_close` encerra uma sessão saudável
com o procedimento validado. `experiment_abort_close` encerra após o latch. Os
três procedimentos são diferentes deliberadamente: o `disconnect()` atual do
checkout legado envia home/sleep e não é uma operação neutra.
`experiment_resync` restaura e confere os modos após a confirmação do operador,
antes de o latch do software poder ser limpo.

O método configurado como `safety.trajectory_validator` recebe um vetor físico
de 14 posições para cada amostra interpolada e retorna exatamente `True` apenas
quando limites geométricos da mesa, colisão entre braços e exceções localizadas
de preensão foram validados. Exceção, `False` ou `None` bloqueiam o comando.

Antes de definir `motion_enabled=true`, registrar no Beads:

- revisão do LeRobot, driver e firmware;
- IP e lado de leaders/followers;
- limites e velocidade por junta;
- conversão e abertura da garra;
- geometria de mesa e braços usada pelo validador;
- comportamento observado de stop/close/abort;
- identificador imutável da calibração.
