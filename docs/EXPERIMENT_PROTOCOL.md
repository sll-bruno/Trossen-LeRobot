# Protocolo de transferência e avaliação

Este arquivo especifica execução; resultados e progresso pertencem ao Beads.

## Contrato preservado

A tarefa principal é levantar a bola. Sucesso mantém o contrato do DreamerRL:
parte inferior da bola 5 cm acima da mesa, evento instantâneo e contato com ao
menos uma pastilha. No real, a anotação precisa de referência de altura
calibrada e confirmação do contato. Sem confirmação, o resultado é
`indeterminate`. Estabilidade e participação visual dos braços são métricas
secundárias.

Ordem física: `left_joint_0..6`, seguida de `right_joint_0..6`. As seis
primeiras posições de cada braço são radianos; a posição sete é a coordenada da
garra calibrada em metros. A configuração física deve registrar qualquer
conversão feita pelo driver. Cada transição conserva alvo solicitado, alvo
efetivamente enviado e estado medido; ACT aprende o alvo enviado.

## Gates físicos

1. Identificar por serial a câmera superior e a frontal e verificar streams
   simultâneos pelo tempo do preflight. Índices UVC não contam como identidade.
2. Medir limites, velocidades e conversão da garra. Criar um identificador de
   calibração imutável e implementar o validador de trajetória da bancada.
3. Validar leitura passiva, um braço em baixa velocidade, dois braços, leaders,
   gravação e somente então inferência de política.
4. Cinco pilotos precisam ser revistos antes de congelar frequência, horizonte,
   imagens e coleta definitiva. Pilotos não entram silenciosamente no treino.
5. Uma falha bloqueia novos comandos. A retomada exige inspeção e ressincronização
   explícita do operador; não há reconexão automática.

O processo de câmera é iniciado separadamente. Um frame com mais de 100 ms,
shape incorreto, bundle incompleto, estado antigo, NaN, limite ou velocidade
inválida aborta o episódio. Uma falha no segundo braço é registrada como envio
parcial e trava o caminho de comandos. O software não promete estacionar ou
manter torque quando a comunicação já falhou.

A fábrica física configurada no harness deve construir o `ManipulatorRobot`
somente com leaders e followers; as câmeras pertencem ao processo isolado deste
repositório. O comando de coleta verifica um bundle completo antes de importar
e instanciar essa fábrica. A fábrica deve ainda envolver o objeto com os métodos
calibrados `experiment_stop`, `experiment_close` e `experiment_abort_close`.
Também deve fornecer `experiment_resync` para conferir os modos após reset.
O `disconnect()` do checkout legado comanda home/sleep e não é usado como
fechamento genérico. Até esses métodos serem implementados e validados na
bancada, o backend físico recusa a conexão.

## Partições e ACT

Fixar fisicamente cinco posições de treino, duas de desenvolvimento e cinco de
teste antes da coleta. O teste não participa de calibração aprendida, ajuste,
seleção de checkpoint nem treinamento ACT. Cobertura da distribuição simulada é
declarada separadamente.

Coletar até cinco sucessos por posição de treino e preservar todas as falhas e
abortos. Formar subconjuntos aninhados e balanceados de 5 e 25 demonstrações.
A configuração em `configs/act_reference.json` contém candidatos, não valores
congelados: os cinco pilotos determinam resolução, frequência temporal e chunk.
Registrar inicialização visual, revisões, seeds, atualizações, GPU e dados de
pré-treinamento. Priorizar condições centrais antes de acrescentar seeds ou o
orçamento intermediário.

Para cada checkpoint escolhido no desenvolvimento, executar dois resets em cada
uma das cinco posições de treino e teste, alternando a ordem dos métodos. Uma
semente produz 20 tentativas por checkpoint. Repetições de um checkpoint não
são sementes independentes. Reportar resultados por posição e seed, abortos,
indeterminados e Wilson 95% usando apenas sucessos e falhas no denominador da
taxa de tarefa; a taxa operacional mantém todas as tentativas.

## Custos e interpretação

Registrar separadamente demonstração, calibração/adaptação, reset, supervisão,
configuração, seleção, coleta, desenvolvimento, teste, GPU e transições
simuladas. ACT é uma referência externa. A comparação com ele não isola a
causalidade da curiosidade e menos demonstrações não significa automaticamente
menor custo total.

Sem transferência validada, a entrega é infraestrutura, diagnóstico de
adaptação e ACT preliminar. Ela não será descrita como demonstração física do
método.
