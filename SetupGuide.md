# 🤖 Trossen Robotics — LeRobot Setup Guide (Stationary)

> Instruções essenciais para configurar e operar braços robóticos da Trossen Robotics utilizando a biblioteca **LeRobot** da Hugging Face.

---

## 📋 Índice

1. [Miniconda Setup](#-1-miniconda-setup)
2. [Instalação e Setup (LeRobot)](#️-2-instalação-e-setup-lerobot)
3. [Configurando Trossen AI](#️-3-configurando-trossen-ai)
4. [Teleoperação](#️-4-teleoperação)
5. [Hugging Face Authentication](#-5-hugging-face-authentication)
6. [Gravação de Datasets](#-6-gravação-de-datasets)
7. [Replay de Episódio](#-7-replay-de-episódio)
8. [Treinamento e Avaliação](#-8-treinamento-e-avaliação)

---

## 🐍 1. Miniconda Setup

O Miniconda é recomendado para isolar o ambiente Python do LeRobot e evitar conflitos de dependências.

Baixe e execute o instalador para Linux:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Siga as instruções na tela e reinicie o terminal ao finalizar. Para verificar a instalação:

```bash
conda --version
```

Crie um ambiente dedicado para o LeRobot com Python 3.10 e ative-o:

```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

> **Importante:** Execute `conda activate lerobot` sempre que abrir um novo terminal antes de usar o LeRobot.

---

## 🛠️ 2. Instalação e Setup (LeRobot)

> Certifique-se de que o ambiente `lerobot` está ativo (`conda activate lerobot`) antes de prosseguir.

Clone o repositório e instale os pacotes:

```bash
git clone -b trossen-ai https://github.com/Interbotix/lerobot.git ~/lerobot
```

```bash
cd ~/lerobot && pip install --no-binary=av -e ".[trossen_ai]"
```
Instalando dependencias extras  que serao necessarias futuramente:
```bash
conda install -y -c conda-forge 'ffmpeg>=7.0'
```
Para verificar instalacao faca:

```bash
which ffmpeg
```
Assim, tambem e possivel verificar qual versao foi instalada. Segundo o tutorial da Trossen AI, e preferivel uma versao superior a 7.0

Antes de seguir para as proximas etapas, certifique-se de que o ambiente conda esta ativado:
```bash
conda activate lerobot
```

---

## ⚙️ 3. Configurando Trossen AI

Nesta etapa, é essencial configurar os arquivos do LeRobot para que as câmeras sejam conectadas e para garantir a conexão correta dos braços do Stationary.

Navegue até o arquivo de configuração:

```
lerobot/common/robot_devices/robots/configs.py
```

No arquivo `configs.py`, configure o endereço IP correto para cada **Leader** e **Follower**, e insira o número serial de cada câmera RealSense no campo correspondente.

Para verificar o número serial de cada câmera, execute:

```bash
realsense-viewer
```

---

## 🕹️ 4. Teleoperação

Se a configuração ocorreu corretamente, será possível teleoperar o Stationary com o comando abaixo:

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=trossen_ai_stationary \
    --robot.max_relative_target=5 \
    --control.type=teleoperate
```

---

## 🔑 5. Hugging Face Authentication
Primeiro, e necessario baixar os pacotes do Hugging Face Hub:
```bash
pip install --upgrade huggingface_hub
```
A autenticação é necessária para baixar modelos pré-treinados e realizar upload de datasets.

```bash
hf auth login
```

Caso precise forçar a reautenticação ou trocar de conta:

```bash
hf auth login --force
```

> **Importante:** Ao criar um token, certifique-se de que ele possui no mínimo permissões de **WRITE**.

Caso seja necessario trocar de token, é necessário utilizar:
```bash
hf auth switch
```
Para baixar arquivos do Hugging Face Hub, é necessário utilizar a função hf_hub_download(). Aqui segue um exemplo para baixar o modelo Pegasus que está na página do tutorial do  Hugging Face Hub:
```bash
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```
Ou, para baixar uma versao especifica, utilize
```bash
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="google/pegasus-xsum",
    filename="config.json",
    revision="4d33b01d79672f27f001f6abade33f22d993b151"
)
```
Para mais detalhes consultar https://huggingface.co/docs/huggingface_hub/quick-start e procurar pela API de hf_hub_download().


## 💾 6. Gravação de Datasets

Grava episódios de demonstração para treinamento de IA. É necessário alterar os números seriais das câmeras e os IPs antes de executar:

Nesta etapa é importante trocar atribuir à variável HF_USER seu nome de usuário no HuggingFaceHub. Outra maneira, é apenas trocar a variável e escrever diretamente no comando seu nome de usuário. Isso necessariamente deverá ser feito também nos passos de treinamento e também no de avaliação da sua política de treinamento.
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=trossen_ai_stationary \
    --robot.max_relative_target=null \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Test recording episode using Trossen AI Stationary." \
    --control.repo_id=${HF_USER}/trossen_ai_stationary_test \
    --control.tags='["tutorial"]' \
    --control.warmup_time_s=5 \
    --control.episode_time_s=30 \
    --control.reset_time_s=30 \
    --control.num_episodes=2 \
    --control.push_to_hub=true
```

---

## 🔁 7. Replay de Episódio

```bash
uv run lerobot-replay \
    --robot.type=bi_widowxai_follower_robot \
    --robot.left_arm_ip_address=192.168.1.5 \
    --robot.right_arm_ip_address=192.168.1.4 \
    --robot.id=bimanual_follower \
    --dataset.repo_id=${HF_USER}/<dataset-id> \
    --dataset.episode=0
```

---

## 🧠 8. Treinamento e Avaliação

### Executando o Treinamento

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=${HF_USER}/trossen_ai_stationary_test \
    --policy.type=act \
    --output_dir=outputs/train/act_trossen_ai_stationary_test \
    --job_name=act_trossen_ai_stationary_test \
    --device=cuda \
    --wandb.enable=true
```

### Avaliando a Política de Treinamento

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=trossen_ai_stationary \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Recording evaluation episode using Trossen AI Stationary." \
    --control.repo_id=${HF_USER}/eval_act_trossen_ai_stationary_test \
    --control.tags='["tutorial"]' \
    --control.warmup_time_s=5 \
    --control.episode_time_s=30 \
    --control.reset_time_s=30 \
    --control.num_episodes=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_trossen_ai_stationary_test/checkpoints/last/pretrained_model \
    --control.num_image_writer_processes=1
```

---

*Última atualização: Abril de 2026*
