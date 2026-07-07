# Docker-образы и отдельные контейнеры

Обычная модель работы: один общий JupyterHub и разные Docker-образы для пользовательских окружений.

## Почему не отдельный JupyterHub на каждого пользователя

Отдельный JupyterHub для каждого пользователя усложняет:

- авторизацию;
- управление портами;
- хранение токенов;
- мониторинг;
- обновления;
- безопасность.

Правильный основной путь: собрать нужный Docker-образ и добавить его в общий JupyterHub.

## Как принять заявку на новый образ

Попросите пользователя прислать:

1. Название проекта.
2. Нужна ли GPU.
3. Язык и версию.
4. Список библиотек.
5. `requirements.txt`, `environment.yml` или `Dockerfile`.
6. Команду проверки.

## Пример Dockerfile для CPU-окружения

```dockerfile
FROM quay.io/jupyter/scipy-notebook:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}
RUN pip install --no-cache-dir \
    opencv-python \
    scikit-image \
    seaborn
```

Сборка:

```bash
sudo docker build -t local/project-name-notebook:2026-07-07 .
```

После сборки добавьте образ в конфигурацию JupyterHub и перезапустите платформу.

## GPU-окружение

На сервере уже настроены:

- NVIDIA driver;
- NVIDIA Container Toolkit;
- CUDA Docker runtime;
- локальный образ `local/research-cuda-notebook:latest`.

Проверка GPU на сервере:

```bash
nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

Проверка CuPy в локальном CUDA-образе:

```bash
sudo docker run --rm --gpus all local/research-cuda-notebook:latest python -c "import cupy as cp; print(cp.arange(10).sum())"
```

## Отдельный JupyterLab-контейнер

Используйте отдельный контейнер только для особых случаев: отдельный порт, нестандартный сервис, долгоживущий worker или проверка готового образа.

Пример `docker-compose.yml` без GPU:

```yaml
services:
  jupyter:
    image: quay.io/jupyter/scipy-notebook:latest
    container_name: project-name-jupyter
    restart: unless-stopped
    ports:
      - "10.200.1.180:8810:8888"
    environment:
      JUPYTER_TOKEN: "CHANGE_ME_LONG_RANDOM_TOKEN"
    volumes:
      - /srv/research/users/USER_LOGIN:/home/jovyan/work
      - /srv/research/projects/PROJECT_NAME:/srv/project
      - /srv/research/datasets/DATASET_NAME:/srv/datasets/DATASET_NAME:ro
    cpus: "4.0"
    mem_limit: 8g
```

Для GPU добавьте:

```yaml
    gpus: all
```

Запуск:

```bash
sudo docker compose up -d
sudo docker compose logs --tail=100
```

Перезапуск:

```bash
sudo docker compose restart
sudo docker compose logs --tail=100
```

Не храните важные данные только внутри контейнера. Подключайте рабочие папки через volumes.

