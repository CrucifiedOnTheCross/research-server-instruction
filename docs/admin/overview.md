# Обзор платформы

Эта страница описывает, как устроен сервер и какие компоненты уже настроены.

## Назначение

Сервер используется как общая исследовательская машина:

- JupyterHub для пользователей;
- Docker-окружения с разными библиотеками;
- GPU-окружение для CUDA-задач;
- общие проектные папки и датасеты;
- мониторинг ресурсов;
- Portainer для администрирования контейнеров.

## Хост

- адрес: `10.200.1.180`
- SSH: `ssh nikita@10.200.1.180`
- hostname: `lab-bio`
- ОС: Ubuntu 26.04 LTS
- CPU: Intel Core i7-14700KF
- RAM: 64 GB
- GPU: NVIDIA GeForce RTX 5080, 16 GB VRAM
- SSD: 1 TB

## Основные сервисы

| Сервис | Адрес | Назначение |
| --- | --- | --- |
| JupyterHub | `http://10.200.1.180:8000` | вход пользователей и запуск JupyterLab |
| Grafana | `http://10.200.1.180:3000` | мониторинг ресурсов |
| Portainer HTTP | `http://10.200.1.180:9000` | управление Docker без предупреждения о сертификате |
| Portainer HTTPS | `https://10.200.1.180:9443` | управление Docker через самоподписанный HTTPS |
| Prometheus | `localhost:9090` | сбор метрик, наружу не открыт |

## Размещение файлов

Основные каталоги на сервере:

```text
/opt/research-platform
/srv/research/users
/srv/research/projects
/srv/research/datasets
/srv/research/shared
/srv/research/images
/srv/research/backups
```

Платформа Docker Compose находится в:

```text
/opt/research-platform
```

## Docker Compose сервисы

В платформе настроены:

- `research-jupyterhub`
- `research-portainer`
- `research-grafana`
- `research-prometheus`
- `research-node-exporter`
- `research-cadvisor`
- `research-dcgm-exporter`

## Пользовательские группы Linux

Созданы группы:

- `research-users`
- `research-admins`
- `research-viewers`
- `research-data-ro`
- `research-data-rw`
- `project-default-rw`

Группа `docker` обычным пользователям не выдается, потому что это фактически root-доступ к серверу.

