# Эксплуатация и мониторинг

Эта страница нужна для ежедневного обслуживания платформы.

## Основные команды

Проверить состояние:

```bash
sudo research-platform status
```

Посмотреть последние логи:

```bash
sudo research-platform logs 200
```

Перезапустить платформу:

```bash
sudo research-platform restart
```

Посмотреть пароль Grafana:

```bash
sudo research-platform grafana-password
```

Создать проект:

```bash
sudo research-create-project PROJECT_NAME
```

## Docker Compose

Платформа находится в:

```text
/opt/research-platform
```

Если нужно работать напрямую:

```bash
cd /opt/research-platform
sudo docker compose ps
sudo docker compose logs --tail=100
sudo docker compose restart
```

## Мониторинг Grafana

Адрес:

```text
http://10.200.1.180:3000
```

Dashboard:

```text
Research Server Overview
```

Логин администратора Grafana:

```text
admin
```

Пароль не хранится в wiki. Используйте:

```bash
sudo research-platform grafana-password
```

## Метрики

В мониторинг включены:

- CPU;
- RAM;
- диск;
- Docker-контейнеры;
- GPU utilization;
- GPU temperature;
- GPU memory;
- DCGM exporter;
- node exporter;
- cAdvisor.

Prometheus доступен внутри сервера на `localhost:9090`.

## Portainer

Адрес без предупреждения о сертификате:

```text
http://10.200.1.180:9000
```

HTTPS-адрес может показывать предупреждение о самоподписанном сертификате:

```text
https://10.200.1.180:9443
```

Portainer используйте для просмотра и перезапуска контейнеров, но изменения платформы лучше фиксировать через файлы в `/opt/research-platform`.

## Проверка GPU

```bash
nvidia-smi
```

Проверка GPU из Docker:

```bash
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

