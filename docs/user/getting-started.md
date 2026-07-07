# Быстрый старт пользователя

Эта страница нужна, чтобы быстро начать работу в JupyterHub.

## 1. Подключитесь к сети

Если вы не в университетской сети, сначала подключитесь к VPN Самарского университета.

Официальный раздел справки:

```text
https://help.ssau.ru/otrs/public.pl?Action=PublicFAQExplorer;CategoryID=1
```

Проверенные прямые страницы:

- Windows IKEv2/IPSec, FAQ 250043: `https://help.ssau.ru/otrs/public.pl?Action=PublicFAQZoom;ItemID=43`
- Windows L2TP/IPSec, FAQ 250037: `https://help.ssau.ru/otrs/public.pl?Action=PublicFAQZoom;ItemID=37`
- Linux, Mac OS и другие системы, FAQ 250040: `https://help.ssau.ru/otrs/public.pl?Action=PublicFAQZoom;ItemID=40`
- Ideco Client, FAQ 250039: `https://help.ssau.ru/otrs/public.pl?Action=PublicFAQZoom;ItemID=39`

Проверьте, что сервер доступен:

```bash
ping 10.200.1.180
```

Если `ping` не отвечает, попробуйте сразу открыть JupyterHub:

```text
http://10.200.1.180:8000
```

## 2. Зарегистрируйтесь в JupyterHub

Откройте страницу регистрации:

```text
http://10.200.1.180:8000/hub/signup
```

После регистрации сообщите ответственному, что учетная запись создана. Администратор должен подтвердить пользователя.

После подтверждения откройте:

```text
http://10.200.1.180:8000
```

Войдите под своим логином и паролем.

## 3. Выберите окружение

Доступные окружения:

- `Python SciPy Notebook` - базовая научная работа на Python;
- `Data Science Notebook` - расширенный набор для анализа данных;
- `R Notebook` - работа на R;
- `PyTorch CPU Notebook` - PyTorch без GPU;
- `CUDA 13 Lab (GPU)` - задачи с видеокартой.

Если не знаете, что выбрать, начните с `Python SciPy Notebook`.

## 4. Где работать

Основная личная папка:

```text
/home/jovyan/work
```

Сохраняйте там личные ноутбуки, скрипты и промежуточные файлы. Подробности есть на странице [Данные и датасеты](data.md).

## 5. Как перезапустить свою работу

Если завис код в ноутбуке:

```text
Kernel -> Restart Kernel
```

Если зависло все окружение:

1. Откройте `File -> Hub Control Panel`.
2. Нажмите `Stop My Server`.
3. После остановки нажмите `Start My Server`.

Файлы в `/home/jovyan/work` сохраняются между перезапусками.

