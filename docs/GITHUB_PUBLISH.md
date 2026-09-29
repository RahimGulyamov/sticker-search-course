# Публикация через ноутбук

Под без интернета не должен подключаться к GitHub. Код переносится на ноутбук,
и `git push` выполняется оттуда. Данные и большие веса хранятся отдельно в YT.

## Готовый архив

Распакуйте `sticker_search_github_ready.zip` в новую папку на Mac. Он содержит
актуальный код SigLIP 2, раздельную генерацию и удаление фона, результаты теста,
презентацию и документацию. Установка ML-зависимостей для публикации не нужна.

```bash
mkdir -p ~/Projects/sticker-search-publish
cd ~/Projects/sticker-search-publish
unzip ~/Downloads/sticker_search_github_ready.zip
cd sticker_search
```

## Если на поде есть собственные изменения кода

Скопируйте `scripts/export_github.py` на под и выполните его из проекта.
Скрипт работает без сторонних пакетов и интернета, читает текущие файлы пода,
не изменяет их и не включает модели, индексы, данные и локальные `.env`.

```bash
cd /home/jovyan/sticker_search
python scripts/export_github.py --output /tmp/sticker_search_pod_code.zip
```

В VS Code откройте папку `/tmp` в подключённом поде, выберите архив и выполните
**Download / Скачать**, если команда доступна. Можно скачать тот же файл через
файловый браузер Jupyter. Сравните исходники из этого архива с готовым комплектом
на ноутбуке и перенесите свои правки перед commit. Старые README и отчёт на поде
могут ещё описывать CLIP, поэтому не заменяйте ими актуальную документацию вслепую.

## Войти и опубликовать с Mac

```bash
brew install gh
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
git init -b main
git add .
git diff --cached --stat
git status --short
git commit -m "Add sticker search project with SigLIP 2 results"
gh repo create sticker-search-course --private --source=. --remote=origin --push
gh repo view --json url --jq .url
```

Команды `git` и `gh repo create` выполняются внутри распакованной папки
`sticker_search`. Если Git просит автора коммита, задайте `git config user.name`
и `git config user.email` для этого репозитория; email можно взять в настройках
GitHub, в том числе адрес `noreply`. Затем повторите commit и создание репозитория.

По умолчанию команда создаёт **приватный** репозиторий. Для публичного замените
`--private` на `--public` перед созданием. Для сдачи приватного проекта нужно
дать преподавателю доступ. Токены в чат и файлы репозитория добавлять не нужно.

Если репозиторий уже создан на GitHub, вместо `gh repo create` выполните:

```bash
git remote add origin https://github.com/YOUR_LOGIN/YOUR_REPOSITORY.git
git push -u origin main
```

Для этого варианта создавайте репозиторий пустым: без README, лицензии и
`.gitignore` на стороне GitHub. Если там уже есть коммиты, сначала согласуйте
историю; `--force` для первоначальной публикации не нужен.

## Что хранится отдельно

- `data/`, `runs/`, `models/`, кеши, окружения и checkpoints не отправляются.
- `configs/pod.env` — локальный файл; в Git есть `configs/pod.env.example`.
- Сохранённые метрики SigLIP — `evidence/siglip2/`. Веса не включены.
- `output/pdf/final_report.pdf` — отчёт предыдущего CLIP-эксперимента.
  Он не заменяет итоговый отчёт с новыми результатами SigLIP 2.

Для точного воспроизведения сохраняйте выбранный checkpoint, индекс, данные и
разметку в YT до удаления пода. Наличие кода на GitHub не сохраняет эти артефакты.

Документация GitHub CLI:
https://cli.github.com/manual/gh_auth_login
https://cli.github.com/manual/gh_repo_create
