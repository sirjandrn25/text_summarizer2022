SECRET_KEY = 'django-insecure-5*l!in4$t5n3x!8gl98s7y38l($$owi6)5ao#w#*#i80i$*dd$'
web: waitress-serve --port=$PORT text_summarization.wsgi:application
web: gunicorn text_summarization.wsgi:application --log-file - --log-level debu