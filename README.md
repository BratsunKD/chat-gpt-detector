# chat-gpt-detector-app

Сервис для определения сгенерированого chat gpt текста

Запуск приложения: 

```
docker-compose build 
docker-compose up
```

Отправить запрос: (response: 1 - человек, 0 - chat gpt)

```
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"text": "Пример текста для предсказания"}'
```
