FROM python:3.12-slim
WORKDIR /app/
COPY requirements.txt ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["./app.py","--host=0.0.0.0"]
