FROM python:3.6.8

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip -r requirements.txt

EXPOSE 5000
CMD ["python", "api.py"]