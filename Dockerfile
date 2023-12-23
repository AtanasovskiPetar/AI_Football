FROM python:3.10

WORKDIR /ai-football

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./AI_Football ./AI_Football

ENV AZUREML_DATAREFERENCE_OUTPUTS /app/outputs

CMD ["python", "./AI_Football/AIFootball/train.py"]
