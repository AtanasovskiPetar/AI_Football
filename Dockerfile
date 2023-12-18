FROM python:3.10

WORKDIR /ai-football

COPY requirements.txt .

RUN pip install -r requirements.txt
#RUN pip install -e git+https://github.com/AtanasovskiPetar/AI_Football.git@5a92ffd99243e7cd6dcd806d830d4c41611cef81#egg=gym_envs&subdirectory=gym-envs
RUN pip install -e git+https://github.com/AtanasovskiPetar/AI_Football.git@5a92ffd99243e7cd6dcd806d830d4c41611cef81#egg=gym_envs&subdirectory=AI_Football/gym-envs

COPY ./AI_Football ./AI_Football

CMD ["python", "./AI_Football/AIFootball/train.py"]
