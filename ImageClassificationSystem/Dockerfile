FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .

RUN python3.10 -m pip install -r requirements.txt

COPY app.py ${LAMBDA_TASK_ROOT}
COPY FoodModel.pt ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]