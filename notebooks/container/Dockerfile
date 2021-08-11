FROM jupyter/base-notebook

COPY . /app
RUN pip install -r /app/requirements.txt

CMD start-notebook.sh
