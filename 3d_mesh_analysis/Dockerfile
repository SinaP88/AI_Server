FROM python:3.8
COPY . .
WORKDIR .
RUN pip install -r requirements.txt
EXPOSE 7000
CMD ["flask", "run", "--host=0.0.0.0", "--port=7000"]