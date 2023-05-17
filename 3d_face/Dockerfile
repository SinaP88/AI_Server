FROM python:3.8
COPY . .
WORKDIR .
RUN pip install -r requirements.txt
WORKDIR ./model
RUN sh ./build.sh
WORKDIR .
EXPOSE 3000
CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]