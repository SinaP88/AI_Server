FROM python:3.8
COPY . .
WORKDIR ./model
RUN pip install -r requirements.txt
RUN sh ./build.sh
WORKDIR .
EXPOSE 3000
CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]