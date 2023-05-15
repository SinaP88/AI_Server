FROM python:3.6
COPY . /opt_app
WORKDIR /opt_app
RUN pip install -r requirments.txt
EXPOSE 8000
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
