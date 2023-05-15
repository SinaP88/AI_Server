FROM python:3.6
COPY . /opt_app
WORKDIR /opt_app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["waitress-serve", "--call CoreApi:app"]
