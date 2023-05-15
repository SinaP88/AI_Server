FROM python:3.6
COPY . .

RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["waitress-serve", "--call", "app"]
