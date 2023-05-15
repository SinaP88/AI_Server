FROM python:3.6
COPY . .

RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["flask", "run", "---host=0.0.0.0", "--port=8000"]
