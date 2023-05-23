FROM python:3.8
COPY . .
WORKDIR .
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN apk add --no-cache gcc musl-dev linux-headers
RUN pip install -r requirements.txt
EXPOSE 7000
CMD ["flask", "run", "--host=0.0.0.0", "--port=7000"]