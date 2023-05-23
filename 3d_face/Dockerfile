FROM python:3.6
COPY . .
WORKDIR .
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install cmake
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN sh ./build.sh
RUN pip cache info
RUN pip cache purge
EXPOSE 3000
CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]