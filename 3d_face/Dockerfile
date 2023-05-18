FROM python:3.6
COPY . .
WORKDIR .
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN sh ./build.sh
EXPOSE 3000
CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]