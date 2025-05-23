FROM python:3.9-slim

RUN pip install --no-cache-dir torch==2.0.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

    RUN pip install --no-cache-dir numpy==1.24.4

COPY Project/train/requirements.txt /tmp/train.txt

RUN pip install --no-cache-dir -r /tmp/train.txt \
    && rm -rf /root/.cache /tmp/*

# Copy training and serving scripts
COPY Project/train/train.py /usr/bin/train
COPY Project/serve/predict.py /usr/bin/serve
COPY Project/serve/model.py /usr/bin/model.py
COPY Project/serve/utils.py /usr/bin/utils.py

# Make scripts executable
RUN chmod 755 /usr/bin/train /usr/bin/serve

# Expose port for Flask web server
EXPOSE 8080