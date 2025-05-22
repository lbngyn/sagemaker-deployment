FROM python:3.9

# Install required packages from serve/requirements.txt and train/requirements.txt
COPY Project/serve/requirements.txt /tmp/serve-requirements.txt
COPY Project/train/requirements.txt /tmp/train-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/serve-requirements.txt -r /tmp/train-requirements.txt

# Copy training and serving scripts
COPY Project/train/train.py /usr/bin/train
COPY Project/serve/predict.py /usr/bin/serve
COPY Project/serve/model.py /usr/bin/model.py
COPY Project/serve/utils.py /usr/bin/utils.py

# Make scripts executable
RUN chmod 755 /usr/bin/train /usr/bin/serve

# Expose port for Flask web server
EXPOSE 8080