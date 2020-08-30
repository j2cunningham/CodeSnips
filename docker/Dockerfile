# docker run -it --entrypoint /bin/bash <image>
# docker run --rm -p 5001:5001 snip:1.0

FROM python:3

# ENV HOME=/var/www

RUN apt-get update && apt-get install net-tools -y \
    curl \
    vim \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /app

# COPY requirements.txt /app
# COPY api.py /app
COPY requirements.txt api.py run.sh wsgi.py /app/

WORKDIR /app

RUN chmod 770 run.sh

RUN pip install -r requirements.txt



CMD ["bash","run.sh"]