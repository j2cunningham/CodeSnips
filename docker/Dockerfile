# docker run -it --entrypoint /bin/bash <image>
# docker run --rm -p 5001:5001 snip:1.0

FROM python:3.7

RUN apt-get update && apt-get install net-tools -y \
    curl \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /app

# COPY requirements.txt /app
# COPY api.py /app
COPY requirements.txt api.py run.sh /app

WORKDIR /app

RUN chmod 770 run.sh

RUN pip install -r requirements.txt
EXPOSE 5001


CMD ["bash","run.sh"]