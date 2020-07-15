FROM python:3.7.4-slim-stretch
WORKDIR /workdir/rewardpredictive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y git python3-dev python3-numpy libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
                       libsmpeg-dev ibsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev \
                       libfreetype6-dev python-matplotlib texlive-latex-extra texlive-latex-recommended \
                       texlive-xetex dvipng && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN git clone https://github.com/lucaslehnert/rewardpredictive.git /workdir/rewardpredictive
RUN pip install -r requirements.txt

COPY --chown=root:root data /workdir/rewardpredictive/data

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''
