FROM artifact-base

USER artifact
WORKDIR /home/artifact
CMD /home/artifact/start-notebook.sh
EXPOSE 8888
COPY bootstrap.sh  bootstrap.sh
RUN ./bootstrap.sh
