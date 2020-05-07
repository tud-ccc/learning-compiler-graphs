docker build -t artifact-base -f Dockerfile-Base . && docker build -t artifact -f Dockerfile . && docker run -it -p 8888:8888 artifact
