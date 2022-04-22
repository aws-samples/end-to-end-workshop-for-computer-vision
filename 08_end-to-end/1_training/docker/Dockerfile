
FROM public.ecr.aws/docker/library/python:3.7
    
ADD requirements.txt /

RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
