FROM public.ecr.aws/lambda/python:3.9
RUN pip install torch transformers huggingface-hub matplotlib numpy scipy PyPDF2 
CMD [ "app.handler" ]
