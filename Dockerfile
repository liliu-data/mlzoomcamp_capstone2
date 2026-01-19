FROM public.ecr.aws/lambda/python:3.9

# Use --index-url to point directly to the CPU wheelhouse
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy your files
COPY model.pt ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]