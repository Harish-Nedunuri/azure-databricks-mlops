# base image
FROM python:3.8
# working dire insdie the container
WORKDIR /code 
# copy this file dependencies file into the container
COPY ./requirements.txt /code/requirements.txt
# Run the pip installer using the base image's pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./app /code/app
COPY ./Railway_attrition_model /code/Railway_attrition_model
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]