FROM tensorflow/serving:latest

ENV MODEL_NAME=gym-experience-model

COPY ../output/serving_model/gym-experience-model /models/gym-experience-model

COPY ../config/prometheus.config /models/prometheus.config

CMD ["--model_name=gym-experience-model", "--model_base_path=/models/gym-experience-model", "--monitoring_config_file=/models/prometheus.config"]