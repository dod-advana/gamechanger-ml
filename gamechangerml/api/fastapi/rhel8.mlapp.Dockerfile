ARG BASE_IMAGE="registry.access.redhat.com/ubi8/python-36:1-148"
FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# tmp switch to root for sys pkg setup
USER root

# LOCALE Prereqs
RUN yum install -y \
        glibc-langpack-en \
    && yum clean all \
    && rm -rf /var/cache/yum

# SET LOCALE TO UTF-8
ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"

# App & Dep Preqrequisites
RUN yum install -y \
        gcc \
        gcc-c++ \
        python36-devel \
        git \
        zip \
        unzip \
        python3-cffi \
        libffi-devel \
        cairo \
    && yum clean all \
    && rm -rf /var/cache/yum

# AWS CLI
RUN curl -LfSo /tmp/awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    && unzip -q /tmp/awscliv2.zip -d /opt \
    && /opt/aws/install \
    && rm -f /tmp/awscliv2.zip

# non-root app USER/GROUP
ARG APP_UID=1001
ARG APP_GID=1001

# per convention in red hat python images
ENV APP_ROOT="${APP_ROOT:-/opt/app-root}"
ENV APP_DIR="${APP_ROOT}/src"
RUN mkdir -p "${APP_ROOT}"

# install python venv w all the packages
ARG APP_REQUIREMENTS_FILE="./k8s.requirements.txt"
ENV MLAPP_VENV_DIR="${APP_DIR}/venv"
COPY "${APP_REQUIREMENTS_FILE}" "/tmp/requirements.txt"
RUN python3 -m venv "${MLAPP_VENV_DIR}" \
    && "${MLAPP_VENV_DIR}/bin/python" -m pip install --upgrade --no-cache-dir pip setuptools wheel \
    && "${MLAPP_VENV_DIR}/bin/python" -m pip install --no-deps --no-cache-dir -r "/tmp/requirements.txt" \
    && chown -R $APP_UID:$APP_GID "${ML_APP_VENV_DIR}"

COPY . "${APP_DIR}"
RUN chown -R $APP_UID:$APP_GID "${APP_DIR}"

USER $APP_UID:$APP_GID
# thou shall not root

WORKDIR "$APP_DIR"
EXPOSE 5000
ENTRYPOINT ["/bin/bash", "./gamechangerml/api/fastapi/startFast.sh"]
CMD ["K8S_DEV"]
