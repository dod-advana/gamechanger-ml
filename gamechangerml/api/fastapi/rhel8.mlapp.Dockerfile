# [CPU] ARG BASE_IMAGE="registry.access.redhat.com/ubi8/python-36:1-148"
#  -- https://catalog.redhat.com/software/containers/rhel8/python-36/5ba244fc5a134643ef2f04ba?container-tabs=dockerfile
#  -- or ironbank equivalents
# [GPU] ARG BASE_IMAGE="nvidia/cuda:11.2.2-cudnn8-runtime-ubi8"
#  -- https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/ubi8-x86_64/base/Dockerfile
#  -- not yet found on ironbank, unfortunately
ARG BASE_IMAGE="nvidia/cuda:11.2.2-cudnn8-runtime-ubi8"
FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# tmp switch to root for sys pkg setup
USER root

# PYTHON & LOCALE ENV VARS
ENV LANG="C.utf8" \
    LANGUAGE="C.utf8" \
    LC_ALL="C.utf8" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="UTF-8"

# App & Dep Preqrequisites
RUN dnf install -y \
        gcc \
        gcc-c++ \
        glibc-langpack-en \
        python36 \
        python36-devel \
        git \
        zip \
        unzip \
        python3-cffi \
        libffi-devel \
        cairo \
    && dnf clean all \
    && rm -rf /var/cache/yum

# AWS CLI
RUN curl -LfSo /tmp/awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    && unzip -q /tmp/awscliv2.zip -d /opt \
    && /opt/aws/install \
    && rm -f /tmp/awscliv2.zip

# non-root app USER/GROUP
ARG APP_UID=1001
ARG APP_GID=1001

# ensure user/group exists, formally
RUN ((getent group $APP_GID &> /dev/null) \
        || groupadd --system --gid $APP_GID app_default \
    ) && ((getent passwd $APP_UID &> /dev/null) \
        || useradd --system --shell /sbin/nologin --gid $APP_GID --uid $APP_UID app_default \
    )

# key directories
ENV APP_ROOT="${APP_ROOT:-/opt/app-root}"
ENV APP_VENV="${APP_VENV:-/opt/app-root/venv}"
ENV APP_DIR="${APP_ROOT}/src"
RUN mkdir -p "${APP_DIR}" "${APP_VENV}"

# install python venv w all the packages
ARG APP_REQUIREMENTS_FILE="./k8s.requirements.txt"
COPY "${APP_REQUIREMENTS_FILE}" "/tmp/requirements.txt"
RUN python3 -m venv "${APP_VENV}" --prompt mlapp-venv \
    && "${APP_VENV}/bin/python" -m pip install --upgrade --no-cache-dir pip setuptools wheel \
    && "${APP_VENV}/bin/python" -m pip install --no-cache-dir -r "/tmp/requirements.txt" \
    && chown -R $APP_UID:$APP_GID "${APP_ROOT}" "${APP_VENV}"

COPY . "${APP_DIR}"
RUN chown -R $APP_UID:$APP_GID "${APP_DIR}"

USER $APP_UID:$APP_GID
# thou shall not root

ENV MLAPP_VENV_DIR="${APP_VENV}"
WORKDIR "$APP_DIR"
EXPOSE 5000

ENTRYPOINT ["/bin/bash", "./gamechangerml/api/fastapi/startFast.sh"]
CMD ["K8S_DEV"]