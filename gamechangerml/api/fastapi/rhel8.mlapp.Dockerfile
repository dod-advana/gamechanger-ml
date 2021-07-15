ARG BASE_IMAGE="registry.access.redhat.com/ubi8/python-36:1-148"
FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# tmp switch to root for sys pkg setup
USER root

# App & Dep Preqrequisites
RUN yum install -y \
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

# per convention in red hat python images
ENV APP_ROOT="${APP_ROOT:-/opt/app-root}"
ENV APP_DIR="${APP_ROOT}/src"
RUN mkdir -p "${APP_DIR}" \
    && chown -R 1001:0 "${APP_ROOT}"
WORKDIR "$APP_DIR"

USER 1001
# thou shall not root

COPY --chown=1001:0 ./requirements.txt "$APP_DIR/requirements.txt"
RUN python3 -m venv "$APP_DIR/venv" \
    && "$APP_DIR/venv/bin/python" -m pip install --upgrade --no-cache-dir pip setuptools wheel
    && "$APP_DIR/venv/bin/python" -m pip install --no-deps --no-cache-dir -r "$APP_DIR/requirements.txt"

COPY --chown=1001:0 . "${APP_DIR}"

EXPOSE 5000
ENTRYPOINT ["/bin/bash", "./gamechangerml/api/fastapi/startFast.sh"]
CMD ["DEV"]