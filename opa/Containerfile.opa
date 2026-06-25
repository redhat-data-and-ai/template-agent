FROM alpine:3.19

# Install OPA (static binary)
RUN apk add --no-cache curl \
    && curl -L -o /usr/local/bin/opa https://openpolicyagent.org/downloads/v1.17.1/opa_linux_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')_static \
    && chmod 755 /usr/local/bin/opa \
    && apk del curl

# Create non-root user
RUN addgroup -g 1000 opa && adduser -D -u 1000 -G opa opa

USER opa
WORKDIR /policies

# Start OPA server
ENTRYPOINT ["/usr/local/bin/opa", "run", "--server", "--addr=0.0.0.0:8181", "/policies"]
