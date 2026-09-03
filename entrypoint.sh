#!/bin/bash

# Custom CA support: mount a PEM file or provide a URL.
#   CUSTOM_CA_PATH — path to a mounted PEM file (preferred, no network call)
#   CUSTOM_CA_URL  — URL to download PEM from (fallback, hits network per pod)
CA_PEM=""

if [ -n "$CUSTOM_CA_PATH" ] && [ -f "$CUSTOM_CA_PATH" ]; then
  CA_PEM="$CUSTOM_CA_PATH"
elif [ -n "$CUSTOM_CA_URL" ]; then
  if curl -fso /tmp/custom-ca.pem "$CUSTOM_CA_URL"; then
    CA_PEM="/tmp/custom-ca.pem"
    echo "INFO: Successfully fetched CA from $CUSTOM_CA_URL" >&2
  else
    echo "WARN: Failed to fetch CA from $CUSTOM_CA_URL, continuing with defaults" >&2
  fi
fi

if [ -n "$CA_PEM" ]; then
  # Use /app (user-writable) instead of /tmp to avoid permission issues with shell redirection
  BUNDLE_PATH="/app/.ca-bundle.pem"

  # Start with system CA bundle
  if command -v python3 &>/dev/null && python3 -m certifi &>/dev/null; then
    cp "$(python3 -m certifi)" "$BUNDLE_PATH"
  elif [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    cp /etc/ssl/certs/ca-certificates.crt "$BUNDLE_PATH"
  elif [ -f /etc/pki/tls/certs/ca-bundle.crt ]; then
    cp /etc/pki/tls/certs/ca-bundle.crt "$BUNDLE_PATH"
  else
    touch "$BUNDLE_PATH"
  fi

  # Make bundle writable (system CA bundles are often read-only)
  chmod u+w "$BUNDLE_PATH"

  # Append custom CA certificate to the bundle
  cat "$CA_PEM" >> "$BUNDLE_PATH" 2>/dev/null || cat "$CA_PEM" | cat >> "$BUNDLE_PATH"
  [ "$CA_PEM" = "/tmp/custom-ca.pem" ] && rm -f /tmp/custom-ca.pem

  export REQUESTS_CA_BUNDLE="$BUNDLE_PATH"
  export SSL_CERT_FILE="$BUNDLE_PATH"
  export CURL_CA_BUNDLE="$BUNDLE_PATH"
  export PIP_CERT="$BUNDLE_PATH"
  export NODE_EXTRA_CA_CERTS="$BUNDLE_PATH"

  echo "INFO: Custom CA bundle configured at $BUNDLE_PATH" >&2
fi

exec "$@"
