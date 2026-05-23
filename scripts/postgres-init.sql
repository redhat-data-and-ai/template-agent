-- Create additional databases needed by demo services.
-- The default database (template_agent) is created by POSTGRES_DB env var.
SELECT 'CREATE DATABASE mcp_server'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mcp_server')\gexec
