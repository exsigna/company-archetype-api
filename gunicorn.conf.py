#!/usr/bin/env python3
"""
Gunicorn configuration for Strategic Analysis API
"""

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Cap at 8 workers
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5 minutes - for long-running PDF analysis
keepalive = 2

# Restart workers after this many requests, to control memory usage
max_requests = 100
max_requests_jitter = 20

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "strategic-analysis-api"

# Application
wsgi_app = "main:app"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Preload app for better memory efficiency
preload_app = True

# Worker temporary directory (for file uploads)
worker_tmp_dir = "/dev/shm"  # Use shared memory for tmp files if available

# Graceful shutdown
graceful_timeout = 30

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal"""
    worker.log.info("Worker received SIGABRT signal")

def on_exit(server):
    """Called just before the master process is initialized"""
    server.log.info("Strategic Analysis API shutting down")

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("Strategic Analysis API ready to serve requests")

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal"""
    worker.log.info("Worker interrupted")

def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    server.log.info(f"Worker spawned successfully (pid: {worker.pid})")

def post_worker_init(worker):
    """Called just after a worker has initialized the application"""
    worker.log.info(f"Worker initialized (pid: {worker.pid})")

def worker_exit(server, worker):
    """Called just after a worker has been exited, in the master process"""
    server.log.info(f"Worker exited (pid: {worker.pid})")

# Environment-specific settings
if os.environ.get('ENVIRONMENT') == 'production':
    # Production settings
    loglevel = "warning"
    workers = min(multiprocessing.cpu_count() * 2 + 1, 6)  # Fewer workers in production
    max_requests = 500  # More requests per worker in production
    preload_app = True
elif os.environ.get('ENVIRONMENT') == 'development':
    # Development settings
    workers = 1
    reload = True
    loglevel = "debug"
    timeout = 600  # Longer timeout for debugging