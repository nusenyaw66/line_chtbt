#!/bin/bash
exec gunicorn --bind 0.0.0.0:$PORT --log-level debug --workers 2 webhook_flask_srvr:app