#!/bin/bash

gunicorn --bind 0.0.0.0:5001 --workers 2 wsgi

