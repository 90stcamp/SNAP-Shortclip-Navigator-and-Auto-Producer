#!/bin/bash

# python3 'server2.py'
docker-compose up
docker-compose down
docker volume rm  kis_shared-data

# while true; do
    # python3 'server2.py'
    # docker-compose up
    # docker-compose down
    # docker volume rm  kis_shared-data
#     sleep 1  
# done