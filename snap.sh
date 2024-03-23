#!/bin/bash

docker-compose up
docker-compose down
docker volume rm youtube-short-generator_shared-data

# When you have multiple requests
# while true; do
#     docker-compose up
#     docker-compose down
#     docker volume rm  youtube-short-generator_shared-data
#     sleep 1  
# done