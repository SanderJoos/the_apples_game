#!/bin/bash

i="0"

while [ $i -lt 5 ]
do
python3 ./agent.py 8001 & 
python3 ./agent.py 8002 &
python3 ./agent.py 8003 &
python3 ./agent.py 8004 &
python3 ./agent.py 8005 &
node play.js ws://localhost:8001 ws://localhost:8002 ws://localhost:8003 ws://localhost:8004 ws://localhost:8005 5

done

