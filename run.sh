#!/bin/bash

i="0"
for (( a = 1; a <= 3; a++ ))
do
	for i in {1..10}
	do
		python3 ./agent.py 8001 &
		python3 ./agent.py 8002 &
		python3 ./agent.py 8003 &
		python3 ./agent.py 8004 &
		python3 ./agent.py 8005 &
		node play.js ws://localhost:8001 ws://localhost:8002 ws://localhost:8003 ws://localhost:8004 ws://localhost:8005 5
	done
	sleep 2m
	echo backing up model ${a}
	cp model.h5 backups/model_exploration_${a}.h5

done
