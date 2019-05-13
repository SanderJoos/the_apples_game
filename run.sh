#!/bin/bash

for (( a = 5; a <= 7; a++ ))
do
	for i in {1..10}
	do
		python3 ./agent.py 8001 &
		python3 ./agent.py 8002 &
		python3 ./agent.py 8003 &
		python3 ./agent.py 8004 &
		python3 ./agent.py 8005 &
		python3 ./agent.py 8006 &
		python3 ./agent.py 8007 &
		python3 ./agent.py 8008 &
		python3 ./agent.py 8009 &
		python3 ./agent.py 8010 &
		node play.js ws://localhost:8001 ws://localhost:8002 ws://localhost:8003 ws://localhost:8004 ws://localhost:8005 ws://localhost:8006 ws://localhost:8007 ws://localhost:8008 ws://localhost:8009 ws://localhost:8010 5
	done
	sleep 2m
	echo backing up model ${a}
	cp model.h5 backups/model_shoot_${a}.h5

done
