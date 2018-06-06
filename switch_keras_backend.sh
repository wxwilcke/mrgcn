#!/bin/bash

declare -r KERAS_CONFIG=$HOME/.keras/keras.json

get_backend () {
	echo $(awk 'match($0, /\"backend\":\s\"(\w+)\"/, result) {print result[1]}' $KERAS_CONFIG)
}

set_backend () {
	local newconfig=$(sed 's/'$1'/'$2'/g' $KERAS_CONFIG)
	if [ ! -z "$newconfig" ]
	then
		echo "$newconfig" > $KERAS_CONFIG
		return 0
	else
		return 1
	fi
}

declare backend=$(get_backend)
if [ "$1" != "$backend" ]
then
	set_backend "$backend" "$1"
	if [ "$?" -eq "0" ]
	then
		echo "Backend set to $1"
		exit 0
	else
		echo "Error setting backend"
		exit 1
	fi
else
	echo "Backend already set to $1"
	exit 2
fi
