#!/bin/bash
 
echo "**********     início     **********"
 
 
echo "**********  pip install   **********"
python3 -m pip install -r requirements.txt
 
echo "********** makemigrations **********"
python3 manage.py makemigrations --noinput
 
echo "**********    migrate     **********"
python3 manage.py migrate --noinput
 
echo "********** collectstatic  **********"
python3 manage.py collectstatic --noinput
 

echo "**********      fim       **********"
