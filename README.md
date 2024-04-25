assignment 1 instructions (public key has been added to instance):
1. start instance
2. cd de300
3. cd assignments
If you run sudo docker ps -a, you will see an image assignments-app for the Python file, and mysql:latest for the database.
4. sudo docker build -t assignments-app .
5. sudo docker-compose run app
Can grab the Python file hw1.py from EC2 instance to local machine (using scp) in order to view the plots.

