assignment 1 instructions (public key has been added to instance):
1. start instance rachelyao_de300
2. cd de300
3. cd assignments
4. run the run.sh file to create initiate all required containers(etl-container and postgres-container) and enter the shell of etl-container with the commands: 1. 'docker start etl-container', 2. 'docker exec -it etl-container /bin/bash'
6. start the jupyter notebook by 'jupyter notebook --ip=0.0.0.0'
7. in a browser, type localhost:8888/tree?token=[paste from terminal output from step 5]
8. run all cells in hw1.ipynb
   
assignment 2 instructions:
1. follow steps 1-7 above, listed for assignment 1
2. fill in aws credentials in cell 2, from https://nu-sso.awsapps.com/start --> access keys
3. run all cells in hw2.ipynb

