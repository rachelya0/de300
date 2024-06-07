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

assignment 3 instructions:
1. follow steps 1-3 above, listed for assignment 1
2. cd src

(to run code in jupyter notebook)

3. 'docker build .'

4. start the jupyter notebook by 'jupyter notebook --ip=0.0.0.0'

5. fill in aws credentials in cell 2, from https://nu-sso.awsapps.com/start --> access keys
 
6. run all cells in hw3.ipynb

assignment 4 instructions:
1. follow steps 1-3, listed for assignment 1
2. cd src
3. download rachel_hw4.py and upload to de300spring2024demo MWAA environment
4. trigger DAG
