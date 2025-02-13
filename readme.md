List IMG                        docker images
List CON                        docker ps -a

Build IMG                       docker-compose build
Run CON detached                docker-compose up -d
Build IMG + Run CON detached    docker-compose up -d --build
Stop CON + Remove CON           docker-compose down

Stop all CONs                   docker stop $(docker ps -a -q)

Remove all CONs                 docker rm $(docker ps -a -q)
Remove all IMGs                 docker rmi $(docker images -q)
Remove everything               docker system prune -a -f