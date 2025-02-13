List IMG                        docker images
List CON                        docker ps -a

Build IMG                       docker-compose build
Run CON detached                docker-compose up -d
Build IMG + Run CON detached    docker-compose up -d --build
Stop CON + Remove CON           docker-compose down

Clean unused IMG & CON          docker system prune -f