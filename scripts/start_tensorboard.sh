#!/bin/bash

docker exec -it -d torch-server /bin/bash -c "cd m-shallowconvnet && tensorboard --logdir logs --host 172.17.0.2;"