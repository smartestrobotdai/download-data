#!/bin/bash
ORI_DIR=`pwd`
DIR=`dirname $0`
LOG_FILE="nightly.log"
if [ ! -z ${DIR} ]; then
  cd ${DIR}
  echo "changing directory to ${DIR}">>${LOG_FILE}
fi
echo "`date` started task">>${LOG_FILE}
echo "node version: `node -v`">>${LOG_FILE}
export PATH=$PATH:/usr/local/bin/:/usr/bin
node daily.js>>${LOG_FILE} 2>&1
node minute.js>>${LOG_FILE} 2>&1
docker exec -t postgres-omxs pg_dumpall -c -U postgres > backup/dump_`date +%d-%m-%Y"_"%H_%M_%S`.sql
cd ${ORI_DIR}
