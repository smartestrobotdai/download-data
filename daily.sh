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
docker-compose up -d
sleep 10
cd data-sink/
node daily.js>>${LOG_FILE} 2>&1
node minute.js>>${LOG_FILE} 2>&1
cd ..
SUFFIX=`date +%d-%m-%Y"_"%H_%M_%S`
docker exec -t postgres-omxs pg_dumpall -c -U postgres > backup/dump_$SUFFIX.sql

gzip backup/dump_$SUFFIX.sql
cp -f backup/dump_$SUFFIX.sql.gz backup/dump.sql.gz
git add backup/dump.sql.gz

PGPASSWORD=dai psql -h 0.0.0.0 -p 5434 -U postgres -f export_data.sql
docker cp postgres-omxs:/tmp/data.csv ./data
gzip data/data.csv
git add data/data.csv.gz
git commit -m "new backup and new data"
git push
cd ${ORI_DIR}
