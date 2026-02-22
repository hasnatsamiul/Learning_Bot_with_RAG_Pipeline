export $(cat .env | xargs)
python source_indexing.py >> cron.log 2>&1