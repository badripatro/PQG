mkdir csv_files
python tools/jsontocsv.py 
th tools/score.lua csv_files/quora_prepro_test_updated_int.csv csv_files/resuts_json.csv -scorer ter




