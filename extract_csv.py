import psycopg2
import pandas as pd
from pathlib import Path
import csv
import os
from psycopg2.extensions import adapt
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME,
                        user=DB_USER, password=DB_PASSWORD)

def get_trimming_sessions():
    cur = conn.cursor()
    # query = ("""
    #     SELECT id, start_time, end_time, trimmed_from, trimmed_to
    #     FROM lttlabs.sessions
    #     WHERE start_time = trimmed_from
    #     AND app_name = 'F1_22'
    #     """)
    query = ("""
        SELECT s.id, start_time, end_time, trimmed_from, trimmed_to
        FROM lttlabs.sessions s
        join lttlabs.gpu_samples gs
        on gs.session_id = s.id
        WHERE start_time != trimmed_from 
        AND start_time IS NOT NULL 
        AND trimmed_from IS NOT NULL
        AND app_name = 'reddeadredemption2'   
        group by s.id, start_time, end_time, trimmed_from, trimmed_to
        having count(distinct(gs.device_name)) = 1
        """)
    cur.execute(query)
    result = cur.fetchall()
    session_id_list = []
    trimmed_from_list = []
    trimmed_to_list = []
    for row in result:
        session_id_list.append(row[0])
        trimmed_from_list.append(row[3])
        trimmed_to_list.append(row[4])
    return session_id_list, trimmed_from_list, trimmed_to_list



def get_GPU_data(session_id):
    cur = conn.cursor()
    query = ("""
        SELECT floor(extract(epoch from "timestamp")/1)*1 AS "time",
        core_clock / 1000 as "Core",
        memory_clock / 1000 as "Memory",
        package_power as "Package Power",
        core_temperature as "Core Temp",
        hot_spot_temperature as "Hot Spot",
        memory_free as "Free",
        memory_used as "Used"
        FROM lttlabs.gpu_samples
        WHERE
        session_id = (%s)
        ORDER BY time ASC
        """)
    cur.execute(query, (session_id, ))
    field_names = [i[0] for i in cur.description]
    col = [(field_names[i] for i in range(len(field_names)))]
    result = cur.fetchall()
    output = col + result
    return output


def get_FPS(session_id):
    cur = conn.cursor()
    query = ("""
        SELECT floor(extract(epoch from "timestamp")/1)*1 AS "time",
        count(frame_time) AS "FPS"
        FROM lttlabs.frame_times
        WHERE
        session_id = (%s) AND
        dropped = false
        GROUP BY 1
        ORDER BY 1,2
        """)
    cur.execute(query, (session_id, ))
    field_names = [i[0] for i in cur.description]
    col = [(field_names[0], field_names[1])]
    result = cur.fetchall()
    output = col + result
    return output
    

def get_trimming_FPS(session_id, trimmed_from, trimmed_to):
    cur = conn.cursor()
    query = ("""
        SELECT floor(extract(epoch from "timestamp")/1)*1 AS "time",
        count(frame_time) AS "FPS"
        FROM lttlabs.frame_times
        WHERE
        session_id = (%s) AND
        "timestamp" BETWEEN (%s) AND (%s) AND
        dropped = false
        GROUP BY 1
        ORDER BY 1,2
        """) 
    cur.execute(query, (session_id, trimmed_from, trimmed_to,))
    field_names = [i[0] for i in cur.description]
    col = [(field_names[0], field_names[1])]
    result = cur.fetchall()
    output = col + result
    return output

def save_file(train_p, test_p, keyword, whole = True, trimmed_from = None, trimmed_to = None):
    for id in session_id_list:
        result = get_FPS(id)
        csv_file = keyword + id + '.csv'
        csv_file_train = os.path.join(train_p, csv_file)
        csv_file_test = os.path.join(test_p, csv_file)
        
        if (not os.path.exists(csv_file_train)) and (not os.path.exists(csv_file_test)):
            fp = open(csv_file_train, 'w')
            myFile = csv.writer(fp)
            myFile.writerows(result)
            fp.close()


if __name__ == '__main__':
    session_id_list, trimmed_from_list, trimmed_to_list = get_trimming_sessions()

    trimming_path_train = Path(r".\reddeadredemption2\reddeadredemption2_trimming")
    whole_path_train = Path(r".\reddeadredemption2\reddeadredemption2_whole")
    gpu_data_path = Path(r".\reddeadredemption2\reddeadredemption2_gpu_data_2sessions")

    if not os.path.exists(trimming_path_train):
        os.makedirs(trimming_path_train)
    if not os.path.exists(whole_path_train):
        os.makedirs(whole_path_train)
    if not os.path.exists(gpu_data_path):
        os.makedirs(gpu_data_path)

    for id in session_id_list:
        result = get_FPS(id)
        csv_file = 'whole_' + id + '.csv'
        csv_file_train = os.path.join(whole_path_train, csv_file)

        if not os.path.exists(csv_file_train):
            fp = open(csv_file_train, 'w')
            myFile = csv.writer(fp)
            myFile.writerows(result)
            fp.close()