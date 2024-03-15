import pymysql
import json


with open("file/key.json", "r") as env:
    env_dict = json.load(env)
    host=env_dict['host']
    user=env_dict['user']
    password = env_dict["password"]

MYSQL_DATABASE_CONN = pymysql.connect(host=host, user=user, password=password, charset='utf8mb4')
conn=MYSQL_DATABASE_CONN
cursor = conn.cursor()

def create_table(conn, cursor, table_name, sql_col):
    sql = f"""CREATE TABLE IF NOT EXISTS {table_name} 
    ({sql_col}) ENGINE=InnoDB DEFAULT CHARSET=utf8"""
    cursor.execute(sql)
    conn.commit()

    return conn, cursor

def get_data_from_table(cursor, sql):
    cursor.execute(sql)
    data_sql = cursor.fetchall()

    return data_sql

def get_first_row(table_name, row):
    # sql_first_row = f"""SELECT {row} FROM {table_name} ORDER BY {row} DESC LIMIT 1;"""
    sql_first_row = f"""SELECT youtube_id, category FROM querys.WEB WHERE stage = '0' ORDER BY time_stamp ASC LIMIT 1;"""
    cursor.execute(sql_first_row)
    first_col = cursor.fetchall()[0]

    return first_col

def get_col_list(cursor, table_name):
    sql_col_list = f"""SELECT `COLUMN_NAME` FROM `INFORMATION_SCHEMA`.`COLUMNS` 
    WHERE `TABLE_SCHEMA`='mu_tech' AND `TABLE_NAME`='{table_name}';"""
    cursor.execute(sql_col_list)
    col_list = cursor.fetchall()

    return col_list

def insert_data_to_table(conn, cursor, list_col, table_sql, tuple_data):
    sql = f"""INSERT INTO {table_sql}({list_col}) VALUES{tuple_data}"""
    cursor.execute(sql)
    conn.commit()

    return conn, cursor


if __name__ == "__main__":
    with open("file/key.json", "r") as env:
        env_dict = json.load(env)
        host=env_dict['host']
        user=env_dict['user']
        password = env_dict["password"]

    MYSQL_DATABASE_CONN = pymysql.connect(host=host, user=user, password=password, charset='utf8mb4')
    
# table_name="querys.server"
    # sql_col="""id varchar(30) NOT NULL"""

    conn=MYSQL_DATABASE_CONN
    cursor = conn.cursor()

    # create_table(conn, cursor, table_name, sql_col)