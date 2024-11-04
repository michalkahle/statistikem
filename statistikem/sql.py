import pandas as pd
from dotenv import load_dotenv
import os
import jaydebeapi
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
os.environ['CLASSPATH'] = '/opt/cacheJDBC/cache-jdbc-2.0.0.jar'
import warnings

def read_sql(
    sql,
    driver='com.intersys.jdbc.CacheDriver',
    db='jdbc:Cache://digger20c.ikem.cz:1972/DOCMINER',
    user='mkah_sql',
    password=None,
    **kwargs):
    if password is None:
        load_dotenv()
        password = os.getenv('DIGGER20C')
    with jaydebeapi.connect(driver, db, [user, password]) as connection:
        with warnings.catch_warnings(action='ignore'):
            return pd.read_sql_query(sql, connection, **kwargs)