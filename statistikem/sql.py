import os
import warnings
from getpass import getpass

import jaydebeapi
import pandas as pd
from dotenv import load_dotenv

_SESSION_CACHE = {}


def read_sql(sql, server='jdbc_analytics', parse_dates=None, url=None, user=None, **kwargs):
    with jaydebeapi.connect(*_get_creds(server, url, user)) as connection:
        with warnings.catch_warnings(action='ignore'):
            return pd.read_sql_query(sql, connection, parse_dates=parse_dates, **kwargs)


def execute_sql(sql, server=None, parse_dates=None, url=None, user=None):
    with jaydebeapi.connect(*_get_creds(server, url, user)) as connection:
        rows_affected = 0
        with connection.cursor() as cursor:
            cursor.execute(sql)
            rows_affected = cursor.rowcount if cursor.rowcount is not None else 0
        connection.commit()
        return rows_affected


def _get_creds(server, url, user):
    global _SESSION_CACHE
    load_dotenv()
    env = os.getenv(server.upper())
    env = [] if env is None else env.split(' ')
    if url is None:
        url = env[0] if len(env) > 0 else input('url:')
    if user is None:
        user = env[1] if len(env) > 1 else input('username:')
    if len(env) > 2:
        password = env[2]
        _SESSION_CACHE = {}
    elif _SESSION_CACHE.get((url, user)):
        password = _SESSION_CACHE[(url, user)]
    else:
        password = getpass('password:')
        _SESSION_CACHE[(url, user)] = password
    driver = os.getenv('JDBC_DRIVER')
    return (driver, url, [user, password])
