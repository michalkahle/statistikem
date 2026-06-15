import os
import warnings
from getpass import getpass

import jaydebeapi
import pandas as pd
from dotenv import load_dotenv

_SESSION_CACHE = {}


def read_sql(sql, server='jdbc_analytics', parse_dates=None, url=None, user=None, **kwargs):
    """Run a SQL query over JDBC and return the result as a DataFrame.

    Credentials are resolved from the environment variable named after `server`
    (uppercased), expected to contain space-separated `url user [password]`. Any
    missing piece is prompted for interactively; passwords entered at the prompt
    are cached in memory for the rest of the session. The JDBC driver path is
    read from the `JDBC_DRIVER` environment variable. A `.env` file in the
    working directory is loaded automatically.

    Parameters
    ----------
    sql : str
        SQL query to execute.
    server : str
        Name of the environment variable holding the connection string.
        Default `'jdbc_analytics'`.
    parse_dates : list or dict, optional
        Forwarded to `pandas.read_sql_query` to coerce columns to datetimes.
    url, user : str, optional
        Override the connection URL or username from the environment.
    **kwargs
        Forwarded to `pandas.read_sql_query`.

    Returns
    -------
    pandas.DataFrame
    """
    with jaydebeapi.connect(*_get_creds(server, url, user)) as connection:
        with warnings.catch_warnings(action='ignore'):
            return pd.read_sql_query(sql, connection, parse_dates=parse_dates, **kwargs)


def read_sql_cached(sql, filename, server='jdbc_analytics', parse_dates=None, **kwargs):
    """Run `read_sql` once and cache the result as a CSV on disk.

    On the first call the query is executed and the result is written to
    `filename`; on subsequent calls the CSV is loaded directly, bypassing the
    database. Delete the file to force a refresh. `parse_dates` is applied on
    both paths so dtypes survive the round-trip through CSV.

    Parameters
    ----------
    sql : str
        SQL query to execute when the cache file is absent.
    filename : str
        Path to the CSV cache file.
    server : str
        Name of the environment variable holding the connection string.
        Default `'jdbc_analytics'`.
    parse_dates : list or dict, optional
        Columns to parse as datetimes, forwarded to `read_sql` on a cache miss
        and to `pandas.read_csv` on a cache hit.
    **kwargs
        Forwarded to `read_sql` on a cache miss only.

    Returns
    -------
    pandas.DataFrame
    """
    if not os.path.exists(filename):
        df = read_sql(sql, server, parse_dates=parse_dates, **kwargs)
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename, parse_dates=parse_dates)
    return df



def execute_sql(sql, server=None, parse_dates=None, url=None, user=None):
    """Execute a modifying SQL statement over JDBC and commit the transaction.

    Intended for `INSERT`, `UPDATE`, `DELETE`, `CREATE`, and similar statements
    that change state rather than return rows. Use `read_sql` for `SELECT`.

    Credentials are resolved from the environment variable named after `server`
    (uppercased), expected to contain space-separated `url user [password]`. Any
    missing piece is prompted for interactively; passwords entered at the prompt
    are cached in memory for the rest of the session. The JDBC driver path is
    read from the `JDBC_DRIVER` environment variable. A `.env` file in the
    working directory is loaded automatically.

    Parameters
    ----------
    sql : str
        SQL statement to execute.
    server : str
        Name of the environment variable holding the connection string.
    url, user : str, optional
        Override the connection URL or username from the environment.

    Returns
    -------
    int
        Number of rows affected, or 0 if the driver does not report it.
    """
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
