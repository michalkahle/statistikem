import os
import warnings
from getpass import getpass
import iris.dbapi # Requires intersystems-irispython v. 5.4.1 or higher!
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
_credentials = {}


def _connect(server, url, user, password):
    """Open an IRIS DB-API connection, resolving and caching credentials."""
    cached = _credentials.get(server, (None, None, None))
    env = os.getenv(f"ZLATOKOP_{server.upper()}", "").split() if server else []
    env += [None] * (3 - len(env))
    creds = (
        url      or cached[0] or env[0] or input('url: '),
        user     or cached[1] or env[1] or input('username: '),
        password or cached[2] or env[2] or getpass('password: '),
    )
    conn = iris.dbapi.connect(connectionstr=creds[0], username=creds[1], password=creds[2])
    _credentials[server] = creds
    return conn


def read_sql(sql, server='analytics', parse_dates=None,
             url=None, user=None, password=None, cache=None, **kwargs):
    """Run a SQL query against a Zlatokop IRIS instance and return a DataFrame.

    Credentials are resolved in this order: explicit arguments, credentials
    cached earlier in this session, the ZLATOKOP_<SERVER> environment variable
    ("url username password", whitespace-separated, may come from .env), and
    finally an interactive prompt. Credentials are cached per server after a
    successful connection; call _credentials.clear() to forget them.

    Parameters
    ----------
    sql : str
        Query to execute.
    server : str or None, default 'analytics'
        Suffix of the ZLATOKOP_<SERVER> environment variable. None skips the
        environment lookup.
    parse_dates : list or dict, optional
        Passed to pandas.read_sql_query.
    url : str, optional
        Connection string, e.g. 'host:1972/NAMESPACE'.
    user, password : str, optional
        Login credentials.
    cache : str, optional
        Path to a pickle cache file. When it exists, it is loaded instead of
        querying the database; otherwise the result is written there. Delete
        the file to force a refresh.
    **kwargs
        Passed to pandas.read_sql_query.

    Returns
    -------
    pandas.DataFrame
    """
    if cache is not None and os.path.exists(cache):
        return pd.read_pickle(cache)
    with _connect(server, url, user, password) as conn, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_sql_query(sql, conn, parse_dates=parse_dates, **kwargs)
    if cache is not None:
        df.to_pickle(cache)
    return df


def execute_sql(sql, server='analytics', url=None, user=None, password=None):
    """Execute a modifying SQL statement (INSERT, UPDATE, CREATE, ...) and commit.

    Credentials are resolved as in `read_sql`. Returns the number of rows
    affected, or 0 if the driver does not report it.
    """
    with _connect(server, url, user, password) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return max(cursor.rowcount or 0, 0)
