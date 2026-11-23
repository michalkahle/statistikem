import glob
import os
import re
import warnings
import zipfile
from getpass import getpass
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
_credentials = {}
_JAR_DIR = os.path.join(os.path.dirname(__file__), 'jars')


def _native_available():
    """True if intersystems-irispython >= 5.4.2 is installed (older versions fall back to JDBC)."""
    try:
        return tuple(int(x) for x in re.findall(r'\d+', version('intersystems-irispython'))[:3]) >= (5, 4, 2)
    except PackageNotFoundError:
        return False


NATIVE = _native_available()
if NATIVE:
    import iris.dbapi
else:
    import jaydebeapi


def _jar_and_driver():
    jars = glob.glob(os.path.join(_JAR_DIR, '*.jar'))
    if not jars:
        raise FileNotFoundError(f'No JDBC driver jar found in {_JAR_DIR}')
    jar = jars[0]
    with zipfile.ZipFile(jar) as z:
        declared = z.read('META-INF/services/java.sql.Driver').decode()
    driver = next(line.strip() for line in declared.splitlines()
                  if line.strip() and not line.startswith('#'))
    return jar, driver


def _connect(server, url, user, password):
    cached = _credentials.get(server, (None, None, None))
    env = os.getenv(f"ZLATOKOP_{server.upper()}", "").split() if server else []
    env += [None] * (3 - len(env))
    creds = (
        url      or cached[0] or env[0] or input('url: '),
        user     or cached[1] or env[1] or input('username: '),
        password or cached[2] or env[2] or getpass('password: '),
    )
    if NATIVE:
        conn = iris.dbapi.connect(connectionstr=creds[0], username=creds[1], password=creds[2])
    else:
        jar, driver = _jar_and_driver()
        conn = jaydebeapi.connect(driver, f'jdbc:IRIS://{creds[0]}', [creds[1], creds[2]], jar)
    _credentials[server] = creds
    return conn


def read_sql(sql, server='analytics', parse_dates=None,
             url=None, user=None, password=None, cache=None, **kwargs):
    """Run a SQL query against a Zlatokop IRIS instance and return a DataFrame.

    Connects with the native driver if intersystems-irispython >= 5.4.2 is
    installed, otherwise through JDBC (jaydebeapi and the vendored jar).

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
        Path to a cache file ending in `.csv`, `.pkl` or `.pickle`. When given,
        the query is executed once and the result written to this path;
        subsequent calls load the file directly, bypassing the database. Delete
        the file to force a refresh. The query text is saved next to it as
        `<cache>.sql`; a warning is issued when the cached result comes from
        a different query.
    **kwargs
        Passed to pandas.read_sql_query (on a cache miss only).

    Returns
    -------
    pandas.DataFrame
    """
    if cache is not None:
        ext = os.path.splitext(cache)[1].lower()
        if ext not in ('.csv', '.pkl', '.pickle'):
            raise ValueError(f"cache must end in .csv, .pkl or .pickle, got {cache!r}")
        if os.path.exists(cache):
            if os.path.exists(cache + '.sql'):
                with open(cache + '.sql') as f:
                    if f.read() != sql:
                        warnings.warn(f'{cache} was produced by a different query. Delete it to refresh.')
            return pd.read_csv(cache, parse_dates=parse_dates) if ext == '.csv' else pd.read_pickle(cache)
    with _connect(server, url, user, password) as conn, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_sql_query(sql, conn, parse_dates=parse_dates, **kwargs)
    if cache is not None:
        df.to_csv(cache, index=False) if ext == '.csv' else df.to_pickle(cache)
        with open(cache + '.sql', 'w') as f:
            f.write(sql)
    return df


def execute_sql(sql, server, url=None, user=None, password=None):
    """Execute a modifying SQL statement (INSERT, UPDATE, CREATE, ...) and commit.

    `server` is required so a modifying statement never goes to a default
    database by accident. Credentials are resolved as in `read_sql`. Returns the number of rows
    affected, or 0 if the driver does not report it.
    """
    with _connect(server, url, user, password) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return max(cursor.rowcount or 0, 0)
