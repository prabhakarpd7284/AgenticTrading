# -*- coding: utf-8 -*-

# ==============================
# Imports
# ==============================
import os
import time
import pandas as pd
from datetime import datetime

import pyotp
from logzero import logger
from SmartApi import SmartConnect

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Any, Dict, List, Set
import mplfinance as mpf
# ==============================
# Environment
# ==============================
load_dotenv()  # load .env variables



"""
Complete, cleaned‑up version of the original notebook.

All components – broker interaction, data enrichment, LLM tooling, and
agent graph – are assembled in a single file for ease of use.
"""
# Global placeholder for the SmartConnect instance – will be set in __main__
smartApi = None
token_fetcher = None
df: pd.DataFrame | None = None

# ==============================
# Helpers
# ==============================
def load_symbol_master(file_path: str) -> pd.DataFrame:
    """Load symbol master data from a JSON file."""
    return pd.read_json(file_path).T.reset_index()


# ==============================
# Token fetcher
# ==============================
class TokenFetcher:
    """Retrieve exToken values for NSE tickers."""

    def __init__(self, symbol_master_df: pd.DataFrame):
        self.symbol_master_df = symbol_master_df

    def get_symbol_token(self, sym_ticker: str) -> str | None:
        """Return the exToken for a given ticker."""
        try:
            logger.info(f"Fetching symbol token for {sym_ticker}")
            return self.symbol_master_df.loc[
                self.symbol_master_df["symTicker"] == sym_ticker, "exToken"
            ].iloc[0]
        except Exception as e:
            logger.error(f"Token fetch failed for {sym_ticker}: {e}")
            return None

    def get_symbol_token_list(self, stocks_list: List[str]) -> Dict[str, str | None]:
        """Return a dict of ticker → exToken for a list of tickers."""
        return {stock: self.get_symbol_token(stock) for stock in stocks_list}


# ==============================
# Broker client
# ==============================
class BrokerClient:
    """
    Wrap SmartConnect and handle authentication / rate limiting.
    """

    def __init__(
        self,
        api_key: str,
        username: str,
        password: str,
        totp_secret: str,
    ):
        self.smart_api = SmartConnect(api_key)
        global smartApi
        smartApi = self.smart_api

        self.username = username
        self.password = password
        self.totp_secret = totp_secret

    def login(self) -> str:
        """Authenticate with the broker."""
        totp_code = pyotp.TOTP(self.totp_secret).now()
        try:
            self.smart_api.generateSession(self.username, self.password, totp_code)
            logger.info("Broker login successful.")
            smartApi = self.smart_api
            return "Login Successful!"
        except Exception as exc:
            logger.error(f"Broker login failed: {exc}")
            raise

    @staticmethod
    def _convert_date_to_string(dt: datetime) -> str:
        """Convert a datetime to the format expected by SmartConnect."""
        return dt.strftime("%Y-%m-%d %H:%M")

    def fetch_historic(
        self,
        exchange: str,
        symbol_token: str,
        start: str,
        end: str,
        interval: str = "ONE_DAY",
    ) -> List[Dict[str, Any]]:
        """Retrieve historic candle data for a symbol."""
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": start,
            "todate": end,
        }
        try:
            response = self.smart_api.getCandleData(params)
            logger.debug("Historic data retrieved: %s items", len(response["data"]))
            time.sleep(0.2)  # guard against rate limits
            return response["data"]
        except Exception as exc:
            logger.exception(f"Historic API failed: {exc}")
            raise

    def fetch_historic_datetime(
        self,
        exchange: str,
        symbol_token: str,
        start_dt: datetime,
        end_dt: datetime,
        interval: str = "ONE_DAY",
    ) -> List[Dict[str, Any]]:
        """Same as fetch_historic but accepts datetime objects."""
        start = self._convert_date_to_string(start_dt)
        end = self._convert_date_to_string(end_dt)
        return self.fetch_historic(exchange, symbol_token, start, end, interval)

def broker_login():
    try:
        broker_client = BrokerClient(
            api_key=os.getenv("SMARTAPI_KEY"),
            username=os.getenv("SMARTAPI_USERNAME"),
            password=os.getenv("SMARTAPI_PASSWORD"),
            totp_secret=os.getenv("SMARTAPI_TOTP_SECRET"),
        )
        logger.info(broker_client.login())
        return broker_client
    except Exception as e:
        logger.error("Failed to initialize or login to BrokerClient: %s", e)

broker_client = broker_login()
token_fetcher = TokenFetcher(load_symbol_master(os.getenv("SYMBOL_MASTER_JSON")))
# ==============================
# Data enrichment
# ==============================
def add_new_high_low_indicators(data: List) -> pd.DataFrame:
    """
    Enrich OHLCV data with new‑high / new‑low indicators and related metrics.

    Input is a list of 6‑tuples:
        (timestamp, open, high, low, close, volume)
    """
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame([dict(zip(cols, row)) for row in data])

    # Cumulative high/low
    df["cumulative_high"] = df["high"].cummax()
    df["cumulative_low"] = df["low"].cummin()

    # Flags for new high / low
    df["new_high"] = df["high"] == df["cumulative_high"]
    df["new_low"] = df["low"] == df["cumulative_low"]
    # Force the first candle to be *neither* a new high nor a new low
    df.at[0, "new_high"] = False
    df.at[0, "new_low"] = False

    # Range & percent
    df["range"] = df["cumulative_high"] - df["cumulative_low"]
    first_close = df["close"].iloc[0]
    first_open = df["open"].iloc[0]
    df["range_percent"] = (df["range"] / first_close) * 100 if first_close else pd.NA

    # Drawdowns (percent)
    df["high_drawdown"] = 100 * (df["cumulative_high"] - df["close"]).abs() / first_open
    df["low_drawdown"] = 100 * (df["cumulative_low"] - df["close"]).abs() / first_open

    # Open‑high / open‑low / doji / pivot
    df["OH"] = df["open"] == df["high"]
    df["OL"] = df["open"] == df["low"]
    df["doji"] = (df["open"] - df["close"]).abs() < 0.10
    df["pivot"] = (df["low"] + df["high"]) / 2

    return df.copy()

# ==============================
# Broker data retrieval (low‑level)
# ==============================
def convert_date_to_string(dt: datetime) -> str:
    """Return a string in the format expected by SmartConnect."""
    return dt.strftime("%Y-%m-%d %H:%M")


from dateutil import parser

def convert_string_to_date(date_str: str) -> datetime:
    """Parse a SmartConnect date string without an explicit format."""
    return parser.parse(date_str)


def fetch_intraday_data(
    symbol: str,
    start_date: datetime,
    interval: str = "FIVE_MINUTE",
    exchange: str = "NSE",
) -> pd.DataFrame:
    """
    Retrieve the most recent full trading day for *symbol* from the broker
    and enrich it with high/low indicators.
    """
    # start = datetime(2025, 10, 30, 0, 0)
    start=convert_string_to_date(start_date)
    end = start + pd.DateOffset(days=1)

    raw_data = fetch_hdata_from(
        exchange=exchange,
        symboltoken=token_fetcher.get_symbol_token(f"{exchange}:{symbol}-EQ"),
        start=convert_date_to_string(start),
        end=convert_date_to_string(end),
        interval=interval,
    )
    return add_new_high_low_indicators(raw_data)


def fetch_hdata_from(
    exchange: str,
    symboltoken: str,
    start: str,
    end: str,
    interval: str,
) -> List:
    """
    Wrapper around SmartConnect.getCandleData that accepts string dates.
    """
    params = {
        "exchange": exchange,
        "symboltoken": symboltoken,
        "interval": interval,
        "fromdate": start,
        "todate": end,
    }
    try:
        response = broker_client.smart_api.getCandleData(params)
        logger.debug("Fetched %s candles", len(response["data"]))
        time.sleep(0.2)  # rate‑limit guard
        return response["data"]
    except Exception as exc:
        logger.exception(f"fetch_hdata_from failed: {exc}")
        raise



# ------------------------------------------------------------------
# State definition for the agent
# ------------------------------------------------------------------
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function in the
    # annotation defines how this state key should be updated
    messages: Annotated[list, add_messages]


@tool
def search(query: str) -> str:
    """Simple web search using Tavily."""
    client = TavilySearchResults()
    results = client.run(query)
    return results


@tool
def ask_lm(question: str) -> str:
    """Ask the OpenAI‑like LLM and return the answer."""
    return llm.invoke(question)

@tool
def fetch_portfolio_from_broker() -> Any:
    """
    Fetch portfolio from broker.

    This tool simply forwards a request to the broker API's `holding`
    endpoint.  It assumes that the global ``smartApi`` variable has
    already been populated with a ``SmartConnect`` instance (see
    ``__main__`` for where this is done).
    """
    if smartApi is None:
        raise RuntimeError("Broker client not yet initialised.")
    # The SmartApi wrapper exposes a ``holding`` method that returns
    # the current portfolio.
    return smartApi.holding()


@tool
def fetch_data_from_broker(symbol: str, interval: str, start_date: str, end_date: str):
    """
    Retrieve OHLCV candle data from broker and add high/low indicators. 
    "    :param symbol: stock symbol in capital letters,
    "    :param interval: FIVE_MINUTE, ONE_HOUR, ONE_DAY etc\n",
    "    :param start_date: str in '%Y-%m-%d %H:%M' format. start date should be before the end date by 1 day for intraday data.\n",
    "    :param end_date: str in '%Y-%m-%d %H:%M' format\n",
    "    :return: 2D array with timestamp, open, high, low, close, volume\n",
    """
    data = smartApi.getCandleData({
        "exchange": 'NSE',
        "symboltoken": token_fetcher.get_symbol_token(f"NSE:{symbol}-EQ"),
        "interval": interval,
        "fromdate": start_date,
        "todate": end_date,
    })['data']
    return add_new_high_low_indicators(data)


@tool
def add_high_low_indicators(data):
    """
    Enrich the OHLCV data with high/low and drawdown indicators.
    """
    return add_new_high_low_indicators(data)

@tool
def fetch_symbol_token_tool(symbol):
    """
    Retrieve exToken values for NSE tickers. Required to fetch data from broker.
    """
    return token_fetcher.get_symbol_token(symbol)

@tool
def fetch_intraday_data_tool(symbol, start_date):
    """
    Fetch intraday OHLCV enriched with day high and low indicators and readings.
    "   :param symbol: stock symbol in capital letters
    "   :param start_date: str in '%Y-%m-%d %H:%M' format.
    "   :return: 2D array with timestamp, open, high, low, close, volume",

    """
    global df
    df = fetch_intraday_data(symbol, start_date)
    return df

import os
import logging
from typing import Optional
from pydantic import BaseModel, Field

logging.basicConfig(level="INFO", format="%(message)s")
log = logging.getLogger("rich")


# ---------- Input Schemas ----------

class CreateFolderInput(BaseModel):
    path: str = Field(..., description="The path where the folder should be created")


class CreateFileInput(BaseModel):
    path: str = Field(..., description="The path where the file should be created")
    content: Optional[str] = Field("", description="Initial content of the file")


class WriteToFileInput(BaseModel):
    path: str = Field(..., description="The file path to write to")
    content: str = Field(..., description="Content to write to file")


class ReadFileInput(BaseModel):
    path: str = Field(..., description="The file path to read")


class ListFilesInput(BaseModel):
    path: Optional[str] = Field(".", description="Folder to list. Defaults to current directory.")


class DeleteFileInput(BaseModel):
    path: str = Field(..., description="File path to delete")


# ---------- Tools ----------
@tool
def create_folder(input: CreateFolderInput):
    """Create a new folder at the specified path."""
    full_path = os.path.abspath(input.path)
    try:
        os.makedirs(full_path, exist_ok=True)
        log.info(f"Created folder: {full_path}")
        return {"success": True, "message": f"Folder created at {full_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def create_file(input: CreateFileInput):
    """Create a new file with optional content."""
    full_path = os.path.abspath(input.path)
    try:
        with open(full_path, "w") as f:
            f.write(input.content or "")
        log.info(f"Created file: {full_path}")
        return {"success": True, "message": f"File created at {full_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def write_to_file(input: WriteToFileInput):
    """Write content to an existing file."""
    full_path = os.path.abspath(input.path)
    try:
        with open(full_path, "w") as f:
            f.write(input.content)
        log.info(f"Wrote to file: {full_path}")
        return {"success": True, "message": f"Content written to {full_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def read_file(input: ReadFileInput):
    """Read the contents of a file."""
    full_path = os.path.abspath(input.path)
    try:
        with open(full_path, "r") as f:
            content = f.read()
        log.info(f"Read file: {full_path}")
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def list_files(input: ListFilesInput):
    """List files in a folder."""
    full_path = os.path.abspath(input.path)
    try:
        files = os.listdir(full_path)
        log.info(f"Listed files in: {full_path}")
        return {"success": True, "files": files}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def delete_file(input: DeleteFileInput):
    """Delete a file."""
    full_path = os.path.abspath(input.path)
    try:
        os.remove(full_path)
        log.info(f"Deleted file: {full_path}")
        return {"success": True, "message": f"File deleted: {full_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# LLM instance
llm = ChatOllama(model="gpt-oss:latest",
    validate_model_on_init=True,
    temperature=0)


# Bind tools to the LLM
tools = [fetch_intraday_data_tool, fetch_data_from_broker, fetch_portfolio_from_broker, create_file, write_to_file, read_file, list_files, delete_file, create_folder]
llm_with_tools = llm.bind_tools(tools)

# ------------------------------------------------------------------
# Graph definition
# ------------------------------------------------------------------
def tool_calling_llm(state: State):
    """
    Node that calls the LLM with the current state messages.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

# Compile the graph
graph = builder.compile()

# ------------------------------------------------------------------
# Visualize the graph (optional)
# ------------------------------------------------------------------
try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # If running outside a notebook, skip visualization
    pass


# ==============================
# Main demo
# ==============================
# if __name__ == "__main__":
    # Wire the global SmartConnect instance for the tool
    # global token_fetcher

    # ------------------------------------------------------------------
    # 1. Load symbol master and initialise helpers
    # ------------------------------------------------------------------
    # symbol_master_path = os.getenv("SYMBOL_MASTER_JSON")
    # if not symbol_master_path:
    #     raise RuntimeError("Set SYMBOL_MASTER_JSON in .env")

    # symbol_master_df = load_symbol_master(symbol_master_path)
    
    # token_fetcher = TokenFetcher(symbol_master_df)

    # ------------------------------------------------------------------
    # 2. Broker client (authentication)
    # ------------------------------------------------------------------
    # broker_client = broker_login()
    # global smartApi
    # smartApi = broker_client.smart_api  # ensure global variable is set

    # ------------------------------------------------------------------
    # 3. Fetch some broker data
    # ------------------------------------------------------------------
    # end_date = datetime(2025, 10, 31, 0, 0)
    # start_date = end_date - pd.DateOffset(days=1)

    # raw_data = fetch_hdata_from(
    #     exchange="NSE",
    #     symboltoken=token_fetcher.get_symbol_token("NSE:MFSL-EQ"),
    #     start=convert_date_to_string(start_date),
    #     end=convert_date_to_string(end_date),
    #     interval="FIVE_MINUTE",
    # )

    # ------------------------------------------------------------------
    # 4. Enrich the data with indicators
    # ------------------------------------------------------------------
    # df = add_new_high_low_indicators(raw_data)
    # logger.info("Data enriched – %s rows, %s columns", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # 5. Run the agent on a sample query
    # ------------------------------------------------------------------
    # sample_query = "What is the intraday movement for stock MFSL on 31st Oct 2025.? Give initial bias based on enriched data?"
    # print("\n=== Query : ", sample_query)

    # response = graph.invoke({"messages": sample_query})
    # print("\n=== Agent output ===")
    # for msg in response["messages"]:
    #     msg.pretty_print()

    # ------------------------------------------------------------------
    # 6. (Optional) Plot new highs
    # ------------------------------------------------------------------
    # new_highs = df[df["new_high"]]
    # print(f"Number of new highs: {len(new_highs)}")