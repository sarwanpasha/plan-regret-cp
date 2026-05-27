"""
Benchmark schema definitions: table lists, join graphs, predicate templates.

Both TPC-H and TPC-DS are generated programmatically inside DuckDB via its
built-in extensions, so no external data download is required.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BenchmarkSchema:
    """Schema for one benchmark."""
    name: str
    tables: List[str]
    # Join graph as (table_a, col_a, table_b, col_b) tuples.
    join_graph: List[Tuple[str, str, str, str]]
    # Predicate column templates per table:
    #   numeric: (col_name, "num", lo, hi)
    #   categorical: (col_name, "cat", [list of values])
    predicate_cols: Dict[str, list] = field(default_factory=dict)

    @property
    def edge_keys(self) -> List[Tuple[str, str]]:
        return [(a, b) for (a, _, b, _) in self.join_graph]


# =============================================================
# TPC-H schema (full)
# =============================================================
TPCH = BenchmarkSchema(
    name="tpch",
    tables=["nation", "region", "supplier", "customer",
            "part", "partsupp", "orders", "lineitem"],
    join_graph=[
        ("nation",   "n_regionkey",  "region",   "r_regionkey"),
        ("supplier", "s_nationkey",  "nation",   "n_nationkey"),
        ("customer", "c_nationkey",  "nation",   "n_nationkey"),
        ("partsupp", "ps_partkey",   "part",     "p_partkey"),
        ("partsupp", "ps_suppkey",   "supplier", "s_suppkey"),
        ("orders",   "o_custkey",    "customer", "c_custkey"),
        ("lineitem", "l_orderkey",   "orders",   "o_orderkey"),
        ("lineitem", "l_partkey",    "part",     "p_partkey"),
        ("lineitem", "l_suppkey",    "supplier", "s_suppkey"),
    ],
    predicate_cols={
        "lineitem": [
            ("l_quantity",     "num", 1, 50),
            ("l_extendedprice","num", 900, 100000),
            ("l_discount",     "num", 0.0, 0.10),
            ("l_returnflag",   "cat", ["A", "N", "R"]),
        ],
        "orders": [
            ("o_totalprice",   "num", 800, 500000),
            ("o_orderstatus",  "cat", ["F", "O", "P"]),
            ("o_orderpriority","cat",
             ["1-URGENT","2-HIGH","3-MEDIUM","4-NOT SPECIFIED","5-LOW"]),
        ],
        "customer": [
            ("c_acctbal",      "num", -1000, 10000),
            ("c_mktsegment",   "cat",
             ["AUTOMOBILE","BUILDING","FURNITURE","HOUSEHOLD","MACHINERY"]),
        ],
        "part": [
            ("p_size",         "num", 1, 50),
            ("p_retailprice",  "num", 900, 2100),
        ],
        "supplier": [
            ("s_acctbal",      "num", -1000, 10000),
        ],
        "partsupp": [
            ("ps_supplycost",  "num", 1, 1000),
        ],
    },
)


# =============================================================
# TPC-DS schema (9-table subset spanning fact tables + core dims)
# =============================================================
TPCDS = BenchmarkSchema(
    name="tpcds",
    tables=["store_sales", "store_returns", "customer", "customer_address",
            "customer_demographics", "date_dim", "household_demographics",
            "item", "store"],
    join_graph=[
        ("store_sales",   "ss_customer_sk",      "customer",        "c_customer_sk"),
        ("store_sales",   "ss_cdemo_sk",         "customer_demographics", "cd_demo_sk"),
        ("store_sales",   "ss_hdemo_sk",         "household_demographics", "hd_demo_sk"),
        ("store_sales",   "ss_addr_sk",          "customer_address","ca_address_sk"),
        ("store_sales",   "ss_store_sk",         "store",           "s_store_sk"),
        ("store_sales",   "ss_item_sk",          "item",            "i_item_sk"),
        ("store_sales",   "ss_sold_date_sk",     "date_dim",        "d_date_sk"),
        ("store_returns", "sr_customer_sk",      "customer",        "c_customer_sk"),
        ("store_returns", "sr_item_sk",          "item",            "i_item_sk"),
        ("store_returns", "sr_returned_date_sk", "date_dim",        "d_date_sk"),
        ("customer",      "c_current_addr_sk",   "customer_address","ca_address_sk"),
        ("customer",      "c_current_cdemo_sk",  "customer_demographics","cd_demo_sk"),
        ("customer",      "c_current_hdemo_sk",  "household_demographics","hd_demo_sk"),
    ],
    predicate_cols={
        "store_sales": [
            ("ss_quantity",   "num", 1, 100),
            ("ss_sales_price","num", 0, 200),
            ("ss_net_profit", "num", -10000, 20000),
        ],
        "store_returns": [
            ("sr_return_quantity", "num", 1, 100),
            ("sr_return_amt",      "num", 0, 20000),
        ],
        "customer": [
            ("c_birth_year", "num", 1924, 1992),
        ],
        "customer_address": [
            ("ca_state", "cat",
             ["AL","AK","AZ","CA","CO","FL","GA","IL","NY","OH","PA","TX","WA"]),
        ],
        "customer_demographics": [
            ("cd_gender",         "cat", ["M","F"]),
            ("cd_marital_status", "cat", ["S","M","D","W","U"]),
            ("cd_education_status","cat",
             ["Primary","Secondary","College","Advanced Degree","Unknown"]),
        ],
        "date_dim": [
            ("d_year",  "num", 1998, 2002),
            ("d_qoy",   "num", 1, 4),
            ("d_month_seq", "num", 1180, 1240),
        ],
        "household_demographics": [
            ("hd_dep_count", "num", 0, 9),
        ],
        "item": [
            ("i_current_price", "num", 0.0, 100.0),
            ("i_category",      "cat",
             ["Books","Electronics","Sports","Home","Children","Music","Jewelry"]),
        ],
        "store": [
            ("s_state", "cat", ["TN","KY","OH","IL","CA","WA"]),
        ],
    },
)


def get_schema(name: str) -> BenchmarkSchema:
    """Look up a schema by name (case-insensitive)."""
    n = name.lower()
    if n in ("tpch", "tpc-h"):
        return TPCH
    if n in ("tpcds", "tpc-ds"):
        return TPCDS
    raise ValueError(f"unknown benchmark: {name}")
