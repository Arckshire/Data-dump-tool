import re
import pandas as pd
import streamlit as st
from datetime import date, datetime

st.set_page_config(page_title="Snowflake Self-Serve Query Builder (Dev)", layout="wide")

# ---------- Config ----------
METADATA_CSV_PATH = "data/table_columns_catalog.csv"
DEFAULT_ROW_LIMIT = 100000

# ---------- Helpers ----------
def normalize_dtype(dtype: str) -> str:
    """Map Snowflake data types into broad UI categories."""
    if not dtype:
        return "other"
    d = dtype.upper()

    if "TIMESTAMP" in d or d in {"DATE", "DATETIME", "TIME"}:
        return "datetime"
    if any(x in d for x in ["NUMBER", "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"]):
        return "number"
    if d in {"BOOLEAN"}:
        return "boolean"
    if any(x in d for x in ["CHAR", "STRING", "TEXT", "VARCHAR"]):
        return "text"
    if any(x in d for x in ["ARRAY", "OBJECT", "VARIANT"]):
        return "semi_structured"
    return "other"

def is_valid_identifier(name: str) -> bool:
    # Simple safety check; Snowflake identifiers can be quoted, but we’ll keep it strict for v1.
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))

def quote_ident(ident: str) -> str:
    """
    Safely quote column identifiers. We still validate to avoid injecting weird identifiers.
    If your metadata includes mixed-case or special chars, this quoting helps.
    """
    if not ident:
        raise ValueError("Empty identifier")
    # allow typical snowflake identifiers; if something odd appears, still quote it.
    return f'"{ident}"' if not is_valid_identifier(ident) else ident

def fq_table(db: str, schema: str, table: str) -> str:
    return f"{db}.{schema}.{table}"

def build_where_clause(filters_payload):
    """
    filters_payload: list of dicts with keys:
      - column_name
      - data_family: datetime/number/text/boolean/other
      - operator
      - value (or start/end for range)
    Returns: (where_sql, errors)
    """
    clauses = []
    errors = []

    for f in filters_payload:
        col = f["column_name"]
        family = f["data_family"]
        op = f.get("operator")
        col_sql = quote_ident(col)

        if family == "datetime":
            mode = f.get("mode")
            if mode == "between":
                start = f.get("start")
                end = f.get("end")
                if not start or not end:
                    errors.append(f"Date range missing for {col}")
                    continue
                clauses.append(f"{col_sql} BETWEEN '{start}' AND '{end}'")
            elif mode == "before":
                val = f.get("value")
                if not val:
                    errors.append(f"Date value missing for {col}")
                    continue
                clauses.append(f"{col_sql} < '{val}'")
            elif mode == "after":
                val = f.get("value")
                if not val:
                    errors.append(f"Date value missing for {col}")
                    continue
                clauses.append(f"{col_sql} > '{val}'")
            elif mode == "on":
                val = f.get("value")
                if not val:
                    errors.append(f"Date value missing for {col}")
                    continue
                clauses.append(f"{col_sql} = '{val}'")
            else:
                errors.append(f"Unknown datetime filter mode for {col}")
                continue

        elif family == "number":
            if op in {"=", "!=", "<", "<=", ">", ">="}:
                val = f.get("value")
                if val is None or val == "":
                    errors.append(f"Numeric value missing for {col}")
                    continue
                clauses.append(f"{col_sql} {op} {val}")
            elif op == "between":
                start = f.get("start")
                end = f.get("end")
                if start in (None, "") or end in (None, ""):
                    errors.append(f"Numeric range missing for {col}")
                    continue
                clauses.append(f"{col_sql} BETWEEN {start} AND {end}")
            else:
                errors.append(f"Unknown numeric operator for {col}")
                continue

        elif family == "boolean":
            val = f.get("value")
            if val is None:
                errors.append(f"Boolean value missing for {col}")
                continue
            clauses.append(f"{col_sql} = {str(val).upper()}")

        else:
            # Treat as text by default
            if op == "equals":
                val = f.get("value")
                if not val:
                    errors.append(f"Value missing for {col}")
                    continue
                escaped = str(val).replace("'", "''")
                clauses.append(f"{col_sql} = '{escaped}'")
            elif op == "contains":
                val = f.get("value")
                if not val:
                    errors.append(f"Value missing for {col}")
                    continue
                escaped = str(val).replace("'", "''")
                clauses.append(f"{col_sql} ILIKE '%{escaped}%'")
            elif op == "in":
                vals = f.get("values_list", [])
                if not vals:
                    errors.append(f"IN list missing for {col}")
                    continue
                safe = [str(v).strip().replace("'", "''") for v in vals if str(v).strip() != ""]
                if not safe:
                    errors.append(f"IN list empty for {col}")
                    continue
                joined = ", ".join([f"'{v}'" for v in safe])
                clauses.append(f"{col_sql} IN ({joined})")
            else:
                errors.append(f"Unknown text operator for {col}")
                continue

    if errors:
        return "", errors

    if not clauses:
        return "", []

    return "WHERE " + "\n  AND ".join(clauses), []

# ---------- Load metadata ----------
@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: db_name, schema_name, table_name, ordinal_position, column_name, data_type, is_nullable
    required = {"db_name", "schema_name", "table_name", "ordinal_position", "column_name", "data_type"}
    missing = required - set(df.columns.str.lower())
    # Some exports preserve case; normalize.
    df.columns = [c.lower() for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")

    # Coerce
    df["ordinal_position"] = pd.to_numeric(df["ordinal_position"], errors="coerce")
    df["data_type"] = df["data_type"].astype(str)
    df["column_name"] = df["column_name"].astype(str)
    return df

st.title("Snowflake Self-Serve Query Builder (Dev)")
st.caption("Dev mode: Uses exported metadata CSV and generates SQL for copy/paste into Snowflake.")

try:
    meta = load_metadata(METADATA_CSV_PATH)
except Exception as e:
    st.error(f"Could not load metadata CSV at '{METADATA_CSV_PATH}'.\n\nError: {e}")
    st.stop()

# Build table list
meta["full_table"] = meta["db_name"].astype(str) + "." + meta["schema_name"].astype(str) + "." + meta["table_name"].astype(str)
tables = sorted(meta["full_table"].unique().tolist())

# ---------- UI Layout ----------
left, right = st.columns([0.35, 0.65], gap="large")

with left:
    st.subheader("1) Select table")
    search = st.text_input("Search table", placeholder="Type to filter tables…")
    filtered_tables = [t for t in tables if search.lower() in t.lower()] if search else tables

    selected_table = st.selectbox("Table", filtered_tables, index=0 if filtered_tables else None)
    row_limit = st.number_input("Row limit (safety)", min_value=1, max_value=5_000_000, value=DEFAULT_ROW_LIMIT, step=10000)

    if not selected_table:
        st.info("Select a table to continue.")
        st.stop()

    db, schema, table = selected_table.split(".", 2)
    st.markdown(f"**Selected:** `{selected_table}`")

# Table columns
tcols = meta[meta["full_table"] == selected_table].sort_values("ordinal_position")
tcols["data_family"] = tcols["data_type"].apply(normalize_dtype)

with right:
    st.subheader("2) Pick columns + filters")

    with st.expander("Columns in selected table", expanded=True):
        st.dataframe(
            tcols[["ordinal_position", "column_name", "data_type", "data_family"]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Output column selection
    st.markdown("### Output columns")
    all_col_names = tcols["column_name"].tolist()
    default_selected = all_col_names[: min(15, len(all_col_names))]  # small default

    out_cols = st.multiselect(
        "Choose columns to export (leave empty for *)",
        options=all_col_names,
        default=default_selected
    )

    st.markdown("### Filters")
    filter_cols = st.multiselect(
        "Choose columns to filter on (each chosen filter must have a value)",
        options=all_col_names,
        default=[]
    )

    filters_payload = []
    if filter_cols:
        st.caption("Fill every filter below. If you don’t want a filter, remove it from the selection.")
        for col in filter_cols:
            row = tcols[tcols["column_name"] == col].iloc[0]
            family = row["data_family"]
            dtype = row["data_type"]

            st.write(f"**{col}**  ·  `{dtype}`")
            c1, c2 = st.columns([0.3, 0.7])

            with c1:
                if family == "datetime":
                    mode = st.selectbox(
                        f"{col} filter type",
                        options=["between", "before", "after", "on"],
                        key=f"{col}_dt_mode"
                    )
                elif family == "number":
                    mode = st.selectbox(
                        f"{col} operator",
                        options=["=", "!=", "<", "<=", ">", ">=", "between"],
                        key=f"{col}_num_op"
                    )
                elif family == "boolean":
                    mode = "equals"
                    st.markdown("Operator: `=`")
                else:
                    mode = st.selectbox(
                        f"{col} operator",
                        options=["equals", "contains", "in"],
                        key=f"{col}_txt_op"
                    )

            with c2:
                payload = {"column_name": col, "data_family": family}

                if family == "datetime":
                    payload["mode"] = mode
                    if mode == "between":
                        start = st.date_input(f"{col} start date", key=f"{col}_dt_start")
                        end = st.date_input(f"{col} end date", key=f"{col}_dt_end")
                        payload["start"] = start.isoformat() if isinstance(start, date) else None
                        payload["end"] = end.isoformat() if isinstance(end, date) else None
                    else:
                        d = st.date_input(f"{col} date", key=f"{col}_dt_single")
                        payload["value"] = d.isoformat() if isinstance(d, date) else None

                elif family == "number":
                    payload["operator"] = mode
                    if mode == "between":
                        start = st.text_input(f"{col} min", key=f"{col}_num_min", placeholder="e.g. 10")
                        end = st.text_input(f"{col} max", key=f"{col}_num_max", placeholder="e.g. 100")
                        payload["start"] = start
                        payload["end"] = end
                    else:
                        val = st.text_input(f"{col} value", key=f"{col}_num_val", placeholder="e.g. 123")
                        payload["value"] = val

                elif family == "boolean":
                    val = st.selectbox(f"{col} value", options=[True, False], key=f"{col}_bool_val")
                    payload["value"] = val

                else:
                    payload["operator"] = mode
                    if mode == "in":
                        vals = st.text_area(
                            f"{col} values (one per line)",
                            key=f"{col}_in_vals",
                            placeholder="value1\nvalue2\nvalue3"
                        )
                        payload["values_list"] = [v for v in vals.splitlines() if v.strip()]
                    else:
                        val = st.text_input(f"{col} value", key=f"{col}_txt_val")
                        payload["value"] = val

            filters_payload.append(payload)
            st.markdown("---")

    # Build query button
    build = st.button("Generate SQL", type="primary")

    if build:
        where_sql, errors = build_where_clause(filters_payload)

        # Enforce your rule: if a filter column is selected, it must have a value.
        if errors:
            st.error("Fix these before generating SQL:\n- " + "\n- ".join(errors))
            st.stop()

        select_sql = "*"
        if out_cols:
            select_sql = ", ".join([quote_ident(c) for c in out_cols])

        sql = f"""SELECT {select_sql}
FROM {selected_table}
{where_sql}
LIMIT {int(row_limit)};
"""

        st.success("SQL generated. Copy/paste into Snowflake.")
        st.code(sql, language="sql")
