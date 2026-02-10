import re
import pandas as pd
import streamlit as st
from datetime import date

st.set_page_config(page_title="Snowflake Self-Serve Query Builder (Dev)", layout="wide")

DEFAULT_ROW_LIMIT = 100000

# ---------------- Helpers ----------------
def normalize_dtype(dtype: str) -> str:
    if not dtype:
        return "other"
    d = str(dtype).upper()
    if "TIMESTAMP" in d or d in {"DATE", "DATETIME", "TIME"}:
        return "datetime"
    if any(x in d for x in ["NUMBER", "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"]):
        return "number"
    if d == "BOOLEAN":
        return "boolean"
    if any(x in d for x in ["CHAR", "STRING", "TEXT", "VARCHAR"]):
        return "text"
    if any(x in d for x in ["ARRAY", "OBJECT", "VARIANT"]):
        return "semi_structured"
    return "other"

def is_valid_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))

def quote_ident(ident: str) -> str:
    if not ident:
        raise ValueError("Empty identifier")
    # Quote if not a simple identifier (keeps things safer with odd names)
    return f'"{ident}"' if not is_valid_identifier(ident) else ident

def build_where_clause(filters_payload):
    clauses = []
    errors = []

    for f in filters_payload:
        col = f["column_name"]
        family = f["data_family"]
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
            elif mode in {"before", "after", "on"}:
                val = f.get("value")
                if not val:
                    errors.append(f"Date value missing for {col}")
                    continue
                op = {"before": "<", "after": ">", "on": "="}[mode]
                clauses.append(f"{col_sql} {op} '{val}'")
            else:
                errors.append(f"Unknown datetime filter mode for {col}")
                continue

        elif family == "number":
            op = f.get("operator")
            if op in {"=", "!=", "<", "<=", ">", ">="}:
                val = f.get("value")
                if val is None or str(val).strip() == "":
                    errors.append(f"Numeric value missing for {col}")
                    continue
                # Keep numeric as-is; user must input numeric.
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
            op = f.get("operator")
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

def standardize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts many common Snowflake export header styles and normalizes to:
    db_name, schema_name, table_name, ordinal_position, column_name, data_type, is_nullable (optional)
    """
    # Lowercase headers for matching
    original_cols = list(df.columns)
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # Possible mappings from Snowflake exports
    candidates = {
        "db_name": ["db_name", "table_catalog", "table_catalog_name", "catalog", "database", "database_name"],
        "schema_name": ["schema_name", "table_schema", "schema", "schema_name_"],
        "table_name": ["table_name", "table", "table_name_"],
        "ordinal_position": ["ordinal_position", "position", "column_id", "column_ordinal_position"],
        "column_name": ["column_name", "column", "name"],
        "data_type": ["data_type", "type"],
        "is_nullable": ["is_nullable", "nullable"],
    }

    resolved = {}
    for target, options in candidates.items():
        for opt in options:
            if opt in df.columns:
                resolved[target] = opt
                break

    required = ["db_name", "schema_name", "table_name", "ordinal_position", "column_name", "data_type"]
    missing = [r for r in required if r not in resolved]
    if missing:
        raise ValueError(
            "CSV headers not recognized. I expected columns like "
            "db_name/table_catalog, schema_name/table_schema, table_name, "
            "ordinal_position, column_name, data_type.\n\n"
            f"Found headers: {original_cols}"
        )

    # Rename into standard names
    rename_map = {resolved[k]: k for k in resolved}
    df = df.rename(columns=rename_map)

    # Coerce types
    df["ordinal_position"] = pd.to_numeric(df["ordinal_position"], errors="coerce")
    for c in ["db_name", "schema_name", "table_name", "column_name", "data_type"]:
        df[c] = df[c].astype(str)

    # Optional nullable
    if "is_nullable" in df.columns:
        df["is_nullable"] = df["is_nullable"].astype(str)

    return df

# ---------------- UI ----------------
st.title("Snowflake Self-Serve Query Builder (Dev)")
st.caption("Dev mode: Upload exported metadata CSV. App generates SQL for copy/paste into Snowflake. No metadata is stored in GitHub.")

with st.sidebar:
    st.header("Metadata")
    uploaded = st.file_uploader("Upload metadata CSV (Snowflake columns export)", type=["csv"])
    st.caption("Tip: This keeps your repo public without publishing table/column structure.")

if not uploaded:
    st.info("Upload your metadata CSV to begin.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    meta = standardize_metadata_columns(raw)
except Exception as e:
    st.error(f"Could not read/normalize your CSV.\n\nError: {e}")
    st.stop()

meta["full_table"] = meta["db_name"] + "." + meta["schema_name"] + "." + meta["table_name"]
tables = sorted(meta["full_table"].unique().tolist())

left, right = st.columns([0.35, 0.65], gap="large")

with left:
    st.subheader("1) Select table")
    search = st.text_input("Search table", placeholder="Type to filter…")
    filtered_tables = [t for t in tables if search.lower() in t.lower()] if search else tables
    selected_table = st.selectbox("Table", filtered_tables, index=0 if filtered_tables else None)

    row_limit = st.number_input("Row limit (safety)", min_value=1, max_value=5_000_000, value=DEFAULT_ROW_LIMIT, step=10000)

    if not selected_table:
        st.info("Select a table to continue.")
        st.stop()

    st.markdown(f"**Selected:** `{selected_table}`")

tcols = meta[meta["full_table"] == selected_table].sort_values("ordinal_position")
tcols["data_family"] = tcols["data_type"].apply(normalize_dtype)

with right:
    st.subheader("2) Pick columns + filters")

    with st.expander("Columns in selected table", expanded=True):
        show_cols = ["ordinal_position", "column_name", "data_type", "data_family"]
        st.dataframe(tcols[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Output columns")
    all_col_names = tcols["column_name"].tolist()
    default_selected = all_col_names[: min(15, len(all_col_names))]

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
        st.caption("Fill every filter below. Remove a filter column if you don’t want to provide a value.")
        for col in filter_cols:
            row = tcols[tcols["column_name"] == col].iloc[0]
            family = row["data_family"]
            dtype = row["data_type"]

            st.write(f"**{col}**  ·  `{dtype}`")
            c1, c2 = st.columns([0.3, 0.7])

            with c1:
                if family == "datetime":
                    mode = st.selectbox(f"{col} filter type", ["between", "before", "after", "on"], key=f"{col}_dt_mode")
                elif family == "number":
                    mode = st.selectbox(f"{col} operator", ["=", "!=", "<", "<=", ">", ">=", "between"], key=f"{col}_num_op")
                elif family == "boolean":
                    mode = "equals"
                    st.markdown("Operator: `=`")
                else:
                    mode = st.selectbox(f"{col} operator", ["equals", "contains", "in"], key=f"{col}_txt_op")

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
                        payload["start"] = st.text_input(f"{col} min", key=f"{col}_num_min", placeholder="e.g. 10")
                        payload["end"] = st.text_input(f"{col} max", key=f"{col}_num_max", placeholder="e.g. 100")
                    else:
                        payload["value"] = st.text_input(f"{col} value", key=f"{col}_num_val", placeholder="e.g. 123")

                elif family == "boolean":
                    payload["value"] = st.selectbox(f"{col} value", [True, False], key=f"{col}_bool_val")

                else:
                    payload["operator"] = mode
                    if mode == "in":
                        vals = st.text_area(f"{col} values (one per line)", key=f"{col}_in_vals", placeholder="value1\nvalue2\nvalue3")
                        payload["values_list"] = [v for v in vals.splitlines() if v.strip()]
                    else:
                        payload["value"] = st.text_input(f"{col} value", key=f"{col}_txt_val")

            filters_payload.append(payload)
            st.markdown("---")

    if st.button("Generate SQL", type="primary"):
        where_sql, errors = build_where_clause(filters_payload)
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
