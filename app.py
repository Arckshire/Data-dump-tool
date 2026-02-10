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
    original_cols = list(df.columns)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    candidates = {
        "db_name": ["db_name", "table_catalog", "catalog", "database", "database_name"],
        "schema_name": ["schema_name", "table_schema", "schema"],
        "table_name": ["table_name", "table"],
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
            "CSV headers not recognized. Expected columns like "
            "db_name/table_catalog, schema_name/table_schema, table_name, "
            "ordinal_position, column_name, data_type.\n\n"
            f"Found headers: {original_cols}"
        )

    rename_map = {resolved[k]: k for k in resolved}
    df = df.rename(columns=rename_map)

    df["ordinal_position"] = pd.to_numeric(df["ordinal_position"], errors="coerce")
    for c in ["db_name", "schema_name", "table_name", "column_name", "data_type"]:
        df[c] = df[c].astype(str)
    if "is_nullable" in df.columns:
        df["is_nullable"] = df["is_nullable"].astype(str)

    return df

# ---------------- UI ----------------
st.title("Snowflake Self-Serve Query Builder (Dev)")
st.caption("Dev mode: Upload exported metadata CSV. App generates SQL for copy/paste into Snowflake.")

with st.sidebar:
    st.header("Metadata")
    uploaded = st.file_uploader("Upload metadata CSV", type=["csv"])
    st.caption("This avoids committing company table structure to a public repo.")

if not uploaded:
    st.info("Upload your metadata CSV to begin.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    meta = standardize_metadata_columns(raw)
except Exception as e:
    st.error(f"Could not read/normalize your CSV.\n\nError: {e}")
    st.stop()

# Full table key used for SQL
meta["full_table"] = meta["db_name"] + "." + meta["schema_name"] + "." + meta["table_name"]

# Display label = table name (with auto-disambiguation if duplicates exist)
table_counts = meta.groupby("table_name")["full_table"].nunique().to_dict()

def display_label(row):
    # If table name is unique across all schemas, show just TABLE
    # If not, show TABLE (SCHEMA) to avoid ambiguity
    if table_counts.get(row["table_name"], 0) <= 1:
        return row["table_name"]
    return f'{row["table_name"]} ({row["schema_name"]})'

table_map_df = (
    meta[["db_name", "schema_name", "table_name", "full_table"]]
    .drop_duplicates()
    .copy()
)
table_map_df["display"] = table_map_df.apply(display_label, axis=1)

# Sort labels nicely
table_map_df = table_map_df.sort_values(["table_name", "schema_name", "db_name"])
display_options = table_map_df["display"].tolist()
display_to_full = dict(zip(table_map_df["display"], table_map_df["full_table"]))

left, right = st.columns([0.35, 0.65], gap="large")

with left:
    st.subheader("1) Select table")

    selected_display = st.radio("Tables", options=display_options, index=0)
    selected_table = display_to_full[selected_display]

    st.markdown("---")
    disable_limit = st.checkbox("Disable row LIMIT (get all matching rows)", value=False)

    row_limit = None
    if not disable_limit:
        row_limit = st.number_input(
            "Row limit (safety)",
            min_value=1, max_value=5_000_000,
            value=DEFAULT_ROW_LIMIT,
            step=10000
        )
    else:
        st.warning("LIMIT is disabled. A date range filter will be required.")

    # Show full table faintly (optional). If you want to hide fully, delete this line.
    st.caption(f"Using: `{selected_table}`")

tcols = meta[meta["full_table"] == selected_table].sort_values("ordinal_position").copy()
tcols["data_family"] = tcols["data_type"].apply(normalize_dtype)

datetime_cols = tcols.loc[tcols["data_family"] == "datetime", "column_name"].tolist()
default_date_col = datetime_cols[0] if datetime_cols else None

with right:
    st.subheader("2) Pick columns + filters")

    with st.expander("Columns in selected table", expanded=True):
        st.dataframe(
            tcols[["ordinal_position", "column_name", "data_type", "data_family"]],
            use_container_width=True,
            hide_index=True
        )

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

    # Mandatory date filter when limit disabled
    if disable_limit:
        if not default_date_col:
            st.error("LIMIT is disabled, but this table has no DATE/TIMESTAMP columns. Re-enable LIMIT or choose a different table.")
            st.stop()

        st.info(
            f"Because LIMIT is disabled, a date filter is required.\n\n"
            f"Default date column selected: **{default_date_col}**. You can change it below."
        )

        chosen_date_col = st.selectbox(
            "Required date column",
            options=datetime_cols,
            index=0,
            key="required_date_col"
        )

        if chosen_date_col == default_date_col:
            st.caption("Using the default first date-type column. Change it if you want a different date field.")

        st.write(f"**Required date filter** · `{chosen_date_col}`")
        dt_mode = st.selectbox("Date filter type", ["between", "before", "after", "on"], key="req_dt_mode")

        required_date_payload = {"column_name": chosen_date_col, "data_family": "datetime", "mode": dt_mode}
        if dt_mode == "between":
            start = st.date_input("Start date", key="req_dt_start")
            end = st.date_input("End date", key="req_dt_end")
            required_date_payload["start"] = start.isoformat() if isinstance(start, date) else None
            required_date_payload["end"] = end.isoformat() if isinstance(end, date) else None
        else:
            d = st.date_input("Date", key="req_dt_single")
            required_date_payload["value"] = d.isoformat() if isinstance(d, date) else None

        st.markdown("---")

    filter_cols = st.multiselect(
        "Choose additional columns to filter on (each chosen filter must have a value)",
        options=all_col_names,
        default=[],
        key="optional_filter_cols"
    )

    filters_payload = []
    if disable_limit:
        filters_payload.append(required_date_payload)

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

        limit_sql = "" if disable_limit else f"\nLIMIT {int(row_limit)}"

        sql = f"""SELECT {select_sql}
FROM {selected_table}
{where_sql}{limit_sql};
"""
        st.success("SQL generated. Copy/paste into Snowflake.")
        st.code(sql, language="sql")
