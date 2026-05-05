import io
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests


EPID_WER_PAGE = "https://www.epid.gov.lk/weekly-epidemiological-report"
SOURCE_NAME = "Epidemiology Unit Weekly Epidemiological Report"

DISTRICT_ALIASES = {
    "Ampara": ["Ampara"],
    "Anuradhapura": ["Anuradhapura", "Anuradhapur"],
    "Badulla": ["Badulla"],
    "Batticaloa": ["Batticaloa"],
    "Colombo": ["Colombo"],
    "Galle": ["Galle"],
    "Gampaha": ["Gampaha"],
    "Hambantota": ["Hambantota"],
    "Jaffna": ["Jaffna"],
    "Kalutara": ["Kalutara"],
    "Kandy": ["Kandy"],
    "Kegalle": ["Kegalle"],
    "Kilinochchi": ["Kilinochchi"],
    "Kurunegala": ["Kurunegala"],
    "Mannar": ["Mannar"],
    "Matale": ["Matale"],
    "Matara": ["Matara"],
    "Monaragala": ["Monaragala", "Moneragala"],
    "Mullaitivu": ["Mullaitivu"],
    "NuwaraEliya": ["Nuwara Eliya", "NuwaraEliya"],
    "Polonnaruwa": ["Polonnaruwa"],
    "Puttalam": ["Puttalam"],
    "Ratnapura": ["Ratnapura", "Rathnapura"],
    "Trincomalee": ["Trincomalee"],
    "Vavuniya": ["Vavuniya"],
}


def _normalize_district(value):
    return re.sub(r"[^a-z]", "", str(value).lower())


NORMALIZED_DISTRICTS = {
    _normalize_district(alias): district
    for district, aliases in DISTRICT_ALIASES.items()
    for alias in aliases
}


def canonical_district_name(value):
    return NORMALIZED_DISTRICTS.get(_normalize_district(value), value)


def init_official_data_tables(db_path):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS official_dengue_cases (
            district TEXT NOT NULL,
            official_district_name TEXT,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            report_year INTEGER NOT NULL,
            report_week INTEGER NOT NULL,
            report_period TEXT,
            current_week_cases INTEGER NOT NULL,
            previous_week_cases INTEGER,
            cumulative_cases INTEGER,
            pulled_at DATETIME NOT NULL,
            PRIMARY KEY (district, source, report_year, report_week)
        )
        """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_official_dengue_latest
        ON official_dengue_cases (district, report_year DESC, report_week DESC)
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS official_data_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pulled_at DATETIME NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT,
            status TEXT NOT NULL,
            rows_imported INTEGER NOT NULL,
            message TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _request_get(url, *, timeout=30, binary=False):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
        )
    }
    session = requests.Session()
    response = session.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.content if binary else response.text


def fetch_epid_report_links(limit=3):
    html = _request_get(EPID_WER_PAGE, timeout=30)
    links = []
    seen = set()

    for match in re.finditer(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, re.I):
        url = urljoin(EPID_WER_PAGE, match.group(1))
        if url in seen:
            continue
        seen.add(url)

        filename = Path(url.split("?", 1)[0]).name.lower()
        if "vol_" not in filename and "vol-" not in filename:
            continue

        links.append(url)
        if len(links) >= limit:
            break

    return links


def _extract_pdf_text(pdf_bytes):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("pypdf is required to import official dengue PDFs") from error

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_report_metadata(text, source_url):
    normalized = re.sub(r"\s+", " ", text)

    year_match = re.search(r"\b(20\d{2})\b", normalized)
    period_match = re.search(
        r"(\d{1,2}(?:st|nd|rd|th)?\s*[–-]\s*"
        r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+20\d{2})",
        normalized,
    )
    table_week_match = re.search(
        r"\((\d{1,2})(?:st|nd|rd|th)?\s+Week\)",
        normalized,
        re.I,
    )
    volume_week_match = re.search(r"Vol\.\s*\d+\s+No\.\s*(\d{1,2})", normalized, re.I)

    if period_match:
        report_year = int(re.search(r"(20\d{2})", period_match.group(1)).group(1))
    else:
        report_year = int(year_match.group(1)) if year_match else datetime.now().year
    report_week = None
    if table_week_match:
        report_week = int(table_week_match.group(1))
    elif volume_week_match:
        report_week = int(volume_week_match.group(1))
    else:
        report_week = datetime.now().isocalendar().week

    return {
        "source": SOURCE_NAME,
        "source_url": source_url,
        "report_year": report_year,
        "report_week": report_week,
        "report_period": period_match.group(1) if period_match else None,
    }


def parse_epid_wer_pdf(pdf_bytes, source_url):
    text = _extract_pdf_text(pdf_bytes)
    metadata = _parse_report_metadata(text, source_url)

    table_start = text.find("RDHS")
    table_text = text[table_start:] if table_start >= 0 else text

    rows = []
    pulled_at = datetime.now().isoformat()
    for district, aliases in DISTRICT_ALIASES.items():
        match = None
        matched_alias = None
        for alias in aliases:
            pattern = rf"(?m)^\s*{re.escape(alias)}\s+(\d+)\s+(\d+)\b"
            match = re.search(pattern, table_text)
            if match:
                matched_alias = alias
                break

        if not match:
            continue

        rows.append(
            {
                **metadata,
                "district": district,
                "official_district_name": matched_alias,
                "current_week_cases": int(match.group(1)),
                "previous_week_cases": None,
                "cumulative_cases": int(match.group(2)),
                "pulled_at": pulled_at,
            }
        )

    return rows


def _write_update_log(db_path, status, rows_imported, source_url=None, message=None):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO official_data_updates
            (pulled_at, source, source_url, status, rows_imported, message)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            SOURCE_NAME,
            source_url,
            status,
            int(rows_imported),
            message,
        ),
    )
    conn.commit()
    conn.close()


def save_official_case_rows(db_path, rows):
    if not rows:
        return 0

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    for source_url in {row["source_url"] for row in rows}:
        c.execute(
            "DELETE FROM official_dengue_cases WHERE source_url = ?",
            (source_url,),
        )

    for row in rows:
        c.execute(
            """
            INSERT OR REPLACE INTO official_dengue_cases (
                district,
                official_district_name,
                source,
                source_url,
                report_year,
                report_week,
                report_period,
                current_week_cases,
                previous_week_cases,
                cumulative_cases,
                pulled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["district"],
                row.get("official_district_name"),
                row["source"],
                row["source_url"],
                row["report_year"],
                row["report_week"],
                row.get("report_period"),
                row["current_week_cases"],
                row.get("previous_week_cases"),
                row.get("cumulative_cases"),
                row["pulled_at"],
            ),
        )
    conn.commit()
    conn.close()
    return len(rows)


def update_official_dengue_data(db_path, max_reports=2):
    init_official_data_tables(db_path)

    try:
        links = fetch_epid_report_links(limit=max_reports)
        if not links:
            message = "No Epidemiology Unit WER PDF links found"
            _write_update_log(db_path, "failed", 0, EPID_WER_PAGE, message)
            return {"success": False, "rows_imported": 0, "message": message}

        total_rows = 0
        latest_source_url = links[0]
        messages = []
        for source_url in links:
            try:
                pdf_bytes = _request_get(source_url, timeout=45, binary=True)
                rows = parse_epid_wer_pdf(pdf_bytes, source_url)
                imported = save_official_case_rows(db_path, rows)
                total_rows += imported
                messages.append(f"{imported} rows from {source_url}")
            except Exception as error:
                messages.append(f"{source_url}: {error}")

        success = total_rows > 0
        status = "success" if success else "failed"
        message = "; ".join(messages)
        _write_update_log(db_path, status, total_rows, latest_source_url, message)
        return {
            "success": success,
            "rows_imported": total_rows,
            "source_url": latest_source_url,
            "message": message,
        }
    except Exception as error:
        message = str(error)
        _write_update_log(db_path, "failed", 0, EPID_WER_PAGE, message)
        return {"success": False, "rows_imported": 0, "message": message}


def _row_to_dict(row):
    return dict(row) if row else None


def get_latest_official_cases(db_path, district):
    district = canonical_district_name(district)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT *
        FROM official_dengue_cases
        WHERE district = ?
        ORDER BY report_year DESC, report_week DESC
        LIMIT 1
        """,
        (district,),
    )
    latest = _row_to_dict(c.fetchone())

    if latest:
        c.execute(
            """
            SELECT current_week_cases
            FROM official_dengue_cases
            WHERE district = ?
              AND report_year = ?
              AND report_week < ?
            ORDER BY report_year DESC, report_week DESC
            LIMIT 1
            """,
            (
                district,
                latest["report_year"],
                latest["report_week"],
            ),
        )
        previous = c.fetchone()
        if previous:
            latest["previous_week_cases"] = int(previous["current_week_cases"])
            latest["case_trend"] = (
                int(latest["current_week_cases"]) - int(previous["current_week_cases"])
            )
        else:
            latest["case_trend"] = None

    conn.close()
    return latest


def get_all_latest_official_cases(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT oc.*
        FROM official_dengue_cases oc
        JOIN (
            SELECT district, MAX(report_year || printf('%02d', report_week)) AS max_key
            FROM official_dengue_cases
            GROUP BY district
        ) latest
          ON latest.district = oc.district
         AND latest.max_key = oc.report_year || printf('%02d', oc.report_week)
        ORDER BY oc.district
        """
    )
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def get_official_data_summary(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT report_year, report_week, report_period, source_url, pulled_at
        FROM official_dengue_cases
        ORDER BY report_year DESC, report_week DESC, pulled_at DESC
        LIMIT 1
        """
    )
    latest = c.fetchone()

    if not latest:
        c.execute(
            """
            SELECT pulled_at, status, rows_imported, message
            FROM official_data_updates
            ORDER BY pulled_at DESC
            LIMIT 1
            """
        )
        update_row = c.fetchone()
        conn.close()
        return {
            "available": False,
            "source": SOURCE_NAME,
            "last_update": dict(update_row) if update_row else None,
        }

    latest = dict(latest)
    c.execute(
        """
        SELECT
            COUNT(*) AS districts,
            SUM(current_week_cases) AS national_current_week_cases,
            SUM(cumulative_cases) AS national_cumulative_cases
        FROM official_dengue_cases
        WHERE report_year = ? AND report_week = ?
        """,
        (latest["report_year"], latest["report_week"]),
    )
    counts = dict(c.fetchone())
    c.execute(
        """
        SELECT pulled_at, status, rows_imported, message
        FROM official_data_updates
        ORDER BY pulled_at DESC
        LIMIT 1
        """
    )
    update_row = c.fetchone()
    conn.close()

    return {
        "available": True,
        "source": SOURCE_NAME,
        **latest,
        **counts,
        "last_update": dict(update_row) if update_row else None,
    }


def official_data_is_stale(db_path, max_age_hours=24):
    summary = get_official_data_summary(db_path)
    if not summary.get("available"):
        return True

    pulled_at = summary.get("pulled_at")
    if not pulled_at:
        return True

    try:
        return datetime.now() - datetime.fromisoformat(pulled_at) > timedelta(
            hours=max_age_hours
        )
    except ValueError:
        return True
